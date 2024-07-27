import json
from typing import Any
import argparse
from beartype import beartype
from agent import(
    PromptConstructor,
    CoTPromptConstructor,
    DirectPromptConstructor
)
from llms import (
    Tokenizer,
    call_llm,
    lm_config,
    APIInput
)
from browser_env import Trajectory,StateInfo
import torch.nn.functional as F
import torch

from retriever import retriever_config
from retriever.retriever_config import EmbeddingConfig
from retriever.utils import last_token_pool
from transformers import AutoTokenizer, AutoModel

class Retriever:
    def __init__(self, *args: Any) -> None:
        self.meta_data = {}
        pass

    def simplify_observation(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    )->str:
        raise NotImplementedError

    def get_retriever_meta_data(self):
        return self.meta_data

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError

class FakeRetriever(Retriever):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
    
    def simplify_observation(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    )->str:
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"]["text"]
        self.meta_data["obs_before_after"] = obs
        return obs
    
    def reset(
        self,
        test_config_file: str,
    ) -> None:
        return

class PromptBasedRetriever(Retriever):
    def __init__(
        self,
        lm_config:lm_config.LMConfig,
        prompt_constructor:PromptConstructor
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor

    @property
    def __prompt_constructor__(self)->PromptConstructor:
        return self.prompt_constructor

    @property
    def __current_prompt__(self)->APIInput:
        return self.current_prompt if self.current_prompt else None

    @beartype
    def simplify_observation(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    )->str:
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        self.current_prompt = prompt # modified
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            self.current_response = response
            print(response)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            self.meta_data["response"] = response
            n += 1
            try:
                observation = self.prompt_constructor.extract_observation(response)
                break
            except ValueError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    observation = trajectory[-1]["observation"][self.prompt_constructor.obs_modality]
                    break
        self.meta_data["obs_before"] = trajectory[-1]["observation"][self.prompt_constructor.obs_modality]
        self.meta_data["obs_after"] = observation
        return observation

    def reset(self, test_config_file: str) -> None:
        pass


class EmbeddingBasedRetriever(Retriever):
    def __init__(
        self,
        embedding_config:EmbeddingConfig
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_config.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(embedding_config.tokenizer_name_or_path, trust_remote_code=True)
        
        self.embedding_config = embedding_config
        return
    
    def simplify_observation(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    )->str:
        reflexion = meta_data["reflexion"]
        obs_nodes_str_list = []

        state_info:StateInfo = trajectory[-1] # type: ignore[assignment]
        tab_title_str = state_info["info"]["observation_metadata"]["text"]["tab_title_str"]
        obs_nodes_info = state_info["info"]["observation_metadata"]["text"]["obs_nodes_info"]

        obs_nodes_pair_list = sorted(obs_nodes_info.items(),key=lambda x:int(x[0])) # sorted by 'id'
        for idx,obs_node in obs_nodes_pair_list:
            obs_nodes_str_list.append(obs_node["text"])
        
        query_list = [self._get_detailed_instruct(intent,reflexion)]
        # get embeddings
        input_texts = query_list + obs_nodes_str_list
        batch_dict = self.tokenizer(
            input_texts, 
            max_length=self.embedding_config.gen_config["max_length"], 
            padding=self.embedding_config.gen_config["padding"], 
            truncation=self.embedding_config.gen_config["truncation"], 
            return_tensors='pt'
        )
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        m = len(query_list)
        scores = (embeddings[:m] @ embeddings[m:].T) * 100

        k = self.embedding_config.k_threshold if self.embedding_config.k_threshold < len(obs_nodes_pair_list) else len(obs_nodes_pair_list)
        
        topk_scores, topk_indices = torch.topk(scores.flatten(), k, dim=0, largest=True, sorted=True)
        
        topk_backend_ids = [obs_nodes_pair_list[idx][0] for idx in topk_indices.tolist()]
        topk_texts = [obs_nodes_pair_list[idx][1]["text"] for idx in topk_indices.tolist()]
        
        sorted_pairs = sorted(zip(topk_backend_ids, topk_texts), key=lambda x: x[0])

        nodes_text_list = [text for _,text in sorted_pairs]

        observation = tab_title_str + '\n\n' + '\n\t'.join(nodes_text_list) 
        
        self.meta_data["obs_before"] = trajectory[-1]["observation"]["text"]
        self.meta_data["obs_after"] = observation
        self.meta_data["scores"] = scores.tolist()
        return observation
    
    def reset(self, test_config_file: str) -> None:
        pass
    
    def _get_detailed_instruct(self,intent:str,reflexion:str) -> str:
        if not reflexion:
            return f"OBJECTIVE: {intent}"
        else:
            return f"OBJECTIVE: {intent}\nREFLEXION: {reflexion}"
    
def construct_retriever(args: argparse.Namespace) -> Retriever:
    r_config = retriever_config.construct_retriever_config(args)
    llm_config = r_config.llm_config

    retriever: Retriever

    if args.retriever_mode == 0:
        retriever = FakeRetriever()
    elif args.retriever_mode == 1:
        with open(args.r_instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.r_provider, args.r_model)
        prompt_constructor = eval(constructor_type)(
            args.r_instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        retriever = PromptBasedRetriever(
            lm_config=llm_config,
            prompt_constructor=prompt_constructor
        )
    elif args.retriever_mode == 2:
        retriever = EmbeddingBasedRetriever(
            r_config.embedding_config
        )
        pass
    else:
        raise ValueError(f"Unknown retriever mode: '{args.retriever_mode}'.")

    return retriever