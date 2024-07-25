from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal
from llms.lm_config import LMConfig

@dataclass(frozen=True)
class EmbeddingConfig:
    k_threshold:int
    model_name_or_path:str
    tokenizer_name_or_path:str
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class RetrieverConfig:
    """A config for a retriever system with different modes.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model.
        tokenizer_cls: The Python class corresponding to the tokenizer.
        mode: The mode of the retriever, can be 'PromptBased' or 'EmbeddingBased'.
        prompt_config: Configuration specific to the PromptBased mode.
        embedding_config: Configuration specific to the EmbeddingBased mode.
    """
    
    retriever_mode: int # 0: Fake; 1: PromptBased 2: EmbeddingBased

    # for PromptBased:
    llm_config: LMConfig

    # for EmbeddingBased:
    embedding_config: EmbeddingConfig

def construct_retriever_config(args:argparse.Namespace) -> RetrieverConfig:
    retriever_config = RetrieverConfig(
        retriever_mode=args.r_mode
    )
    llm_config = LMConfig()
    embedding_config = EmbeddingConfig()

    if args.r_mode == 0:
        pass
    elif args.r_mode == 1:
        if args.r_provier == "openai":
            llm_config.gen_config["temperature"] = args.r_temperature
            llm_config.gen_config["top_p"] = args.r_top_p
            llm_config.gen_config["context_length"] = args.r_context_length
            llm_config.gen_config["max_tokens"] = args.r_max_tokens
            llm_config.gen_config["stop_token"] = args.r_stop_token
            llm_config.gen_config["max_obs_length"] = args.r_max_obs_length
            llm_config.gen_config["max_retry"] = args.r_max_retry
        elif args.r_provider == "huggingface":
            llm_config.gen_config["temperature"] = args.r_temperature
            llm_config.gen_config["top_p"] = args.r_top_p
            llm_config.gen_config["max_new_tokens"] = args.r_max_tokens
            llm_config.gen_config["stop_sequences"] = (
                [args.r_stop_token] if args.r_stop_token else None
            )
            llm_config.gen_config["max_obs_length"] = args.r_max_obs_length
            llm_config.gen_config["model_endpoint"] = args.r_model_endpoint
            llm_config.gen_config["max_retry"] = args.r_max_retry
        else:
            raise NotImplementedError(f"provider {args.r_provider} not implemented")
    elif args.r_mode == 1:
        embedding_config.k_threshold = args.r_k_threshold
        embedding_config.model_name_or_path = args.r_model_name_or_path
        embedding_config.tokenizer_name_or_path = args.r_tokenizer_name_or_path
        embedding_config.gen_config["max_length"] = args.r_max_length
        embedding_config.gen_config["padding"] = args.r_padding
        embedding_config.gen_config["truncation"] = args.r_truncation
    
    retriever_config.llm_config = llm_config
    retriever_config.embedding_config = embedding_config

    return retriever_config