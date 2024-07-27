from typing import Any

import tiktoken
from transformers import LlamaTokenizer,AutoTokenizer  # type: ignore
import os
from dotenv import load_dotenv
load_dotenv()


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            if model_name == "llama3:8b" or model_name == "llama2:7b" or model_name == "qwen2:7b": # same as gpt4
                self.tokenizer = tiktoken.encoding_for_model('gpt-4')
            else:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
