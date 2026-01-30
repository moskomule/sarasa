import dataclasses
from typing import Literal

from loguru import logger

from .base import BaseModel as BaseModel


@dataclasses.dataclass
class ModelConfig:
    name: Literal["nanochat_gpt", "llama3"] = "nanochat_gpt"
    num_layers: int = 12
    head_dim: int = 128

    num_heads: int | None = None  # inferred later if None
    num_kv_heads: int | None = None  # inferred later if None
    hidden_dim: int | None = None  # inferred later if None
    vocab_size: int | None = None  # set later based on tokenizer
    seq_len: int | None = None  # set later based on data config
    qk_norm: bool = False  # whether to use RMSNorm on q/k

    def __post_init__(self):
        # infer hidden_dim, num_heads, num_kv_heads if not provided using the rules presented in nanochat
        self.hidden_dim = self.hidden_dim or (self.num_layers * 64 + self.head_dim - 1) // self.head_dim * self.head_dim
        self.num_heads = self.num_heads or self.hidden_dim // self.head_dim
        self.num_kv_heads = self.num_kv_heads or self.num_heads

        # sanity checks
        assert self.hidden_dim % self.head_dim == 0
        assert self.head_dim * self.num_heads == self.hidden_dim
        assert self.num_kv_heads <= self.num_heads and self.num_heads % self.num_kv_heads == 0

    def create(self) -> BaseModel:
        if self.vocab_size is None or self.seq_len is None:
            raise ValueError("vocab_size and seq_len must be set before creating the model")

        match self.name:
            case "nanochat_gpt":
                from .nanochat_gpt import GPT

                if not self.qk_norm:
                    logger.warning("nanochat_gpt model without qk_norm is not recommended")

                return GPT(self)

            case "llama3":
                from .llama3 import Llama3

                if self.qk_norm:
                    logger.warning("llama3 model with qk_norm is not standard")

                return Llama3(self)

            case _:
                raise ValueError(f"Unknown model name: {self.name}")
