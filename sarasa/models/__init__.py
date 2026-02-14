from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger
from torch import nn

if TYPE_CHECKING:
    from sarasa.models.attention import VarlenMetaData


@dataclasses.dataclass
class ModelConfig:
    name: Literal["nanochat_gpt", "llama3"] = "nanochat_gpt"
    """ Name of the model architecture to use.

    - nanochat_gpt: GPT architecture in NanoChat
    - llama3: LLaMA 3 architecture ()
    """

    num_layers: int = 12
    head_dim: int = 128

    num_heads: int | None = None  # inferred later if None
    num_kv_heads: int | None = None  # inferred later if None
    hidden_dim: int | None = None  # inferred later if None
    vocab_size: int | None = None  # set later based on tokenizer
    seq_len: int | None = None  # set later based on data config
    qk_norm: bool = False  # whether to use RMSNorm on q/k
    rms_eps: float | None = None  # epsilon for RMSNorm, default to library default if None
    rms_learnable: bool = False  # whether RMSNorm has learnable scale parameter

    attn_type: Literal["sdpa", "varlen"] = "sdpa"
    """ Attention type, either standard dense attention (sdpa) or variable-length attention (varlen)"""

    extra: dict[str, int | float | bool | str] = dataclasses.field(default_factory=dict)
    """ Extra model-specific configurations. 
    Expected to be used in config files, but can be updated in CLI,
    like --model.extra '{"multiple_of": 1024, "ffn_dim_multiplier": 1.4}'.
    """

    def __post_init__(self):
        # infer hidden_dim, num_heads, num_kv_heads if not provided using the rules presented in nanochat
        self.hidden_dim = self.hidden_dim or (self.num_layers * 64 + self.head_dim - 1) // self.head_dim * self.head_dim
        self.num_heads = self.num_heads or self.hidden_dim // self.head_dim
        self.num_kv_heads = self.num_kv_heads or self.num_heads

        # sanity checks
        assert self.hidden_dim % self.head_dim == 0
        assert self.head_dim * self.num_heads == self.hidden_dim
        assert self.num_kv_heads <= self.num_heads and self.num_heads % self.num_kv_heads == 0

        if self.extra is None:
            self.extra = {}

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


class BaseModel(nn.Module, abc.ABC):
    # Common interface for all models in Sarasa

    blocks: list[nn.Module]  # TF blocks
    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    @torch.no_grad()
    def init_weights(self) -> None:
        # Actual initialization of model weights
        pass

    @abc.abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        # Return parameter groups for optimizer
        pass

    @property
    def num_params_flops(
        self,
    ) -> tuple[int, int]:
        # Return number of parameters and FLOPs per token (for dense model)
        config = self.config

        # for tied embeddings, num_params -= num_params_emb
        num_params = sum(p.numel() for p in self.parameters())
        num_params_emb = self.token_emb.weight.numel()

        # If forward pass has 1 matmul, then backward pass has 2 matmuls
        # Each self-attention has 2 matmuls
        num_flops_per_token = 6 * (
            (num_params - num_params_emb) + (config.num_layers * config.num_heads * config.head_dim * config.seq_len)
        )

        return num_params, num_flops_per_token

    @abc.abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        *,
        metadata: VarlenMetaData | None = None,
    ) -> torch.Tensor:
        # Forward pass of the model, input is (B, T) token ids, output is (B, T, vocab_size) logits
        # metadata is used for, e.g., variable-length attention
        pass
