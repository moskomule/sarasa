import abc

import torch
from torch import nn

from sarasa.models import ModelConfig


class BaseModel(nn.Module, abc.ABC):
    # Common interface for all models in Sarasa

    blocks: list[nn.Module]  # TF blocks
    config: ModelConfig

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
