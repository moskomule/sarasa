import torch
from torch import nn
from torch.nn import functional as F


class SDPAttention(nn.Module):
    def __init__(
        self,
        is_causal: bool,
        enable_gqa: bool,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa

        if nn.attention.current_flash_attention_impl() == "FA4":
            self.sdpa_backends = nn.attention.SDPBackend.FLASH_ATTENTION
        else:
            self.sdpa_backends = [
                nn.attention.SDPBackend.CUDNN_ATTENTION,
                nn.attention.SDPBackend.FLASH_ATTENTION,
                nn.attention.SDPBackend.MATH,
            ]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        with nn.attention.sdpa_kernel(self.sdpa_backends):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self.is_causal,
                enable_gqa=self.enable_gqa,
            )
