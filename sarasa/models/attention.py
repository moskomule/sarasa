import torch
from torch import nn
from torch.nn import functional as F

from sarasa.models import ModelConfig
from sarasa.models.utils import RMSNorm, RoPE


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


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads
        self.c_q = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.qk_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()

        # todo: support varlen etc and kv caching
        self.attn = SDPAttention(is_causal=True, enable_gqa=self.num_heads != self.num_kv_heads)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = RoPE.apply(q, cos, sin), RoPE.apply(k, cos, sin)
        q, k = self.qk_norm(q), self.qk_norm(k)
        y = self.attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
