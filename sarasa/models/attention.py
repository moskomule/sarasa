from __future__ import annotations

import inspect
from functools import partial
from typing import Callable, ClassVar, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.varlen import varlen_attn as _varlen_attn
from torch.types import Number

from sarasa.models import ModelConfig
from sarasa.models.utils import RoPE

if inspect.signature(_varlen_attn).parameters.get("window_size") is not None:
    # torch>2.10
    # after the release of torch 2.11, this branch can be removed
    varlen_attn = partial(_varlen_attn, window_size=(-1, 0))
else:
    # torch==2.10
    varlen_attn = partial(_varlen_attn, is_causal=True)


class VarlenMetaData(NamedTuple):
    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: Number
    max_k: Number

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
    ) -> VarlenMetaData:
        return self._replace(
            cu_seq_q=self.cu_seq_q.to(device, non_blocking=non_blocking),
            cu_seq_k=self.cu_seq_k.to(device, non_blocking=non_blocking),
        )


class VarlenAttention(nn.Module):
    compiled_varlen: ClassVar[Callable] = torch.compile(
        varlen_attn,
        mode="max-autotune-no-cudagraphs",
    )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        metadata: VarlenMetaData,
    ) -> torch.Tensor:

        out = VarlenAttention.compiled_varlen(
            query.transpose(1, 2).flatten(0, 1),  # (B*T, num_heads, head_dim)
            key.transpose(1, 2).flatten(0, 1),
            value.transpose(1, 2).flatten(0, 1),
            cu_seq_q=metadata.cu_seq_q.view(-1),
            cu_seq_k=metadata.cu_seq_k.view(-1),
            max_q=int(metadata.max_q),
            max_k=int(metadata.max_k),
        )  # (B*T, num_heads, head_dim)
        return out.reshape(query.size(0), query.size(2), -1)  # (B, T, num_heads * head_dim)


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
        *,
        metadata=None,
    ) -> torch.Tensor:
        assert metadata is None
        with nn.attention.sdpa_kernel(self.sdpa_backends):
            out = F.scaled_dot_product_attention(
                query,  # (B, num_heads, T, head_dim)
                key,
                value,
                is_causal=self.is_causal,
                enable_gqa=self.enable_gqa,
            )
        return out.transpose(1, 2).reshape(query.size(0), query.size(2), -1)  # (B, T, num_heads * head_dim)


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
        self.qk_norm = (
            nn.RMSNorm(self.head_dim, eps=config.rms_eps, elementwise_affine=config.rms_learnable)
            if config.qk_norm
            else nn.Identity()
        )

        if config.attn_type == "varlen":
            self.attn = VarlenAttention()
        else:
            self.attn = SDPAttention(is_causal=True, enable_gqa=self.num_heads != self.num_kv_heads)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        *,
        metadata: VarlenMetaData | None = None,
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
        y = self.attn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            metadata=metadata,
        )  # (B, T, num_heads * head_dim)
        y = self.c_proj(y)
        return y
