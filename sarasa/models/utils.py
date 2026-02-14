from typing import Callable

import torch
from torch import nn


class RoPE:
    @staticmethod
    def precompute(
        seq_len: int,
        head_dim: int,
        device: torch.device = None,
        base: float = 10000,
    ):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)[None, :, None, :]
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos, sin

    @staticmethod
    def apply(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == 4
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3)


def create_varlen_metadata_prehook(
    bos_token_id: int,
) -> Callable[[nn.Module, tuple, dict], tuple[tuple, dict]]:
    """Create a forward pre-hook to prepare VarlenMetaData for variable-length attention.
    The returned pre-hook rewrites `metadata` in kwargs.
    """

    from sarasa.models.attention import VarlenMetaData

    def prepare_varlen_metadata_prehook[A, K](
        module: nn.Module,
        args: A,
        kwargs: K,
    ) -> tuple[A, K]:
        if kwargs.get("metadata") is not None:
            return args, kwargs

        if args[0].device.type == "meta":
            # need a dummy metadata for shape inference
            kwargs["metadata"] = VarlenMetaData(
                cu_seq_q=torch.zeros(1, dtype=torch.int64),
                cu_seq_k=torch.zeros(1, dtype=torch.int64),
                max_q=1,
                max_k=1,
            )
            return args, kwargs

        input = args[0]  # (B, T)
        # flatten it to 1D
        (bos_positions,) = (input.flatten() == bos_token_id).nonzero(as_tuple=True)
        cu_seq = torch.cat((
            bos_positions,
            bos_positions.new_tensor([input.size(0) * input.size(1)]),
        ))
        max_seqlen = torch.diff(cu_seq).max()
        metadata = VarlenMetaData(
            cu_seq_q=cu_seq,
            cu_seq_k=cu_seq,
            max_q=max_seqlen,
            max_k=max_seqlen,
        )
        kwargs["metadata"] = metadata
        return args, kwargs

    return torch.compiler.disable(prepare_varlen_metadata_prehook)
