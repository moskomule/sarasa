import torch


class RMSNorm(torch.nn.RMSNorm):
    # RMSNorm without affine parameters
    def __init__(
        self,
        normalized_shape: int,
    ):
        super().__init__(normalized_shape, eps=None, elementwise_affine=False)


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
