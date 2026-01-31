import torch
from torch import nn
from torch.nn import functional as F

from sarasa.models import BaseModel, ModelConfig
from sarasa.models.attention import CausalSelfAttention
from sarasa.models.utils import RMSNorm, RoPE


class MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(8 * config.hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(config.hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config, multiple_of, ffn_dim_multiplier)
        self.norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attention(self.norm(x), cos_sin)
        x = x + self.mlp(self.norm(x))
        return x


class Llama3(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        multiple_of: int = 1024,
        ffn_dim_multiplier: float | None = 1.4,
    ):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.max_seq_len = config.seq_len * 16
        self.head_dim = config.head_dim
        cos, sin = RoPE.precompute(self.max_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([
            Block(config, layer_idx, multiple_of, ffn_dim_multiplier) for layer_idx in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        self.cos, self.sin = RoPE.precompute(self.max_seq_len, self.head_dim, device=self.cos.device)
        torch.nn.init.normal_(self.token_emb.weight)
        for block in self.blocks:
            block: Block
            init_std = 0.02 / (2 * (block.layer_idx + 1)) ** 0.5

            nn.init.trunc_normal_(block.attention.c_q.weight, std=0.02)
            nn.init.trunc_normal_(block.attention.c_k.weight, std=0.02)
            nn.init.trunc_normal_(block.attention.c_v.weight, std=0.02)
            nn.init.trunc_normal_(block.attention.c_proj.weight, std=init_std)

            nn.init.trunc_normal_(block.mlp.w1.weight, std=0.02)
            nn.init.trunc_normal_(block.mlp.w2.weight, std=init_std)
            nn.init.trunc_normal_(block.mlp.w3.weight, std=init_std)

        final_out_std = self.output.weight.shape[-1] ** -0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.token_emb.parameters())
        lm_head_params = list(self.output.parameters())
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) + len(lm_head_params))

        return {
            "matrix": matrix_params,
            "embedding": embedding_params,
            "lm_head": lm_head_params,
        }

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        B, T = input.size()
        x = self.token_emb(input)  # (B, T, C)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        for block in self.blocks:
            x = block(x, cos_sin)

        x = self.norm(x)
        logits = self.output(x)
        return logits
