import torch
from torch import nn
from torch.nn import functional as F

from sarasa.models import BaseModel, ModelConfig
from sarasa.models.attention import CausalSelfAttention
from sarasa.models.utils import RoPE
from sarasa.models.varlen import VarlenMetaData


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
        self.attn_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), cos_sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Llama3(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__(config)
        multiple_of = config.extra.get("multiple_of", 1024)
        ffn_dim_multiplier = config.extra.get("ffn_dim_multiplier", 1.4)

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.max_seq_len = config.seq_len * 16
        self.head_dim = config.head_dim
        cos, sin = RoPE.precompute(self.max_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList([
            Block(config, layer_idx, multiple_of, ffn_dim_multiplier) for layer_idx in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_eps)
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

        for mod in self.modules():
            if isinstance(mod, nn.RMSNorm):
                mod.reset_parameters()

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        matrix_params = [param for param in self.blocks.parameters() if param.ndim == 2]
        embedding_params = list(self.token_emb.parameters())
        lm_head_params = list(self.output.parameters())
        rms_norm_params = [
            mod.weight for mod in self.modules() if isinstance(mod, nn.RMSNorm) and mod.elementwise_affine
        ]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(rms_norm_params)
        )

        return {
            "matrix": matrix_params,
            "embedding": embedding_params,
            "lm_head": lm_head_params,
            "rms_norm": rms_norm_params,
        }

    def forward(
        self,
        input: torch.Tensor,
        *,
        metadata: VarlenMetaData | None = None,
    ) -> torch.Tensor:
        B, T = input.size()
        x = self.token_emb(input)  # (B, T, C)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        for block in self.blocks:
            x = block(x, cos_sin)

        x = self.norm(x)
        logits = self.output(x)
        return logits
