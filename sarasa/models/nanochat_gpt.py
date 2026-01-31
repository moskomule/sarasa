# NanoChat's GPT model, adapted from https://github.com/karpathy/nanochat


import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from sarasa.models import BaseModel, ModelConfig
from sarasa.models.attention import CausalSelfAttention
from sarasa.models.utils import RMSNorm, RoPE


class MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.c_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm(x), cos_sin)
        x = x + self.mlp(self.norm(x))
        return x


class GPT(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        pad_vocab_size_to=64,
    ):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.seq_len
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((self.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != self.vocab_size:
            logger.warning(
                f"Padding vocab_size from {self.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}"
            )
        self.token_emb = nn.Embedding(padded_vocab_size, self.hidden_dim)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(self.num_layers)])
        self.lm_head = nn.Linear(self.hidden_dim, padded_vocab_size, bias=False)
        self.norm = RMSNorm(self.hidden_dim)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(self.num_layers))  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(self.num_layers))  # fake init, real init in init_weights()
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = self.seq_len * 16  # 10X over-compute should be enough, TODO make nicer?
        cos, sin = RoPE.precompute(self.rotary_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.hidden_dim
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)  # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)  # 0.0 => skip connection to input is disabled at init

        # Rotary embeddings
        head_dim = self.hidden_dim // self.num_heads
        self.cos, self.sin = RoPE.precompute(self.rotary_seq_len, head_dim, device=self.cos.device)

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.token_emb.weight.device.type == "cuda":
            self.token_emb.to(dtype=torch.bfloat16)

    def param_groups(
        self,
    ) -> dict[str, list[torch.nn.Parameter]]:
        # Separate out all parameters into 5 groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas)
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.token_emb.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
        )

        return {
            "matrix": matrix_params,
            "embedding": embedding_params,
            "lm_head": lm_head_params,
            "resid_lambdas": resid_params,
            "x0_lambdas": x0_params,
        }

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        B, T = input.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert input.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {input.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        cos_sin = self.cos[:, :T], self.sin[:, :T]  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.token_emb(input)
        x = self.norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for block, resid_lambda, x0_lambda in zip(self.blocks, self.resid_lambdas, self.x0_lambdas):
            x = resid_lambda * x + x0_lambda * x0
            x = block(x, cos_sin)
        x = self.norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x)  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        return logits
