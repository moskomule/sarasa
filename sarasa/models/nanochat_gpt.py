# NanoChat's GPT model, adapted from https://github.com/karpathy/nanochat

import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from sarasa.models.base import BaseModel


def rms_norm(
    x: torch.Tensor,
) -> torch.Tensor:
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        n_embd: int,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # QK rotary embedding
        q, k = rms_norm(q), rms_norm(k)  # QK norm
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=self.n_head != self.n_kv_head,
        )

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
    ):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        n_embd: int,
        layer_idx: int,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(n_head, n_kv_head, n_embd, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(BaseModel):
    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        vocab_size: int,
        seq_len: int,
        pad_vocab_size_to=64,
    ):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != vocab_size:
            logger.info(
                f"Padding vocab_size from {vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}"
            )
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, hidden_dim),
            "h": nn.ModuleList([
                Block(num_heads, num_kv_heads, hidden_dim, layer_idx) for layer_idx in range(num_layers)
            ]),
        })
        self.blocks = self.transformer.h  # for BaseModel interface
        self.lm_head = nn.Linear(hidden_dim, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(num_layers))  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))  # fake init, real init in init_weights()
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = seq_len * 10  # 10X over-compute should be enough, TODO make nicer?
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.hidden_dim
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
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
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, the term 12 * l * h * q * t accounts for key @ query matmul flops inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.num_layers,
            self.num_heads,
            self.hidden_dim // self.num_heads,
            self.seq_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def param_groups(
        self,
    ) -> dict[str, list[torch.nn.Parameter]]:
        # Separate out all parameters into 5 groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
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
        x = self.transformer.wte(input)
        x = rms_norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin)
        x = rms_norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x)  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        return logits
