"""A small GPT-style decoder-only Transformer."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128  # maximum context length
    dim: int = 128  # width of every token vector
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.2  # fraction of activations randomly zeroed during training
    norm: str = "layernorm"  # "layernorm" or "rmsnorm"
    pos: str = "learned"  # "learned" position embeddings or "rope" (rotary)

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    """Rescale each token vector to unit root-mean-square, then apply a learned per-channel gain.

    Unlike LayerNorm it does not subtract the mean and has no bias, so it is cheaper.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Rotary(nn.Module):
    """Rotary position embedding (RoPE): rotate query and key vectors by a position-dependent angle.

    The vector is split into two halves that are treated as the x and y coordinates of
    head_dim / 2 points. Point i is rotated by position * theta_i, with theta_i shrinking
    geometrically, so low-index pairs spin fast (nearby positions) and high-index pairs spin slowly.
    A dot product between a rotated query and key depends only on how far apart they are.
    """

    def __init__(self, head_dim, block_size, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head size"
        theta = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # (head_dim/2,)
        angles = torch.arange(block_size).float()[:, None] * theta[None, :]  # (T, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        """x: (B, heads, T, head_dim) -> same shape, rotated."""
        cos, sin = self.cos[: x.shape[-2]], self.sin[: x.shape[-2]]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def make_norm(cfg):
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.dim)
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.dim)
    raise ValueError(f"unknown norm {cfg.norm!r}, expected 'layernorm' or 'rmsnorm'")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token only sees itself and earlier tokens."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)  # queries, keys, values at once
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)  # mixes the heads together
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.rotary = Rotary(self.head_dim, cfg.block_size) if cfg.pos == "rope" else None
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)  # True where looking is allowed

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.rotary is not None:  # position enters only here, through q and k (never v)
            q, k = self.rotary(q), self.rotary(k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # hide the future
        weights = self.attn_drop(F.softmax(scores, dim=-1))

        out = (weights @ v).transpose(1, 2).reshape(B, T, C)  # concatenate heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: expand 4x, non-linearity, shrink back."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    """Transformer block: attention (tokens exchange information), then feed-forward."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.ff = FeedForward(cfg.dim, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)  # token ID -> vector
        if cfg.pos not in ("learned", "rope"):
            raise ValueError(f"unknown pos {cfg.pos!r}, expected 'learned' or 'rope'")
        # position -> vector; RoPE has no such table because it rotates q and k instead
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # share weights with the embedding
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: (B, T) token IDs. Returns logits (B, T, vocab) and the loss if targets are given."""
        T = idx.shape[1]
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Extend `idx` (B, T) by sampling one token at a time."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # scores for the next token
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
