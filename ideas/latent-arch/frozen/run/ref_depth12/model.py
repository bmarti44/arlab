"""latent-arch REFERENCE depth12 (frozen, reported only): the baseline GPT at 12 layers x 384.

Ported from karpathy/autoresearch train.py (MIT License, Copyright (c) 2026 Andrej Karpathy) with GB10 settings
from mazar/autoresearch-spark (numbers only): SDPA, full attention, depth 6 / width 384, value embeddings.
Any causal architecture in pure PyTorch may replace it; train.py builds it with model_config() and GPT(config),
and the evaluator calls forward(idx) -> logits (B, T, vocab_size).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH = 12       # layers (reference: 2x the baseline depth, same width, same 330 s)
N_EMBD = 384     # width (m3b: DEPTH * ASPECT_RATIO 64, rounded up to HEAD_DIM)
HEAD_DIM = 128


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 3
    n_kv_head: int = 3
    n_embd: int = 384


def model_config(vocab_size, seq_len):
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=DEPTH, n_head=N_EMBD // HEAD_DIM,
                     n_kv_head=N_EMBD // HEAD_DIM, n_embd=N_EMBD)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head, self.n_kv_head, self.n_embd = config.n_head, config.n_kv_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:  # value residual (ResFormer) with an input-dependent gate per head
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           enable_gqa=self.n_kv_head != self.n_head)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(config.vocab_size, config.n_embd),
                                          "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, kv_dim)
                                           for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            for w in (block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.mlp.c_fc.weight):
                torch.nn.init.uniform_(w, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        self.to_bf16_embeddings()

    def to_bf16_embeddings(self):
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._rotary(self.rotary_seq_len, head_dim)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
        return freqs.cos().bfloat16()[None, :, None, :], freqs.sin().bfloat16()[None, :, None, :]

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        softcap = 15
        logits = self.lm_head(norm(x)).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=reduction)
        return logits
