"""Frozen model runtime for looped-latent: everything except the loop machinery.

Frozen here (identical for every arm, so any gain over B1 is due to the loop): the base model, LoRA r16/alpha 32 on
all 7 projections of all 28 layers, the forward pass (prelude -> loop -> coda -> norm -> lm_head), the loss, AdamW,
the LR schedule, gradient clipping, and the trainable-parameter set. The surface supplies only
  LOOP_START, LOOP_END, LOOP_LR  and  build(d_model, n_layers, seed) -> nn.Module `loop`
whose forward(p, block, ctx) -> h receives the prelude output p (B, T, d) and a frozen callable `block(x)` that runs
the (LoRA-adapted) layers LOOP_START..LOOP_END-1 once on x of exactly p's shape. The loop never sees token ids,
the layers or the base weights. With loop_off=True the frozen code runs block(p) once (the loop-disabled ablation).
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LR, BETAS, WD = 2e-4, (0.9, 0.99), 0.0
WARMUP, MIN_LR_FRAC = 0.05, 0.1        # fractions of the wall-clock budget / of the peak LR (both groups)
GRAD_CLIP = 1.0
MAX_WINDOW = 6


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).t()) @ (self.B.to(x.dtype).t() * self.scale)


def add_lora(base) -> list[nn.Parameter]:
    """Freeze the base and wrap its projections with zero-initialized LoRA (identity at init). Returns the LoRA params."""
    for p in base.parameters():
        p.requires_grad_(False)
    lora = []
    for layer in base.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                if isinstance(getattr(parent, name, None), nn.Linear):
                    w = LoRALinear(getattr(parent, name), LORA_R, LORA_ALPHA)
                    setattr(parent, name, w)
                    lora += [w.A, w.B]
    if len(lora) != 2 * len(LORA_TARGETS) * len(base.model.layers):
        raise RuntimeError("unexpected base architecture: not every LoRA target was found")
    return lora


def base_params(base) -> list[tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in base.named_parameters() if not n.endswith((".A", ".B"))]


def tensor_hash(named) -> str:
    h = hashlib.sha256()
    for n, t in named:
        h.update(n.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


class Ctx:
    """What the loop may use: training flag, budget progress, aux_loss (a scalar it may set), and in training
    readout(h) = the frozen answer loss of an intermediate state h passed through coda + head (deep supervision)."""

    def __init__(self, training: bool, progress: float = 1.0):
        self.training, self.progress, self.aux_loss = training, progress, None
        self.readout = None


class LoopedModel(nn.Module):
    def __init__(self, base, loop: nn.Module, start: int, end: int):
        super().__init__()
        L = base.config.num_hidden_layers
        if not (isinstance(start, int) and isinstance(end, int) and 1 <= start < end <= L - 1 and end - start <= MAX_WINDOW):
            raise ValueError(f"loop window [{start}, {end}) must satisfy 1 <= start < end <= {L - 1}, size <= {MAX_WINDOW}")
        if not isinstance(loop, nn.Module):
            raise TypeError("build() must return an nn.Module")
        self.base, self.loop, self.start, self.end, self.n_layers = base, loop, start, end, L
        self.block_tokens = 0     # tokens pushed through the loop block (all calls)
        self.fwd_tokens = 0       # B*T of every forward

    def _run(self, h, lo, hi, pe, pos):
        for layer in self.base.model.layers[lo:hi]:
            out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def hidden(self, ids, ctx: Ctx, loop_off: bool = False, targets=None):
        m = self.base.model
        B, T = ids.shape
        h = m.embed_tokens(ids)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        pe = m.rotary_emb(h, pos)
        p = self._run(h, 0, self.start, pe, pos)
        self.fwd_tokens += B * T

        def block(x):
            if not isinstance(x, torch.Tensor) or x.shape != p.shape:
                raise ValueError(f"block(x) needs x of shape {tuple(p.shape)}, got {getattr(x, 'shape', type(x))}")
            self.block_tokens += B * T
            return self._run(x.to(p.dtype), self.start, self.end, pe, pos)

        def tail(x):
            return m.norm(self._run(x, self.end, self.n_layers, pe, pos))

        if targets is not None:
            mask = targets != -100
            ctx.readout = lambda x: F.cross_entropy(self.base.lm_head(tail(x)[mask]).float(), targets[mask])
        hl = block(p) if loop_off else self.loop(p, block, ctx)
        if not isinstance(hl, torch.Tensor) or hl.shape != p.shape:
            raise ValueError(f"loop must return a tensor of shape {tuple(p.shape)}")
        return tail(hl.to(p.dtype))

    def forward(self, ids, loop_off: bool = False):
        return self.base.lm_head(self.hidden(ids, Ctx(training=False), loop_off))

    def depth_ratio(self) -> float:
        """Decoder-layer work per token / n_layers, from the frozen block counter."""
        if not self.fwd_tokens:
            return 0.0
        window = self.end - self.start
        return (self.n_layers - window + window * self.block_tokens / self.fwd_tokens) / self.n_layers


def lr_mult(progress: float) -> float:
    if progress < WARMUP:
        return progress / WARMUP
    t = (progress - WARMUP) / (1 - WARMUP)
    return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


class Trainer:
    """Frozen optimizer and train step; the trainable set is fixed to LoRA + the loop's own parameters."""

    def __init__(self, model: LoopedModel, lora: list, loop_lr: float):
        if not (isinstance(loop_lr, (int, float)) and 0 < loop_lr <= 1):
            raise ValueError(f"LOOP_LR must be in (0, 1], got {loop_lr!r}")
        self.model, self.lora = model, lora
        self.loop_params = list(model.loop.parameters())
        ids = {id(p) for p in self.loop_params}
        if ids & {id(p) for _, p in base_params(model.base)} or ids & {id(p) for p in lora}:
            raise ValueError("the loop module must not hold base or LoRA parameters")
        self.allowed = {id(p) for p in lora} | ids
        self.opt = torch.optim.AdamW([{"params": lora, "lr": LR, "peak": LR, "weight_decay": WD},
                                      {"params": self.loop_params, "lr": loop_lr, "peak": loop_lr, "weight_decay": 0.0}],
                                     betas=BETAS)
        self.base_named = base_params(model.base)

    def check_trainable(self):
        opt_ids = {id(p) for g in self.opt.param_groups for p in g["params"]}
        live = {id(p) for p in self.model.parameters() if p.requires_grad}
        if opt_ids != self.allowed or not live <= self.allowed or any(p.requires_grad for _, p in self.base_named):
            raise RuntimeError("trainable parameter set changed: only LoRA and the loop's own parameters may train")

    def step(self, ids, targets, progress: float) -> float:
        self.check_trainable()
        self.model.train()
        ctx = Ctx(training=True, progress=progress)
        mask = targets != -100
        h = self.model.hidden(ids, ctx, targets=targets)[mask]
        loss = F.cross_entropy(self.model.base.lm_head(h).float(), targets[mask])
        total = loss
        if ctx.aux_loss is not None:
            if not (isinstance(ctx.aux_loss, torch.Tensor) and ctx.aux_loss.dim() == 0):
                raise ValueError("ctx.aux_loss must be a scalar tensor")
            total = loss + ctx.aux_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], GRAD_CLIP)
        for g in self.opt.param_groups:
            g["lr"] = g["peak"] * lr_mult(progress)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(total.detach())
