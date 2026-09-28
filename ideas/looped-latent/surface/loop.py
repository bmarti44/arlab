"""Editable surface: a looped mid-block retrofit of Qwen3-0.6B with LoRA adapters.

BASELINE STATE = the no-loop arm (LOOPS_TRAIN = (1,), LOOPS_EVAL = 1): same adapters, optimizer and wall-clock
budget as any loop arm, so every keep must beat the compute-matched no-loop fine-tune.

Forward:  prelude = layers[:LOOP_START]; block = layers[LOOP_START:LOOP_END]; coda = layers[LOOP_END:]
    p = prelude(embed(ids))
    h = block(p)                                     # pass 1 = the original model
    for j in 2..K:  h = h + gate_j * (block(h + inj(p)) - h)
    logits = lm_head(norm(coda(h)))
gate_j and inj start at zero, so ANY K reproduces the base model exactly at init (identity-at-init).
The frozen harness calls: build(base, config) -> state, train_step(state, ids, targets, progress) -> float,
logits(state, ids) -> (B, T, V). state["model"] must be the nn.Module holding every parameter (base included).
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

LOOP_START, LOOP_END = 12, 16      # loop block = layers 12..15 of 28
LOOPS_TRAIN = (1,)                 # K sampled uniformly from this tuple each step (e.g. (1, 2, 3, 4))
LOOPS_EVAL = 1                     # K at evaluation (inference cost guard: effective depth <= 2x)
K_MAX = 8                          # gates exist for K <= K_MAX
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LR, LOOP_LR, WD = 2e-4, 1e-2, 0.0  # LOOP_LR: gates + input injection
WARMUP, MIN_LR_FRAC = 0.05, 0.1    # fractions of the wall-clock budget / of peak LR
GRAD_CLIP = 1.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).t()) @ (self.B.to(x.dtype).t() * self.scale)


class Looped(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        for layer in base.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                for name in LORA_TARGETS:
                    if isinstance(getattr(parent, name, None), nn.Linear):
                        setattr(parent, name, LoRALinear(getattr(parent, name), LORA_R, LORA_ALPHA))
        d, dev = base.config.hidden_size, base.model.embed_tokens.weight.device
        self.gates = nn.Parameter(torch.zeros(K_MAX - 1, d, device=dev))  # gate_j for passes j = 2..K_MAX
        self.inj = nn.Linear(d, d, bias=False, device=dev)
        nn.init.zeros_(self.inj.weight)

    def _layers(self, h, lo, hi, pe, pos):
        for layer in self.base.model.layers[lo:hi]:
            out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def hidden(self, ids, k):
        m = self.base.model
        h = m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        pe = m.rotary_emb(h, pos)
        p = self._layers(h, 0, LOOP_START, pe, pos)
        h = self._layers(p, LOOP_START, LOOP_END, pe, pos)
        for j in range(1, k):
            g = self._layers(h + self.inj(p.to(self.inj.weight.dtype)).to(h.dtype), LOOP_START, LOOP_END, pe, pos)
            h = h + self.gates[j - 1].to(h.dtype) * (g - h)
        h = self._layers(h, LOOP_END, len(m.layers), pe, pos)
        return m.norm(h)

    def forward(self, ids, k=None):
        return self.base.lm_head(self.hidden(ids, LOOPS_EVAL if k is None else k))


def build(base, config):
    model = Looped(base)
    loop = [model.gates, model.inj.weight]
    lora = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    groups = [{"params": lora, "lr": LR, "weight_decay": WD}, {"params": loop, "lr": LOOP_LR, "weight_decay": 0.0}]
    for g in groups:
        g["peak_lr"] = g["lr"]
    return {"model": model, "opt": torch.optim.AdamW(groups, betas=(0.9, 0.99)), "rng": random.Random(config["seed"])}


def lr_mult(progress):
    if progress < WARMUP:
        return progress / WARMUP
    t = (progress - WARMUP) / max(1e-9, 1 - WARMUP)
    return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def train_step(state, ids, targets, progress):
    """One optimizer step on a right-padded batch; targets[t] is the token after ids[t] or -100 (prompt/pad)."""
    model, opt = state["model"], state["opt"]
    model.train()
    k = state["rng"].choice(LOOPS_TRAIN)
    mask = targets != -100
    h = model.hidden(ids, k)[mask]
    loss = F.cross_entropy(model.base.lm_head(h).float(), targets[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], GRAD_CLIP)
    for g in opt.param_groups:
        g["lr"] = g["peak_lr"] * lr_mult(progress)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


@torch.no_grad()
def logits(state, ids):
    state["model"].eval()
    return state["model"](ids)
