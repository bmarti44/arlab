"""Frozen minimal LoRA for the Qwen3 decoder (no peft dependency): attach / detach for training, merge / unmerge for
evaluation, and a safetensors file format. The harness and the evaluator both use it, so an adapter means the same
thing in RUN and EVALUATE."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MAX_RANK = 64


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, gen: torch.Generator | None = None):
        super().__init__()
        self.base, self.r, self.scale, self.enabled = base, r, alpha / r, True
        dev = base.weight.device
        a = torch.randn(r, base.in_features, generator=gen) / math.sqrt(base.in_features)
        self.A = nn.Parameter(a.to(dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (x @ self.A.to(x.dtype).t()) @ (self.B.to(x.dtype).t() * self.scale)


def _parents(model):
    for li, layer in enumerate(model.model.layers):
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if hasattr(parent, name):
                    yield li, parent, name


def _int(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"{what} must be an int, got {x!r}")
    return x


def check_config(cfg: dict) -> dict:
    r = _int(cfg["rank"], "LoRA rank")
    if not 1 <= r <= MAX_RANK:
        raise ValueError(f"LoRA rank must be in [1, {MAX_RANK}]")
    if isinstance(cfg["targets"], str):
        raise ValueError("LoRA targets must be a list of names")
    targets = tuple(cfg["targets"])
    if not targets or not set(targets) <= set(TARGETS) or len(set(targets)) != len(targets):
        raise ValueError(f"LoRA targets must be a non-empty subset of {TARGETS}")
    alpha = cfg["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 1024:
        raise ValueError("LoRA alpha must be a finite number in (0, 1024]")
    layers = cfg.get("layers")
    if layers is not None:
        layers = sorted({_int(x, "layer index") for x in layers})
        if not layers or layers[0] < 0 or layers[-1] > 255:
            raise ValueError("layers must be a non-empty list of layer indices")
    return {"rank": r, "alpha": float(alpha), "targets": [t for t in TARGETS if t in targets], "layers": layers}


def attach(model, cfg: dict, seed: int) -> dict[str, LoRALinear]:
    """Wrap the targeted nn.Linear modules in place. Returns {qualified name: LoRALinear}."""
    cfg = check_config(cfg)
    gen = torch.Generator().manual_seed(seed)
    mods = {}
    for li, parent, name in _parents(model):
        if name in cfg["targets"] and (cfg["layers"] is None or li in cfg["layers"]):
            lin = getattr(parent, name)
            assert isinstance(lin, nn.Linear), f"layer {li} {name} is already wrapped"
            m = LoRALinear(lin, cfg["rank"], cfg["alpha"], gen)
            setattr(parent, name, m)
            mods[f"{li}.{name}"] = m
    return mods


def detach(model) -> None:
    """Remove every LoRALinear (restores the original nn.Linear objects; base weights were never touched)."""
    for _, parent, name in _parents(model):
        m = getattr(parent, name)
        if isinstance(m, LoRALinear):
            setattr(parent, name, m.base)


def n_wrapped(model) -> int:
    return sum(isinstance(getattr(p, n), LoRALinear) for _, p, n in _parents(model))


@contextmanager
def disabled(mods: dict):
    """Temporarily run the base model (for KL-to-base targets)."""
    for m in mods.values():
        m.enabled = False
    try:
        yield
    finally:
        for m in mods.values():
            m.enabled = True


def tensors(mods: dict) -> dict[str, torch.Tensor]:
    out = {}
    for k, m in mods.items():
        out[f"{k}.A"] = m.A.detach().float().cpu().contiguous()
        out[f"{k}.B"] = m.B.detach().float().cpu().contiguous()
    return out


def save(path: str, cfg: dict, tens: dict) -> None:
    os.makedirs(path, exist_ok=True)
    save_file(tens, f"{path}/adapter.safetensors")
    json.dump(check_config(cfg), open(f"{path}/adapter.json", "w"))


def load(path: str) -> tuple[dict, dict]:
    return check_config(json.load(open(f"{path}/adapter.json"))), load_file(f"{path}/adapter.safetensors")


def validate(model, cfg: dict, tens: dict) -> str | None:
    """None if the adapter matches the model and is finite; else the reason."""
    want = set()
    lin = {f"{li}.{n}": getattr(p, n) for li, p, n in _parents(model)}
    for li, _, n in _parents(model):
        if n in cfg["targets"] and (cfg["layers"] is None or li in cfg["layers"]):
            want |= {f"{li}.{n}.A", f"{li}.{n}.B"}
    if set(tens) != want:
        return f"adapter tensors do not match its config ({len(tens)} vs {len(want)})"
    for k, t in tens.items():
        base = lin[k.rsplit(".", 1)[0]]
        shape = (cfg["rank"], base.in_features) if k.endswith(".A") else (base.out_features, cfg["rank"])
        if tuple(t.shape) != shape:
            return f"{k}: shape {tuple(t.shape)} != {shape}"
        if not torch.isfinite(t).all():
            return f"{k}: non-finite values"
    return None


def n_params(tens: dict) -> int:
    return sum(t.numel() for t in tens.values())


class Merged:
    """Merge an adapter into the base weights (W += scale * B @ A in float32) and restore them exactly afterwards."""

    def __init__(self, model, cfg: dict, tens: dict):
        self.model, self.cfg, self.tens, self.saved = model, cfg, tens, {}

    def __enter__(self):
        lin = {f"{li}.{n}": getattr(p, n) for li, p, n in _parents(self.model)}
        scale = self.cfg["alpha"] / self.cfg["rank"]
        with torch.no_grad():
            for k in {k.rsplit(".", 1)[0] for k in self.tens}:
                w = lin[k].weight
                self.saved[k] = w.detach().clone()
                a, b = self.tens[f"{k}.A"].to(w.device), self.tens[f"{k}.B"].to(w.device)
                w.copy_((w.float() + scale * (b @ a)).to(w.dtype))
        return self

    def __exit__(self, *exc):
        lin = {f"{li}.{n}": getattr(p, n) for li, p, n in _parents(self.model)}
        with torch.no_grad():
            for k, w in self.saved.items():
                lin[k].weight.copy_(w)
        self.saved = {}
        return False


def fingerprint(model) -> str:
    """sha256 over the exact bytes of every base parameter and buffer (name, dtype, shape, raw bytes), in a fixed
    order. Any change to any base weight, including permutations, changes it."""
    import hashlib
    h = hashlib.sha256()
    with torch.no_grad():
        items = list(model.named_parameters(remove_duplicate=True)) + list(model.named_buffers(remove_duplicate=True))
        for name, t in items:
            t = t.detach().contiguous()
            h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.view(-1).view(torch.uint8).cpu().numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()
