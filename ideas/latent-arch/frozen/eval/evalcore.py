"""Frozen scoring and measurement code for latent-arch EVALUATE (importable by the pack tests).

Everything is computed here from the raw logits of the loaded model; nothing the surface reports is trusted.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.utils.flop_counter as fcm
from torch.utils._python_dispatch import TorchDispatchMode

from common import VOCAB

ALLOWED_NS = ("aten", "prims")


def logits(model, x: torch.Tensor) -> torch.Tensor:
    """model(x) under the same autocast the LM scorer uses (bf16 on CUDA, none on CPU); shape-checked, fp32."""
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        lg = model(x)
    if not isinstance(lg, torch.Tensor) or lg.dim() != 3 or tuple(lg.shape) != (*x.shape, VOCAB):
        raise ValueError(f"model(x) must return logits (B, T, {VOCAB}) for x {tuple(x.shape)}; "
                         f"got {getattr(lg, 'shape', type(lg))}")
    return lg.float()


@torch.no_grad()
def predict(model, tokens: np.ndarray, device: str, batch: int = 128) -> tuple[np.ndarray, float]:
    """Full-vocabulary argmax at the LAST position of each prompt (BOS + program + `q?`), one forward pass, no
    generated tokens. All prompts have the same length. Returns (predicted ids, seconds for the timed passes)."""
    x_all = torch.as_tensor(np.asarray(tokens, dtype=np.int64), device=device)
    logits(model, x_all[:min(batch, len(x_all))])  # warm-up, not timed
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = []
    for i in range(0, len(x_all), batch):
        last = logits(model, x_all[i:i + batch])[:, -1]
        if not torch.isfinite(last).all():
            raise ValueError("non-finite logits at the answer position")
        out.append(last.argmax(-1))
    if device == "cuda":
        torch.cuda.synchronize()
    return torch.cat(out).cpu().numpy(), time.time() - t0


class OpAudit(TorchDispatchMode):
    """Records every dispatched op outside the aten/prims namespaces (custom kernels, torch.library ops, ...)."""

    def __init__(self):
        super().__init__()
        self.bad: set[str] = set()
        self.n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n_ops += 1
        if getattr(func, "namespace", None) not in ALLOWED_NS:
            self.bad.add(str(func))
        return func(*args, **(kwargs or {}))


def _sdpa_cpu_flops(q, k, v, *args, out_shape=None, **kwargs) -> int:
    return fcm.sdpa_flop_count(q, k, v)


# FlopCounterMode counts SDPA on CUDA (flash / efficient / cudnn) but not the CPU kernel; count it the same way.
CUSTOM_FLOPS = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}


@torch.no_grad()
def probe(model, inputs: list[np.ndarray], device: str) -> dict:
    """Counted FLOPs per scored token over the probe inputs (each a (B, T) int array), plus the op audit."""
    audit = OpAudit()
    counter = fcm.FlopCounterMode(display=False, custom_mapping=CUSTOM_FLOPS)
    tokens = 0
    with counter, audit:
        for arr in inputs:
            x = torch.as_tensor(np.asarray(arr, dtype=np.int64), device=device)
            logits(model, x)
            tokens += x.numel()
    flops = counter.get_total_flops()
    return {"flops": int(flops), "tokens": tokens, "flops_per_token": flops / max(tokens, 1),
            "bad_ops": sorted(audit.bad), "n_ops": audit.n_ops}


def accuracy_report(pred: np.ndarray, d: dict, idx: np.ndarray) -> dict:
    """Scores for the selected items idx of programs.npz d. pred aligned with idx."""
    correct = (pred == d["answer"][idx]).astype(np.float64)
    group, k = d["group"][idx], d["k"][idx]
    pos = {int(j): n for n, j in enumerate(idx)}
    m = {}
    for name, g in (("acc_id", 0), ("acc_depth", 1), ("acc_ext", 2)):
        sel = group == g
        m[name] = float(correct[sel].mean()) if sel.any() else None
    for kk in sorted(set(k[group < 3].tolist())):
        m[f"acc_k{kk}"] = float(correct[(k == kk) & (group < 3)].mean())
    pairs = [(n, pos[int(d["cf_of"][j])]) for n, j in enumerate(idx) if group[n] == 3 and int(d["cf_of"][j]) in pos]
    m["cf_both"] = float(np.mean([correct[a] * correct[b] for a, b in pairs])) if pairs else None
    main = group < 2
    val = d["value"][idx][main]
    for h in ("h_last_const", "h_own_const", "h_root"):
        m["floor_" + h[2:]] = float((d[h][idx][main] == val).mean())
    m["floor_modal"] = float(np.bincount(val, minlength=100).max() / max(len(val), 1))
    m["accuracy"] = 0.5 * m["acc_id"] + 0.5 * m["acc_depth"]
    ids = {0: "id", 1: "depth"}
    items = {f"{ids[int(g)]}{int(j):05d}": float(c) for j, g, c in zip(idx, group, correct) if g < 2}
    return {"metrics": m, "items": items}
