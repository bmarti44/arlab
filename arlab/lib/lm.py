"""Shared LM helpers for training packs: the harness-owned token-budget loop and the frozen LM scorer.

The scorer never calls surface loss code: it feeds tokens[:-1], takes the logits the model returns and computes
log_softmax / cross-entropy itself (PLAN §6.1, anti-cheat (f) and (g)).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch


def token_budget_loop(surface, train_tokens: str, budget: int, batch: int, seq_len: int, seed: int, out: str, config: dict) -> dict:
    """Random fixed-size windows from a uint16 token stream until `budget` tokens are consumed.

    The surface supplies build(config) -> state, train_step(state, (x, y), step, total_steps) -> loss, save(state, path).
    """
    stream = np.memmap(train_tokens, dtype=np.uint16, mode="r")
    per_step = batch * seq_len
    total_steps = budget // per_step
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    state = surface.build({**config, "seq_len": seq_len, "batch": batch, "total_steps": total_steps, "seed": seed})
    pinned = torch.empty((batch, seq_len + 1), dtype=torch.long).pin_memory()
    t0, step, nan_at = time.time(), 0, None
    for step in range(total_steps):
        starts = rng.integers(0, len(stream) - seq_len - 1, batch)
        pinned.copy_(torch.from_numpy(np.stack([stream[s:s + seq_len + 1] for s in starts]).astype(np.int64)))
        xy = pinned.to("cuda", non_blocking=True)
        loss = surface.train_step(state, (xy[:, :-1], xy[:, 1:]), step, total_steps)
        if step % 50 == 0 or step == total_steps - 1:
            lv = float(loss)
            print(f"step {step}/{total_steps} loss {lv:.4f} {time.time() - t0:.1f}s", flush=True)
            if not math.isfinite(lv):
                nan_at = step
                print(f"NaN loss at step {step}; stopping early", flush=True)
                break
    torch.cuda.synchronize()
    train_time = time.time() - t0
    used = (step + 1) * per_step
    surface.save(state, f"{out}/model.pt")
    json.dump({"tokens": used}, open(f"{out}/budget.json", "w"))
    print(f"trained {used} tokens in {train_time:.1f}s ({used / max(train_time, 1e-9):.0f} tok/s)", flush=True)
    return {"tokens": used, "train_time": train_time, "nan_at": nan_at}


@torch.no_grad()
def lm_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = model(x)
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != x.shape or logits.dim() != 3:
        raise ValueError(f"model(x) must return logits (B, T, V); got {getattr(logits, 'shape', type(logits))}")
    return torch.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def score_bpb(model, rows: np.ndarray, token_bytes: np.ndarray, batch: int = 32, device: str = "cuda") -> dict:
    """val_bpb over fixed rows of (T+1) tokens: frozen cross-entropy from the model's logits."""
    tb = torch.as_tensor(token_bytes.astype(np.int64), device=device)
    nats, nbytes = 0.0, 0
    for i in range(0, len(rows), batch):
        r = torch.as_tensor(rows[i:i + batch].astype(np.int64), device=device)
        x, y = r[:, :-1], r[:, 1:]
        lp = lm_logprobs(model, x)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        b = tb[y]
        if not torch.isfinite(nll[b > 0]).all():
            return {"valid": False, "message": "non-finite log-probs"}
        nats += float((nll * (b > 0)).sum())
        nbytes += int(b.sum())
    return {"valid": True, "val_bpb": nats / (math.log(2) * nbytes), "tokens": int(rows.shape[0] * (rows.shape[1] - 1))}


@torch.no_grad()
def causality_check(model, rows: np.ndarray, vocab: int, n: int = 4, tol: float = 1e-3, seed: int = 0, device: str = "cuda") -> dict:
    """Perturbing tokens after position t must not change log-probs at positions <= t (anti-cheat (f))."""
    g = np.random.default_rng(seed)
    x = torch.as_tensor(rows[:n, :-1].astype(np.int64), device=device)
    T = x.shape[1]
    worst = 0.0
    for t in (T // 4, T // 2, (3 * T) // 4):
        x2 = x.clone()
        x2[:, t + 1:] = torch.as_tensor(g.integers(0, vocab, (n, T - t - 1)), device=device)
        a, b = lm_logprobs(model, x)[:, :t + 1], lm_logprobs(model, x2)[:, :t + 1]
        worst = max(worst, float((a - b).abs().max()))
    return {"causal": worst <= tol, "max_diff": worst}
