"""Frozen constants and helpers shared by PREPARE, the RUN harness and EVALUATE (all scoring math lives here or in
frozen/eval; the surface only supplies logits)."""
from __future__ import annotations

import numpy as np
import torch

# Pinned, cached, offline (no downloads anywhere in this pack).
MODEL_DIR = "/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
OASST_FILE = ("/hf/hub/datasets--OpenAssistant--oasst2/snapshots/179dd21fc55192153d94adb0e0ce8f69e222bf75/"
              "2023-11-05_oasst2_ready.trees.jsonl.gz")

# Qwen3 chat format with thinking disabled (what apply_chat_template(enable_thinking=False) produces): answer only.
PROMPT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TARGET = "{a}<|im_end|>"
IM_END, EOT = 151645, 151643          # <|im_end|>, <|endoftext|>
STOP_IDS = (IM_END, EOT)
PAD_ID = EOT
MAX_NEW = 6                           # answers are 0..99 (Qwen3 splits digits) + <|im_end|>; no room for reasoning


def check_logits(lg, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(lg, torch.Tensor) or lg.dim() != 3 or tuple(lg.shape[:2]) != tuple(x.shape):
        raise ValueError(f"logits(ids) must return (B, T, V) for ids {tuple(x.shape)}; got {getattr(lg, 'shape', type(lg))}")
    return lg


@torch.no_grad()
def token_nll(logits_fn, rows: np.ndarray, batch: int, device: str) -> np.ndarray:
    """Per-token NLL (nats) of rows[:, 1:] given rows[:, :-1], computed here from the raw logits. Shape (n, T)."""
    out = []
    for i in range(0, len(rows), batch):
        r = torch.as_tensor(rows[i:i + batch].astype(np.int64), device=device)
        x, y = r[:, :-1], r[:, 1:]
        lp = torch.log_softmax(check_logits(logits_fn(x), x).float(), dim=-1)
        out.append((-lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def greedy_decode(logits_fn, prompts: list[list[int]], batch: int, device: str, max_new: int = MAX_NEW) -> tuple[list[list[int]], int]:
    """Greedy answers with full forward passes (no KV cache), right padding, per-row lengths.

    Returns (generated ids per prompt, number of forward calls whose last-position logits were non-finite)."""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[list[int]] = [[] for _ in prompts]
    stop = torch.tensor(STOP_IDS, device=device)
    bad = 0
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        n = len(idx)
        lens = torch.tensor([len(prompts[i]) for i in idx], device=device)
        ids = torch.full((n, int(lens.max()) + max_new), PAD_ID, dtype=torch.long, device=device)
        for r, i in enumerate(idx):
            ids[r, :len(prompts[i])] = torch.tensor(prompts[i], dtype=torch.long)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        rows = torch.arange(n, device=device)
        for _ in range(max_new):
            x = ids[:, :int(lens.max())]
            last = check_logits(logits_fn(x), x)[rows, lens - 1].float()
            bad += int(not torch.isfinite(last).all())
            nxt = last.argmax(-1)
            active = ~done
            for r in active.nonzero().flatten().tolist():
                out[idx[r]].append(int(nxt[r]))
            ids[rows[active], lens[active]] = nxt[active]
            lens = lens + active.long()
            done = done | (active & torch.isin(nxt, stop))
            if bool(done.all()):
                break
    return out, bad
