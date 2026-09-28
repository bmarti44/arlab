"""Frozen constants and integrity helpers shared by the RUN harness, EVALUATE and the pack tests."""
from __future__ import annotations

import hashlib

import torch

VOCAB = 8192
SEQ_LEN = 1024
TEXT_ROWS, PROG_ROWS = 58, 6          # rows per 64 x 1024 training batch (programs ~9 % of tokens)
MAX_OVERRUN_S = 30                    # the last train_step may end at most this far past the budget


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(model: torch.nn.Module) -> str:
    """sha256 over every parameter and buffer (names, dtypes, shapes, bytes) of a loaded model."""
    h = hashlib.sha256()
    for name, t in sorted(list(model.named_parameters()) + list(model.named_buffers()), key=lambda kv: kv[0]):
        t = t.detach().contiguous().cpu()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def count_params(model: torch.nn.Module) -> int:
    """Parameters + floating-point buffers (shared tensors counted once)."""
    seen, n = set(), 0
    for t in list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()]:
        if id(t) not in seen:
            seen.add(id(t))
            n += t.numel()
    return n
