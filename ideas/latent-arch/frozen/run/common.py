"""Frozen constants and checkpoint helpers shared by the RUN supervisor, its trainer child, the EVALUATE worker,
the evaluator and the pack tests.

The checkpoint is DATA ONLY: a flat {name: tensor} dict of every parameter and buffer (persistent or not, any dtype)
of the surface's model, written by frozen code and read with torch.load(weights_only=True).
"""
from __future__ import annotations

import hashlib

import torch

VOCAB = 8192
SEQ_LEN = 1024
TEXT_ROWS, PROG_ROWS = 58, 6          # rows per 64 x 1024 training batch (programs ~9 % of tokens)
GRACE_S = 30                          # the supervisor kills the trainer at budget + GRACE_S (budget.limit = 330 + 30)
MAX_CKPT_TENSORS = 100_000


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_tensors(model: torch.nn.Module) -> dict:
    """Every parameter and buffer (including non-persistent ones) as detached CPU tensors."""
    out = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        out[name] = t.detach().to("cpu", copy=True).contiguous()
    return out


def load_checkpoint(path: str) -> dict:
    """Trusted loader: weights_only unpickling; must be a flat {str: Tensor} dict. Raises ValueError otherwise."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"checkpoint is not a weights-only tensor file: {e!r}"[:500])
    if not isinstance(ck, dict) or len(ck) > MAX_CKPT_TENSORS:
        raise ValueError("checkpoint must be a flat dict {name: tensor}")
    for k, v in ck.items():
        if not isinstance(k, str) or type(v) is not torch.Tensor or v.is_sparse or v.is_quantized:
            raise ValueError(f"checkpoint entry {k!r} is not a dense tensor")
    return ck


def tensors_hash(ck: dict) -> str:
    """sha256 over names, dtypes, shapes and bytes of a {name: tensor} dict."""
    h = hashlib.sha256()
    for name in sorted(ck):
        t = ck[name].detach().contiguous().cpu()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def count_elements(ck: dict) -> int:
    """All tensor state: every floating-point element counts 1; every BYTE of a non-floating tensor (integer, bool)
    counts 1, so weights bit-packed into integer tensors are counted at least once each."""
    return sum(t.numel() if t.is_floating_point() or t.is_complex() else t.numel() * t.element_size()
               for t in ck.values())


def count_bytes(ck: dict) -> int:
    """Storage capacity of all tensor state in bytes (packing values into wider dtypes gains nothing here)."""
    return sum(t.numel() * t.element_size() for t in ck.values())
