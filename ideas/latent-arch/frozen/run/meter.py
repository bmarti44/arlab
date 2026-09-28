"""FLOP counter + op-namespace audit wrapped around every scored forward in the EVALUATE worker.

FlopCounterMode counts matmul/attention FLOPs of the ops actually dispatched on the actual inputs (so loops, per-token
recursion depth, halting and extra positions are counted where they happen); OpAudit records every op outside the
aten/prims namespaces (custom Triton/CUDA kernels registered as ops, torch.library ops): any such op makes the run
invalid. Known blind spots (IDEA.md "As built"): kernels launched without the dispatcher, and tampering with these
objects by surface code running in the same worker process.
"""
from __future__ import annotations

import torch
import torch.utils.flop_counter as fcm
from torch.utils._python_dispatch import TorchDispatchMode

ALLOWED_NS = ("aten", "prims")


class OpAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.bad: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if getattr(func, "namespace", None) not in ALLOWED_NS:
            self.bad.add(str(func))
        return func(*args, **(kwargs or {}))


def _sdpa_cpu_flops(q, k, v, *args, out_shape=None, **kwargs) -> int:
    return fcm.sdpa_flop_count(q, k, v)


# FlopCounterMode counts SDPA on CUDA (flash / efficient / cudnn) but not the CPU kernel; count it the same way.
CUSTOM_FLOPS = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}


@torch.no_grad()
def metered(model, x: torch.Tensor) -> tuple[torch.Tensor, int, list]:
    """(logits, counted FLOPs, ops outside aten/prims) of one forward under the scorer's autocast (bf16 on CUDA)."""
    audit = OpAudit()
    counter = fcm.FlopCounterMode(display=False, custom_mapping=CUSTOM_FLOPS)
    with counter, audit, torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = model(x)
    return out, int(counter.get_total_flops()), sorted(audit.bad)
