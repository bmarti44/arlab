"""FLOP counter + op-namespace audit wrapped around every scored forward in the EVALUATE worker.

FlopCounterMode counts matmul/attention FLOPs by formula on the actual inputs (so loops, per-token recursion depth,
halting and extra positions are counted where they happen). Every other op that reads tensors and is not a pure
view (elementwise arithmetic, reductions, gathers, copies, scans, ...) is charged a lower bound of
max(elements read, elements written), so rewriting a matmul as broadcast-multiply + sum costs the same.
Ops without tensor inputs (factories: zeros, arange, ...) and view/alias ops are free. OpAudit records every op
outside the aten/prims namespaces (custom Triton/CUDA kernels registered as ops, torch.library ops): any such op
makes the run invalid. Known blind spots (IDEA.md "As built"): kernels launched without the dispatcher, and
tampering with these objects by surface code running in the same worker process.
"""
from __future__ import annotations

import torch
import torch.utils.flop_counter as fcm
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ALLOWED_NS = ("aten", "prims")


def _sdpa_cpu_flops(q, k, v, *args, out_shape=None, **kwargs) -> int:
    return fcm.sdpa_flop_count(q, k, v)


# FlopCounterMode counts SDPA on CUDA (flash / efficient / cudnn) but not the CPU kernel; count it the same way.
CUSTOM_FLOPS = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}
FORMULA = set(fcm.flop_registry) | set(CUSTOM_FLOPS)


def _numel(xs) -> int:
    return sum(x.numel() for x in tree_flatten(xs)[0] if isinstance(x, torch.Tensor))


class OpAudit(TorchDispatchMode):
    """Namespace audit + elementwise lower-bound FLOPs for every op without a FLOP formula."""

    def __init__(self):
        super().__init__()
        self.bad: set[str] = set()
        self.other_flops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "namespace", None) not in ALLOWED_NS:
            self.bad.add(str(func))
        out = func(*args, **kwargs)
        if func._overloadpacket not in FORMULA and not getattr(func, "is_view", False):
            n_in = _numel((args, kwargs))
            if n_in:
                self.other_flops += max(n_in, _numel(out))
        return out


@torch.no_grad()
def metered(model, x: torch.Tensor) -> tuple[torch.Tensor, int, list]:
    """(logits, counted FLOPs, ops outside aten/prims) of one forward under the scorer's autocast (bf16)."""
    audit = OpAudit()
    counter = fcm.FlopCounterMode(display=False, custom_mapping=CUSTOM_FLOPS)
    with counter, audit, torch.autocast(x.device.type, dtype=torch.bfloat16):
        out = model(x)
    return out, int(counter.get_total_flops()) + audit.other_flops, sorted(audit.bad)
