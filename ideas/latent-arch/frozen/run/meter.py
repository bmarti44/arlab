"""FLOP counter + op audit wrapped around every scored forward in the EVALUATE worker.

Every op the dispatcher sees in a scored forward is classified:
  * FORMULA ops (matmul / fused matmul / conv / attention / FFT): FlopCounterMode's formulas, plus the fused and
    CPU variants it lacks (_addmm_activation, addbmm, addmv, mv, addr, dot, _int_mm, CPU flash SDPA, FFTs);
  * views (func.is_view) and ops with no tensor inputs (factories: zeros, arange, rand, ...) are free;
  * ALLOWED ops (tagged pointwise or reduction, plus an explicit list of normalization / softmax / data-movement /
    scan / sort ops) are charged a lower bound of max(elements read, elements written);
  * anything else is REJECTED: listed as "unpriced:<op>" and the run is invalid (so a fused matmul or kernel the
    counter does not know cannot be bought at elementwise price). Ops outside the aten/prims namespaces (custom
    Triton/CUDA kernels, torch.library ops, higher-order ops) are rejected too.
The dispatch-mode stack is thread-local: the worker forbids threads/processes created by surface code (worker.py).
Known blind spots (IDEA.md "As built"): kernels launched without the dispatcher, and tampering with these objects by
surface code running in the same worker process.
"""
from __future__ import annotations

import math

import torch
import torch.utils.flop_counter as fcm
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ALLOWED_NS = ("aten", "prims")
aten = torch.ops.aten


# FlopCounterMode passes custom formulas the SHAPES of tensor arguments (torch.Size), not the tensors.
def _sdpa_cpu_flops(q, k, v, *args, out_shape=None, **kwargs) -> int:
    return fcm.sdpa_flop_count(q, k, v)


def _addmm_act_flops(self, mat1, mat2, *args, out_shape=None, **kwargs) -> int:
    return fcm.mm_flop(mat1, mat2) + 2 * mat1[0] * mat2[1]           # + bias add + activation


def _addbmm_flops(self, b1, b2, *args, out_shape=None, **kwargs) -> int:
    return fcm.bmm_flop(b1, b2) + b1[1] * b2[2] * (b1[0] + 1)          # + the sum over the batch + self


def _mv_flops(mat, vec, *args, out_shape=None, **kwargs) -> int:
    return 2 * mat[0] * mat[1]


def _addmv_flops(self, mat, vec, *args, out_shape=None, **kwargs) -> int:
    return 2 * mat[0] * mat[1] + mat[0]


def _addr_flops(self, v1, v2, *args, out_shape=None, **kwargs) -> int:
    return 3 * v1[0] * v2[0]


def _dot_flops(a, b, *args, out_shape=None, **kwargs) -> int:
    return 2 * math.prod(a)


def _int_mm_flops(a, b, *args, out_shape=None, **kwargs) -> int:
    return fcm.mm_flop(a, b)


def _fft_flops(x, *args, out_shape=None, **kwargs) -> int:
    n = max(math.prod(x), 2)
    return int(5 * n * math.log2(n))


CUSTOM_FLOPS = {}
for _name, _fn in (("_scaled_dot_product_flash_attention_for_cpu", _sdpa_cpu_flops), ("_addmm_activation", _addmm_act_flops),
                   ("addbmm", _addbmm_flops), ("mv", _mv_flops), ("addmv", _addmv_flops), ("addr", _addr_flops),
                   ("dot", _dot_flops), ("vdot", _dot_flops), ("_int_mm", _int_mm_flops),
                   ("_fft_r2c", _fft_flops), ("_fft_c2r", _fft_flops), ("_fft_c2c", _fft_flops)):
    if hasattr(aten, _name):                        # the container's torch may lack an op the host has, or vice versa
        CUSTOM_FLOPS[getattr(aten, _name)] = _fn
FORMULA = set(fcm.flop_registry) | set(CUSTOM_FLOPS)

# Priced at max(elements read, written): everything tagged pointwise / reduction, plus these (in-place `name_`
# variants included). Adding an op here is a pack change, never a surface change.
ALLOWED = set("""
_softmax _log_softmax softmax log_softmax _safe_softmax native_layer_norm _fused_rms_norm rms_norm native_group_norm
native_batch_norm _native_batch_norm_legit _native_batch_norm_legit_no_training native_dropout
_to_copy copy clone contiguous _unsafe_view cat stack index index_put _index_put_impl index_select gather scatter
scatter_add scatter_reduce index_add index_copy index_fill masked_fill masked_scatter masked_select embedding
tril triu constant_pad_nd reflection_pad1d replication_pad1d fill zero repeat repeat_interleave roll flip tile
cumsum cumprod logcumsumexp _cummax_helper _cummin_helper cummax cummin sort topk kthvalue argsort searchsorted
bucketize nonzero _local_scalar_dense where one_hot _unique2 unique_dim new_zeros new_ones new_empty new_full
empty_like zeros_like ones_like full_like rand_like randn_like bernoulli uniform normal diag_embed diagonal_scatter
slice_scatter select_scatter as_strided_scatter _unsafe_index linalg_vector_norm norm frobenius_norm _pdist_forward
cdist _euclidean_dist _cdist_forward
sum mean amax amin max min argmax argmin aminmax prod var std var_mean std_mean logsumexp any all nansum
count_nonzero median nanmedian mode
""".split())
_TAGS = tuple(t for t in (getattr(torch.Tag, "pointwise", None), getattr(torch.Tag, "reduction", None)) if t is not None)


def _numel(xs) -> int:
    return sum(x.numel() for x in tree_flatten(xs)[0] if isinstance(x, torch.Tensor))


def _allowed(func) -> bool:
    if any(t in func.tags for t in _TAGS):
        return True
    name = func._overloadpacket.__name__
    return name in ALLOWED or name.rstrip("_") in ALLOWED


class OpAudit(TorchDispatchMode):
    """Namespace audit, rejection of unpriced compute ops, and lower-bound FLOPs for allowed elementwise ops."""

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
                if _allowed(func):
                    self.other_flops += max(n_in, _numel(out))
                else:
                    self.bad.add(f"unpriced:{func}")
        return out


@torch.no_grad()
def metered(model, x: torch.Tensor) -> tuple[torch.Tensor, int, list]:
    """(logits, counted FLOPs, rejected ops) of one forward under the scorer's autocast (bf16)."""
    audit = OpAudit()
    counter = fcm.FlopCounterMode(display=False, custom_mapping=CUSTOM_FLOPS)
    with counter, audit, torch.autocast(x.device.type, dtype=torch.bfloat16):
        out = model(x)
    return out, int(counter.get_total_flops()) + audit.other_flops, sorted(audit.bad)
