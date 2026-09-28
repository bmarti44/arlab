"""Frozen per-item test-time-training loop for ttt-context (imported by harness.py and by the CPU tests).

Two copies of the model: ANSWER (never handed to surface code) and WORK (the surface's working copy). Per item:
  1. reseed every RNG from the item seed; prefill the document on ANSWER (pristine weights; not timed as TTT) and
     give the surface its own clone of that KV cache;
  2. TIMED (device-synced before and after): import a fresh copy of `ttt`, call adapt(WORK, ctx), validate its
     return value {"weights": {param name: tensor}, "doc_in_context": bool} and clone the weights;
  3. restore the process state surface code could have touched: global and per-module hooks, the class dictionaries
     of every model module class, the dictionaries of the modules the answer path uses (this file, torch.nn.functional,
     modeling_qwen3, attention/mask registries), sys.modules entries the surface added; refuse a still-running thread;
  4. check ANSWER still equals the pristine weights, then answer on it with only the returned weights copied in,
     from the frozen prompt (the stored question tokens, after the harness's own document cache if doc_in_context);
     no surface code runs while answering;
  5. restore ANSWER's weights and WORK entirely (weights, buffers, hooks, instance attributes, structure check) and
     verify both equal the pristine copy exactly, plus a probe-logit check on ANSWER.
The evaluator scores an item 0 if its TTT time exceeded the limit, and invalidates the run if any restore failed or
the SHA-256 of either model's weights at the end differs from the pristine weights at load.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional
import torch.nn.modules.module as _torch_module
import transformers.cache_utils
import transformers.integrations.sdpa_attention
import transformers.masking_utils
import transformers.modeling_utils
import transformers.models.qwen3.modeling_qwen3
from transformers.cache_utils import DynamicCache, DynamicLayer

from common import MAX_NEW, STOP_IDS

HEAD_IDS = (151644, 872, 198)   # "<|im_start|>user\n": the first tokens of every document prompt
PROBE = tuple(range(1000, 1064))
PROBE_TOL = 0.25                # bf16 logits of size ~20 have an ulp of 0.125: allow kernel-level nondeterminism
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
               "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called")


def fresh_import(name: str):
    """Import the surface module anew (module-level state never survives between items)."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)          # also seeds every CUDA device


class TTTContext:
    """Everything the surface gets for ONE item: the document and question tokens (copies), its own clone of the
    base model's KV cache of the document, a per-item random generator and the deadline. No answers, no other items."""

    def __init__(self, doc_ids: torch.Tensor, question_ids: torch.Tensor, cache, seed: int, deadline: float):
        self.doc_ids = doc_ids                  # (1, L) long: chat head + document (what the model reads)
        self.question_ids = question_ids        # (1, Q) long: question + assistant header + "Answer:"
        self.doc_len = int(doc_ids.shape[1])
        self.device = doc_ids.device
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)   # CPU generator: use it for every random choice
        self._cache = cache
        self._deadline = deadline

    def time_left(self) -> float:
        """Seconds left of this item's TTT budget (can be negative)."""
        return self._deadline - time.perf_counter()

    def prefix_cache(self, n: int | None = None) -> DynamicCache:
        """A new DynamicCache with the base model's keys/values of document positions [0, n) (default: all)."""
        if self._cache is None:
            raise RuntimeError("no document cache")
        n = self.doc_len if n is None else int(n)
        if not 0 < n <= self.doc_len:
            raise ValueError(f"prefix length {n} outside (0, {self.doc_len}]")
        c = DynamicCache()
        for k, v in self._cache:
            new = DynamicLayer()
            new.dtype, new.device = k.dtype, k.device
            new.keys, new.values = k[:, :, :n], v[:, :, :n]
            c.layers.append(new)
        return c


@torch.no_grad()
def prefill(model, ids: torch.Tensor):
    """Base-model KV cache of a prompt (no logits)."""
    return model.model(input_ids=ids, use_cache=True).past_key_values


def cache_view(cache, n: int) -> DynamicCache:
    c = DynamicCache()
    for layer in cache.layers:
        new = DynamicLayer()
        new.dtype, new.device = layer.keys.dtype, layer.keys.device
        new.keys, new.values = layer.keys[:, :, :n], layer.values[:, :, :n]
        c.layers.append(new)
    return c


@torch.no_grad()
def greedy(model, prompt_ids: torch.Tensor, cache, max_new: int = MAX_NEW) -> tuple[list[int], int]:
    """Greedy answer tokens after prompt_ids (appended to cache, or from scratch if cache is None)."""
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    toks, bad = [], 0
    for step in range(max_new):
        lg = out.logits[0, -1].float()
        bad += int(not torch.isfinite(lg).all())
        nxt = int(lg.argmax())
        toks.append(nxt)
        if nxt in STOP_IDS or step == max_new - 1:
            break
        out = model(input_ids=torch.tensor([[nxt]], device=prompt_ids.device), past_key_values=out.past_key_values,
                    use_cache=True, logits_to_keep=1)
    return toks, bad


def _spec(model) -> list:
    return ([(n, tuple(p.shape), p.dtype, p.device) for n, p in model.named_parameters()]
            + [(n, tuple(b.shape), b.dtype) for n, b in model.named_buffers()]
            + [(n, type(m).__name__) for n, m in model.named_modules()])


def sha_params(named) -> str:
    """SHA-256 over (name, raw bytes) of every tensor, in name order."""
    h = hashlib.sha256()
    for n, t in sorted(named, key=lambda x: x[0]):
        h.update(n.encode() + b"\0")
        h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _restore_dict(d: dict, snap: dict) -> None:
    for k in [k for k in d if k not in snap]:
        del d[k]
    for k, v in snap.items():
        if d.get(k, d) is not v:
            d[k] = v


class ProcessState:
    """Snapshot of the in-process state the answer path depends on, restored after every adapt()."""

    def __init__(self, model):
        self.globals = {k: v for k, v in vars(_torch_module).items() if k.startswith("_global")}
        self.global_dicts = {k: dict(v) for k, v in self.globals.items() if isinstance(v, dict)}
        self.classes = {c: dict(vars(c)) for m in model.modules() for c in type(m).__mro__ if c is not object}
        mods = [sys.modules[__name__], torch.nn.functional, transformers.models.qwen3.modeling_qwen3,
                transformers.integrations.sdpa_attention, transformers.masking_utils, transformers.cache_utils,
                transformers.modeling_utils]
        self.modules = {m: dict(vars(m)) for m in mods}
        self.registries = [(r, dict(type(r)._global_mapping), dict(r._local_mapping)) for r in
                           (transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS,
                            transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS)]
        self.sys_modules = set(sys.modules)
        self.threads = set(threading.enumerate())

    def restore(self) -> None:
        extra = [t for t in threading.enumerate() if t not in self.threads and t.is_alive()]
        if extra:
            raise RuntimeError(f"the surface left threads running: {[t.name for t in extra]}")
        for k, v in self.globals.items():
            if k in self.global_dicts:
                v.clear()
                v.update(self.global_dicts[k])
            setattr(_torch_module, k, v)
        for c, snap in self.classes.items():
            cur = vars(c)
            for k in [k for k in cur if k not in snap]:
                delattr(c, k)
            for k, v in snap.items():
                if cur.get(k, cur) is not v:
                    setattr(c, k, v)
        for m, snap in self.modules.items():
            _restore_dict(vars(m), snap)
        for r, g, loc in self.registries:
            _restore_dict(type(r)._global_mapping, g)
            _restore_dict(r._local_mapping, loc)
        for k in [k for k in sys.modules if k not in self.sys_modules and k not in sys.builtin_module_names]:
            f = getattr(sys.modules[k], "__file__", None) or ""
            if not any(s in f for s in ("site-packages", "dist-packages", "/lib/python")):
                del sys.modules[k]      # the surface's own helper modules and file-less stash modules


class ModelGuard:
    """Restores a model to the pristine weights, buffers, hooks and instance attributes."""

    def __init__(self, model, pristine: dict, buffers: dict):
        self.spec = _spec(model)
        self.params, self.buffers = pristine, buffers
        self.attrs = {m: dict(vars(m)) for m in model.modules()}
        self.config = copy.deepcopy(vars(model.config))

    def restore(self, model) -> bool:
        for m in model.modules():
            for d in _HOOK_DICTS:
                getattr(m, d, {}).clear()
            if "forward" in vars(m) and "forward" not in self.attrs.get(m, {}):
                raise RuntimeError(f"the surface monkeypatched forward() of {type(m).__name__}; change weights only")
        if _spec(model) != self.spec:
            raise RuntimeError("the surface changed the model's modules/parameters/buffers; change weights only")
        for m, snap in self.attrs.items():
            _restore_dict(vars(m), snap)
        _restore_dict(vars(model.config), copy.deepcopy(self.config))
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.requires_grad_(False)
                p.grad = None
                p.copy_(self.params[n])
            for n, b in model.named_buffers():
                b.copy_(self.buffers[n])
        model.eval()
        return all(torch.equal(p, self.params[n]) for n, p in model.named_parameters()) and \
            all(torch.equal(b, self.buffers[n]) for n, b in model.named_buffers())


def check_update(ret, params: dict) -> tuple[dict, bool]:
    """Validate adapt()'s return value and clone the weights (no surface object survives this call)."""
    if type(ret) is not dict or set(ret) != {"weights", "doc_in_context"}:
        raise TypeError('adapt() must return {"weights": {name: tensor}, "doc_in_context": bool}')
    doc_in_context, weights = ret["doc_in_context"], ret["weights"]
    if type(doc_in_context) is not bool:
        raise TypeError(f"doc_in_context must be a bool, got {type(doc_in_context).__name__}")
    if type(weights) is not dict:
        raise TypeError("weights must be a dict {parameter name: tensor}")
    out = {}
    for n, t in weights.items():
        if type(n) is not str or n not in params:
            raise KeyError(f"unknown parameter {n!r}")
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(params[n].shape) or t.dtype != params[n].dtype:
            raise TypeError(f"{n}: expected a {params[n].dtype} tensor of shape {tuple(params[n].shape)}")
        t = t.detach().to(params[n].device).clone()
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{n}: non-finite values")
        out[n] = t
    return out, doc_in_context


def run_items(model, load_surface, docs: list[np.ndarray], items: list[dict], *, ttt_seconds: float, seed: int,
              device, log=print) -> tuple[dict, dict, dict]:
    """The frozen item loop. `model` becomes the ANSWER model; a deep copy is the surface's WORK model.
    Returns (preds {id: token ids}, per-item records {id: {...}}, run stats)."""
    t_run = time.perf_counter()
    _prefill, _greedy, _check, _sync_, _seed, _cview = prefill, greedy, check_update, _sync, seed_all, cache_view
    answer = model
    work = copy.deepcopy(model)
    params = {n: p.detach().clone() for n, p in answer.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in answer.named_buffers()}
    sha_start = sha_params(list(params.items()) + list(buffers.items()))
    g_answer, g_work = ModelGuard(answer, params, buffers), ModelGuard(work, params, buffers)
    probe_ids = torch.tensor([PROBE], device=device)
    with torch.no_grad():
        probe_ref = answer(input_ids=probe_ids, use_cache=False).logits[0, -8:].float()
    state = ProcessState(answer)
    head = torch.tensor([HEAD_IDS], device=device)
    preds, per = {}, {}
    for idx, (doc, it) in enumerate(zip(docs, items)):
        item_seed = seed * 1_000_003 + idx
        _seed(item_seed)
        doc_np, q_np = np.asarray(doc, dtype=np.int64).copy(), np.asarray(it["q"], dtype=np.int64).copy()
        t = time.perf_counter()
        cache = _prefill(answer, torch.as_tensor(doc_np, device=device).unsqueeze(0))
        own = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
        _sync_(device)
        prefill_s = time.perf_counter() - t
        # ---------------- timed: everything surface-controlled
        t0 = time.perf_counter()
        ctx = TTTContext(torch.as_tensor(doc_np, device=device).unsqueeze(0),
                         torch.as_tensor(q_np, device=device).unsqueeze(0), own, item_seed, t0 + ttt_seconds)
        surface = load_surface()
        with torch.enable_grad():
            ret = surface.adapt(work, ctx)
        weights, use_doc = _check(ret, params)
        _sync_(device)
        ttt_s = time.perf_counter() - t0
        # ---------------- trusted from here on
        del ctx, ret, surface, own
        state.restore()
        t = time.perf_counter()
        with torch.no_grad():
            clean = all(torch.equal(p, params[n]) for n, p in answer.named_parameters()) and \
                all(torch.equal(b, buffers[n]) for n, b in answer.named_buffers())    # untouched by the surface?
            for n, p in answer.named_parameters():
                if n in weights:
                    p.copy_(weights[n])
        q_ids = torch.as_tensor(q_np, device=device).unsqueeze(0)
        if use_doc:
            toks, bad = _greedy(answer, q_ids, _cview(cache, len(doc_np)))
        else:
            toks, bad = _greedy(answer, torch.cat([head, q_ids], dim=1), None)
        _sync_(device)
        answer_s = time.perf_counter() - t
        del cache
        ok = g_answer.restore(answer) and g_work.restore(work)
        with torch.no_grad():
            diff = float((answer(input_ids=probe_ids, use_cache=False).logits[0, -8:].float() - probe_ref).abs().max())
        preds[it["id"]] = toks
        per[it["id"]] = {"ttt_s": ttt_s, "prefill_s": prefill_s, "answer_s": answer_s, "changed_tensors": len(weights),
                         "doc_in_context": use_doc, "nonfinite": bad, "probe_diff": diff,
                         "reset_ok": bool(clean and ok and diff <= PROBE_TOL)}
        if idx % 20 == 0 or idx == len(items) - 1:
            log(f"item {idx + 1}/{len(items)} ttt {ttt_s:.2f}s prefill {prefill_s:.2f}s answer {answer_s:.2f}s "
                f"updated {len(weights)} reset_ok {per[it['id']]['reset_ok']} "
                f"elapsed {time.perf_counter() - t_run:.0f}s")
    stats = {"n_items": len(items), "ttt_seconds_limit": ttt_seconds, "seed": seed, "sha_start": sha_start,
             "sha_end": sha_params(list(answer.named_parameters()) + list(answer.named_buffers())),
             "sha_end_work": sha_params(list(work.named_parameters()) + list(work.named_buffers())),
             "probe_tol": PROBE_TOL, "all_reset_ok": all(v["reset_ok"] for v in per.values()),
             "nonfinite": sum(v["nonfinite"] for v in per.values()), "wall_s": time.perf_counter() - t_run}
    return preds, per, stats
