"""Frozen per-item test-time-training loop for ttt-context (imported by harness.py and by the CPU tests).

For every item, in order:
  1. load a FRESH copy of the surface module (no module state survives between items; the import is timed as TTT);
  2. prefill the document with the base weights (KV cache; not timed as TTT, identical for every arm);
  3. call surface.adapt(model, ctx) under a wall-clock deadline of --ttt-seconds (timed, after a device sync);
  4. answer greedily (frozen): the question after the document cache (DOC_IN_CONTEXT, the default), after the
     cache adapt() returned, or alone (DOC_IN_CONTEXT = False);
  5. reset the model: restore every parameter and buffer from the pristine copy, clear all module hooks, check
     the structure is unchanged, check the parameters equal the pristine copy exactly, and check that the logits on
     a fixed probe match the untouched model's (within PROBE_TOL, bf16 noise).
The evaluator scores an item 0 if its TTT time exceeded the limit, and invalidates the run if any reset failed or
the SHA-256 of the model weights at the end differs from the one at load time.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import time

import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from common import MAX_NEW, STOP_IDS

HEAD_IDS = (151644, 872, 198)   # "<|im_start|>user\n": the first tokens of every document prompt
PROBE = tuple(range(1000, 1064))
PROBE_TOL = 0.25
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
               "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called")


def fresh_import(name: str):
    """Import the surface module anew (module-level state never survives between items)."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class TTTContext:
    """Everything the surface gets for ONE item: the document and question tokens, the base-model KV cache of the
    document (read-only; use prefix_cache), a per-item random generator and the deadline. No answers, no other items."""

    def __init__(self, doc_ids: torch.Tensor, question_ids: torch.Tensor, cache, seed: int, deadline: float):
        self.doc_ids = doc_ids                  # (1, L) long: chat head + document (what the model reads)
        self.question_ids = question_ids        # (1, Q) long: question + "<|im_end|>\n<|im_start|>assistant\n<think>..."
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
        """A new DynamicCache holding the base model's keys/values for document positions [0, n) (default: all).
        It shares storage with the harness's cache: appending to it (a forward pass) is fine, in-place writes are not."""
        if self._cache is None:
            raise RuntimeError("the document was not prefilled (PREFILL_DOC = False)")
        n = self.doc_len if n is None else int(n)
        if not 0 < n <= self.doc_len:
            raise ValueError(f"prefix length {n} outside (0, {self.doc_len}]")
        c = DynamicCache()
        for layer in self._cache.layers:
            new = DynamicLayer()
            new.dtype, new.device = layer.keys.dtype, layer.keys.device
            new.keys, new.values = layer.keys[:, :, :n], layer.values[:, :, :n]
            c.layers.append(new)
        return c


@torch.no_grad()
def prefill(model, ids: torch.Tensor):
    """Base-model KV cache of a prompt (no logits)."""
    return model.model(input_ids=ids, use_cache=True).past_key_values


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


class WeightGuard:
    """Pristine copy of the model, restored and verified after every item."""

    def __init__(self, model, device):
        self.spec = _spec(model)
        self.params = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.probe_ids = torch.tensor([PROBE], device=device)
        self.probe_ref = self.probe(model)
        noise = float((self.probe(model) - self.probe_ref).abs().max())
        # Exact parameter/buffer equality is the guarantee; the probe catches state outside the parameters (global
        # hooks, patched kernels). Floor 0.25: bf16 logits of size ~20 have an ulp of 0.125, so kernel-level
        # nondeterminism alone can move them by one ulp.
        self.tol = max(PROBE_TOL, 4 * noise)
        self.sha_start = sha_params(list(self.params.items()) + list(self.buffers.items()))

    @torch.no_grad()
    def probe(self, model) -> torch.Tensor:
        return model(input_ids=self.probe_ids, use_cache=False).logits[0, -8:].float()

    @torch.no_grad()
    def changed(self, model) -> int:
        """Number of parameter tensors that differ from the pristine copy (diagnostic, before the reset)."""
        return sum(not torch.equal(p, self.params[n]) for n, p in model.named_parameters() if n in self.params)

    def reset(self, model) -> dict:
        for m in model.modules():
            for d in _HOOK_DICTS:
                getattr(m, d, {}).clear()
            if "forward" in m.__dict__:
                raise RuntimeError(f"the surface monkeypatched forward() of {type(m).__name__}; change parameter values only")
        if _spec(model) != self.spec:
            raise RuntimeError("the surface changed the model's modules/parameters/buffers; change parameter values only")
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.requires_grad_(False)
                p.grad = None
                p.copy_(self.params[n])
            for n, b in model.named_buffers():
                b.copy_(self.buffers[n])
        model.eval()
        equal = all(torch.equal(p, self.params[n]) for n, p in model.named_parameters()) and \
            all(torch.equal(b, self.buffers[n]) for n, b in model.named_buffers())
        diff = float((self.probe(model) - self.probe_ref).abs().max())
        return {"reset_ok": bool(equal and diff <= self.tol), "probe_diff": diff}

    def sha_now(self, model) -> str:
        return sha_params(list(model.named_parameters()) + list(model.named_buffers()))


def run_items(model, load_surface, docs: list[np.ndarray], items: list[dict], *, ttt_seconds: float, seed: int,
              device, log=print) -> tuple[dict, dict, dict]:
    """The frozen item loop. Returns (preds {id: token ids}, per-item records {id: {...}}, run stats)."""
    t_run = time.perf_counter()
    guard = WeightGuard(model, device)
    head = torch.tensor([HEAD_IDS], device=device)
    preds, per = {}, {}
    for idx, (doc, it) in enumerate(zip(docs, items)):
        t = time.perf_counter()
        surface = load_surface()                                   # fresh module state for every item
        load_s = time.perf_counter() - t
        doc_ids = torch.as_tensor(np.asarray(doc, dtype=np.int64), device=device).unsqueeze(0)
        q_ids = torch.as_tensor(np.asarray(it["q"], dtype=np.int64), device=device).unsqueeze(0)
        use_doc = bool(getattr(surface, "DOC_IN_CONTEXT", True))
        t = time.perf_counter()
        cache = prefill(model, doc_ids) if (use_doc or getattr(surface, "PREFILL_DOC", True)) else None
        _sync(device)
        prefill_s = time.perf_counter() - t
        t0 = time.perf_counter()
        ctx = TTTContext(doc_ids, q_ids, cache, seed * 1_000_003 + idx, t0 + ttt_seconds - load_s)
        with torch.enable_grad():
            ret = surface.adapt(model, ctx)
        _sync(device)
        ttt_s = load_s + time.perf_counter() - t0
        changed = guard.changed(model)
        t = time.perf_counter()
        if ret is not None:
            if not isinstance(ret, Cache):
                raise TypeError(f"adapt() must return None or a transformers Cache, got {type(ret).__name__}")
            ans_cache, prompt = ret, q_ids
        elif use_doc:
            ans_cache, prompt = ctx.prefix_cache(), q_ids
        else:
            ans_cache, prompt = None, torch.cat([head, q_ids], dim=1)
        cache_len = 0 if ans_cache is None else int(ans_cache.get_seq_length())
        with torch.no_grad():
            toks, bad = greedy(model, prompt, ans_cache)
        _sync(device)
        answer_s = time.perf_counter() - t
        del ctx, cache, ans_cache, ret, surface
        r = guard.reset(model)
        preds[it["id"]] = toks
        per[it["id"]] = {"ttt_s": ttt_s, "prefill_s": prefill_s, "answer_s": answer_s, "load_s": load_s,
                         "changed_tensors": changed, "cache_len": cache_len, "doc_in_context": use_doc,
                         "nonfinite": bad, **r}
        if idx % 20 == 0 or idx == len(items) - 1:
            log(f"item {idx + 1}/{len(items)} ttt {ttt_s:.2f}s prefill {prefill_s:.2f}s answer {answer_s:.2f}s "
                f"changed {changed} reset_ok {r['reset_ok']} elapsed {time.perf_counter() - t_run:.0f}s")
    stats = {"n_items": len(items), "ttt_seconds_limit": ttt_seconds, "seed": seed, "sha_start": guard.sha_start,
             "sha_end": guard.sha_now(model), "probe_tol": guard.tol,
             "all_reset_ok": all(v["reset_ok"] for v in per.values()),
             "nonfinite": sum(v["nonfinite"] for v in per.values()), "wall_s": time.perf_counter() - t_run}
    return preds, per, stats
