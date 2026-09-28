"""Frozen EVALUATE for latent-arch. This process never imports or runs surface code.

  integrity   RUN's trusted supervisor finished normally (not killed, train_seconds <= limit, no NaN); model.pt is
              the file it hashed; it is a data-only {name: tensor} dict (weights-only load) with the same tensor hash
  params_m    every element of every checkpoint tensor (all dtypes), / 1e6
  worker      the surface's make_model() + the checkpoint run in a Landlock-sandboxed worker process
              (frozen/run/worker.py) that cannot read the eval data, this process's memory or the result dir; it
              receives token ids only and returns logits
  val_bpb     cross-entropy computed HERE from the worker's logits on the 2048 relabelled text rows (permuted bytes)
  accuracy    full-vocab argmax at the answer position of BOS + program + `q?` (no generated tokens), computed here:
              0.5 * acc_id (k 1..6) + 0.5 * acc_depth (k 7..10); items = one 0/1 per program (4000)
  FLOPs/ops   counted on EVERY scored forward (text and programs, actual shapes); infer_flops_tok = mean of the
              per-token counts on text and on programs; any op outside aten/prims -> invalid
  infer_s     wall time of the scored program forwards (after an untimed warm-up on a copy of the first batch)
  causality   original rows through the scoring worker vs. perturbed-future rows through a FRESH worker (no cache
              can connect them): log-probs at positions <= t must agree, on text and on program rows
  reported    acc per k (1..12), acc_ext (k 11..12), cf_both (counterfactual pairs), heuristic floors
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from common import SEQ_LEN, VOCAB, count_elements, file_sha256, load_checkpoint, tensors_hash
from evalcore import Worker, WorkerFailure, accuracy_report, nll_sum

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/data/private")
ap.add_argument("--work", default="/work")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-train-seconds", type=float, default=360)
ap.add_argument("--limit", type=int, default=0, help="tests only (never in pack.yaml): first N items per group")
a = ap.parse_args()
dev = a.device
TEXT_BATCH, PROG_BATCH, CAUSAL_ROWS, CAUSAL_TOL = 16, 128, 4, 1e-3


def write(obj):
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": str(msg)[:3000]})
    print("INVALID:", str(msg)[:3000])
    sys.exit(0)


# ------------------------------------------------------------------ RUN outputs and integrity (trusted supervisor)
try:
    budget = json.load(open(f"{a.run}/budget.json"))
    stats = json.load(open(f"{a.run}/stats.json"))
    ts = float(budget["train_seconds"])
except Exception as e:
    invalid(f"missing or unreadable run outputs: {e!r}")
if stats.get("status") != "ok":
    invalid(f"RUN supervisor status {stats.get('status')!r} (killed={stats.get('killed')})")
if not math.isfinite(ts) or ts > a.max_train_seconds or ts != stats.get("train_seconds"):
    invalid(f"train_seconds {ts} > {a.max_train_seconds} or inconsistent with stats.json")
if (stats.get("trainer") or {}).get("nan_at") is not None:
    invalid(f"training diverged (non-finite loss at step {stats['trainer']['nan_at']})")
ckpt = f"{a.run}/model.pt"
if os.path.islink(ckpt) or not os.path.isfile(ckpt) or file_sha256(ckpt) != stats.get("model_sha256"):
    invalid("model.pt is not the checkpoint the RUN supervisor hashed after the deadline")
try:
    ck = load_checkpoint(ckpt)
except ValueError as e:
    invalid(e)
if tensors_hash(ck) != stats.get("tensors_hash"):
    invalid("checkpoint tensors differ from those recorded at the deadline")
params_m = count_elements(ck) / 1e6
del ck

# ------------------------------------------------------------------ data (stays in this process)
d = dict(np.load(f"{a.data}/programs.npz"))
rows = np.load(f"{a.data}/text_rows.npy")
tb = torch.as_tensor(np.load(f"{a.data}/token_bytes.npy").astype(np.int64), device=dev)
idx = np.arange(len(d["group"]))
text_batch = TEXT_BATCH
if a.limit:  # first N of each group; counterfactual twins whose originals are kept
    keep = np.zeros(len(idx), bool)
    for g in (0, 1, 2):
        keep[np.flatnonzero(d["group"] == g)[:a.limit]] = True
    keep |= (d["group"] == 3) & np.isin(d["cf_of"], np.flatnonzero(keep))
    idx = idx[keep]
    rows, text_batch = rows[:max(4, a.limit // 16)], 2
prompts = d["tokens"][idx]
T, L = rows.shape[1] - 1, prompts.shape[1]
deny = [os.path.abspath(f"{a.data}/{f}") for f in ("programs.npz", "text_rows.npy", "token_bytes.npy")]
max_floats = max(text_batch, CAUSAL_ROWS) * T * VOCAB
config = {"vocab_size": VOCAB, "seq_len": int(stats.get("seq_len", SEQ_LEN)), "device": dev}
g = np.random.default_rng(0)
causal_x = {"text": rows[:CAUSAL_ROWS, :-1], "program": prompts[:CAUSAL_ROWS]}
perturbed = {k: [] for k in causal_x}
for k, x in causal_x.items():
    for t in (x.shape[1] // 4, x.shape[1] // 2, (3 * x.shape[1]) // 4):
        x2 = x.copy()
        x2[:, t + 1:] = g.integers(0, VOCAB, x2[:, t + 1:].shape)
        perturbed[k].append((t, x2))


def start():
    w = Worker(a.work, ckpt, dev, deny, max_floats)
    if w.hello.get("sandbox", {}).get("ok") is not True:
        w.close()
        invalid(f"model worker sandbox failed: {w.hello}")
    w.build(config)
    return w


# ------------------------------------------------------------------ measurements
bad_ops, flops = set(), {"text": 0, "program": 0}
try:
    w = start()
    try:
        w.forward(prompts[:PROG_BATCH].copy(), last=True)                    # warm-up: untimed, not scored
        nats, nbytes, finite = 0.0, 0, True
        for i in range(0, len(rows), text_batch):
            r = torch.as_tensor(rows[i:i + text_batch].astype(np.int64), device=dev)
            lg, f, bad = w.forward(rows[i:i + text_batch, :-1])
            flops["text"] += f
            bad_ops |= set(bad)
            n, b, ok = nll_sum(lg.to(dev), r[:, 1:], tb)
            nats, nbytes, finite = nats + n, nbytes + b, finite and ok
        if not finite:
            invalid("non-finite log-probs on text")
        preds, t0 = [], time.monotonic()
        for i in range(0, len(prompts), PROG_BATCH):
            lg, f, bad = w.forward(prompts[i:i + PROG_BATCH], last=True)
            flops["program"] += f
            bad_ops |= set(bad)
            if not torch.isfinite(lg).all():
                invalid("non-finite logits at the answer position")
            preds.append(lg[:, -1].argmax(-1).numpy())
        infer_s = time.monotonic() - t0
        orig = {k: torch.log_softmax(w.forward(x)[0], -1) for k, x in causal_x.items()}
    finally:
        w.close()
    if bad_ops:
        invalid(f"ops outside aten/prims in the forward pass (custom kernels are not allowed): {sorted(bad_ops)[:5]}")
    w = start()                                                            # fresh process: has never seen the originals
    causal = {}
    try:
        for k, cases in perturbed.items():
            causal[k] = max(float((torch.log_softmax(w.forward(x2)[0], -1)[:, :t + 1] - orig[k][:, :t + 1]).abs().max())
                            for t, x2 in cases)
    finally:
        w.close()
except WorkerFailure as e:
    invalid(e)
for k, diff in causal.items():
    if not diff <= CAUSAL_TOL:
        invalid(f"non-causal model on {k} rows: future tokens changed past log-probs by {diff:.3g}")
rep = accuracy_report(np.concatenate(preds), d, idx)
m = rep["metrics"]
tok_text, tok_prog = len(rows) * T, len(prompts) * L
m.update(val_bpb=nats / (math.log(2) * nbytes), params_m=params_m, infer_s=infer_s,
         flops_tok_text=flops["text"] / tok_text, flops_tok_prog=flops["program"] / tok_prog,
         infer_flops_tok=0.5 * (flops["text"] / tok_text + flops["program"] / tok_prog),
         causal_max_diff_text=causal["text"], causal_max_diff_prog=causal["program"], n_items=len(rep["items"]),
         train_seconds_supervisor=ts, stray_processes_killed=stats.get("stray_processes_killed"))
for k in ("train_steps", "tokens_seen", "program_tokens_seen", "build_s", "first_step_s", "train_loss_last50"):
    v = (stats.get("trainer") or {}).get(k)  # trainer-reported, informational only (never gated)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        m["run_" + k] = float(v)
if not all(math.isfinite(v) for v in m.values() if isinstance(v, float)):
    invalid(f"non-finite metric: {m}")
write({"valid": True, "primary": m["accuracy"], "metrics": m, "items": rep["items"], "message": "ok"})
print(json.dumps(m))
