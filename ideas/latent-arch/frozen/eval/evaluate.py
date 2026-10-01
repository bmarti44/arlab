"""Frozen EVALUATE for latent-arch. This process never imports or runs surface code.

  integrity   RUN's trusted supervisor finished normally (not killed, train_seconds <= limit, no NaN); model.pt is
              the file it hashed; it is a data-only {name: tensor} dict (weights-only load) with the same tensor hash
  params_m    floating-point elements + bytes of non-floating tensors, over all checkpoint tensors, / 1e6
  params_bytes  bytes of all checkpoint tensors / 1e6 (bit-packing into wider dtypes gains nothing)
  worker      the surface's make_model() + the checkpoint run in a Landlock-sandboxed worker process
              (frozen/run/worker.py) that cannot read the eval data, this process's memory or the result dir; it
              receives token ids only and returns logits. After load + eval() and again after scoring its tensors must
              hash to the checkpoint and no unregistered tensor state may be reachable (else invalid).
  val_bpb     cross-entropy computed HERE from the worker's logits on the 2048 relabelled text rows (permuted bytes)
  acc_depth   PRIMARY (v2.1): full-vocab argmax at the answer position (`?`) of BOS + record (no generated tokens),
              computed here; equal-k mean over k 7..10 (2000 records per k); items = one 0/1 per depth record (8000).
              acc_id (k 1..6, equal-k mean) is a guard; accuracy = their mean and logp_* (mean log-prob of the right
              answer) are reported only
  FLOPs/ops   counted on EVERY scored forward (text and programs, actual shapes): matmul/attention by formula, every
              other non-view op by max(elements read, written); infer_flops_tok = mean of the per-token counts on text
              and on programs; any op outside aten/prims -> invalid (also in the warm-up and causality forwards)
  infer_s     wall time of the scored program forwards (after an untimed, audited warm-up on RANDOM tokens: no
              evaluation prompt reaches the scoring worker outside the scored pass)
  causality   the scored logits of the first text rows vs. a FRESH worker (never scores, never saw the originals) on
              the same rows with x[t+1:] randomized, 16 random cut points per row stratified over the whole row
              (the last stratum ends at T-2): log-probs at positions <= t must agree. Program rows are not checked:
              only the last position is scored and the answer is never in the input.
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

from common import SEQ_LEN, VOCAB, count_bytes, count_elements, file_sha256, load_checkpoint, tensors_hash
from evalcore import Worker, WorkerFailure, accuracy_report, nll_sum

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/data/private")
ap.add_argument("--work", default="/work")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-train-seconds", type=float, default=690)
ap.add_argument("--limit", type=int, default=0, help="tests only (never in pack.yaml): first N items per group")
a = ap.parse_args()
dev = a.device
TEXT_BATCH, PROG_BATCH, CAUSAL_ROWS, CAUSAL_CUTS, CAUSAL_TOL = 16, 128, 4, 16, 1e-3


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
ck_hash, params_m, params_bytes = stats["tensors_hash"], count_elements(ck) / 1e6, count_bytes(ck) / 1e6
del ck

# ------------------------------------------------------------------ data (stays in this process)
d = dict(np.load(f"{a.data}/programs.npz"))
rows = np.load(f"{a.data}/text_rows.npy")
tb = torch.as_tensor(np.load(f"{a.data}/token_bytes.npy").astype(np.int64), device=dev)
idx = np.arange(len(d["group"]))
text_batch, causal_rows = TEXT_BATCH, CAUSAL_ROWS
if a.limit:  # first N of each group; counterfactual twins whose originals are kept
    keep = np.zeros(len(idx), bool)
    for g in (0, 1, 2):
        keep[np.flatnonzero(d["group"] == g)[:a.limit]] = True
    keep |= (d["group"] == 3) & np.isin(d["cf_of"], np.flatnonzero(keep))
    idx = idx[keep]
    rows, text_batch, causal_rows = rows[:max(4, a.limit // 16)], 2, 2
prompts = d["tokens"][idx]
T, L = rows.shape[1] - 1, prompts.shape[1]
deny = [os.path.abspath(f"{a.data}/{f}") for f in ("programs.npz", "text_rows.npy", "token_bytes.npy")]
max_floats = text_batch * T * VOCAB
config = {"vocab_size": VOCAB, "seq_len": int(stats.get("seq_len", SEQ_LEN)), "device": dev}
g = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))   # unpredictable cut points and warm-up tokens
# Causality over the whole row: CAUSAL_CUTS cut points per row, one uniformly random t in each of CAUSAL_CUTS equal
# strata of [0, T-2] (the last stratum ends at T-2, so the last ~64 positions are always probed); the future
# x[t+1:] is replaced by random tokens and positions <= t must not change.
cases = []
for r in range(causal_rows):
    for j in range(CAUSAL_CUTS):
        lo, hi = j * (T - 1) // CAUSAL_CUTS, (j + 1) * (T - 1) // CAUSAL_CUTS - 1
        t = int(g.integers(lo, hi + 1))
        x2 = rows[r, :-1].copy()
        x2[t + 1:] = g.integers(0, VOCAB, T - t - 1)
        cases.append((r, t, x2))
warm = g.integers(0, VOCAB, (min(PROG_BATCH, len(prompts)), L)).astype(np.int64)   # never an evaluation prompt


def start():
    w = Worker(a.work, ckpt, dev, deny, max_floats)
    if w.hello.get("sandbox", {}).get("ok") is not True:
        w.close()
        invalid(f"model worker sandbox failed: {w.hello}")
    w.build(config, ck_hash)
    return w


# ------------------------------------------------------------------ measurements
bad_ops, flops, orig = set(), {"text": 0, "program": 0}, {}
try:
    w = start()
    try:
        bad_ops |= set(w.forward(warm, last=True)[2])                        # warm-up: random tokens, audited, untimed
        nats, nbytes, finite = 0.0, 0, True
        for i in range(0, len(rows), text_batch):
            r = torch.as_tensor(rows[i:i + text_batch].astype(np.int64), device=dev)
            lg, f, bad = w.forward(rows[i:i + text_batch, :-1])
            flops["text"] += f
            bad_ops |= set(bad)
            for j in range(len(r)):
                if i + j < causal_rows:                                      # the scored logits are the originals
                    orig[i + j] = torch.log_softmax(lg[j].float(), -1)
            n, b, ok = nll_sum(lg.to(dev), r[:, 1:], tb)
            nats, nbytes, finite = nats + n, nbytes + b, finite and ok
        if not finite:
            invalid("non-finite log-probs on text")
        preds, logps, t0 = [], [], time.monotonic()
        for i in range(0, len(prompts), PROG_BATCH):
            lg, f, bad = w.forward(prompts[i:i + PROG_BATCH], last=True)
            flops["program"] += f
            bad_ops |= set(bad)
            if not torch.isfinite(lg).all():
                invalid("non-finite logits at the answer position")
            preds.append(lg[:, -1].argmax(-1).numpy())
            ans = torch.as_tensor(d["answer"][idx[i:i + PROG_BATCH]].astype(np.int64))
            logps.append(torch.log_softmax(lg[:, -1].float(), -1).gather(1, ans[:, None])[:, 0].numpy())
        infer_s = time.monotonic() - t0
        w.check(ck_hash, "after scoring (state changed during evaluation)")
    finally:
        w.close()
    if bad_ops:
        invalid(f"rejected ops in the forward pass (outside aten/prims, or compute without a FLOP price): {sorted(bad_ops)[:5]}")
    w = start()                                                            # fresh process: has never seen the originals
    causal = 0.0
    try:
        for i in range(0, len(cases), text_batch):
            chunk = cases[i:i + text_batch]
            lg, _, bad = w.forward(np.stack([x2 for _, _, x2 in chunk]))
            bad_ops |= set(bad)
            lp = torch.log_softmax(lg, -1)
            for j, (r, t, _) in enumerate(chunk):
                diff = float((lp[j, :t + 1] - orig[r][:t + 1]).abs().max())
                if not diff <= CAUSAL_TOL:
                    invalid(f"non-causal model: replacing tokens after position {t} changed log-probs at positions "
                            f"<= {t} by {diff:.3g}")
                causal = max(causal, diff)
    finally:
        w.close()
    if bad_ops:
        invalid(f"rejected ops in the forward pass (outside aten/prims, or compute without a FLOP price): {sorted(bad_ops)[:5]}")
except WorkerFailure as e:
    invalid(e)
rep = accuracy_report(np.concatenate(preds), d, idx, np.concatenate(logps))
m = rep["metrics"]
tok_text, tok_prog = len(rows) * T, len(prompts) * L
m.update(val_bpb=nats / (math.log(2) * nbytes), params_m=params_m, params_bytes=params_bytes, infer_s=infer_s,
         flops_tok_text=flops["text"] / tok_text, flops_tok_prog=flops["program"] / tok_prog,
         infer_flops_tok=0.5 * (flops["text"] / tok_text + flops["program"] / tok_prog),
         causal_max_diff=causal, causal_cuts=len(cases), n_items=len(rep["items"]),
         train_seconds_supervisor=ts, stray_processes_killed=stats.get("stray_processes_killed"))
for k in ("train_steps", "tokens_seen", "program_tokens_seen", "build_s", "first_step_s", "train_loss_last50"):
    v = (stats.get("trainer") or {}).get(k)  # trainer-reported, informational only (never gated)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        m["run_" + k] = float(v)
if not all(math.isfinite(v) for v in m.values() if isinstance(v, float)):
    invalid(f"non-finite metric: {m}")
write({"valid": True, "primary": m["acc_depth"], "metrics": m, "items": rep["items"], "message": "ok"})
print(json.dumps(m))
