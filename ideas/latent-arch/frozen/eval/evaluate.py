"""Frozen EVALUATE for latent-arch.

Loads the checkpoint eagerly with the surface's load(), then, from raw logits only:
  integrity   model.pt sha256 and tensor hash equal what RUN recorded right after the deadline; budget <= limit
  params_m    parameters + floating-point buffers of the loaded model
  probe       FlopCounterMode + op-namespace audit on 8 text rows x 1024 + 256 programs of all depths
              -> infer_flops_tok; any op outside aten/prims -> invalid
  causality   arlab.lib.lm.causality_check on text rows and on program rows
  val_bpb     arlab.lib.lm.score_bpb on the 2048 relabelled text rows with the permuted token_bytes
  accuracy    full-vocab argmax at the answer position of BOS + program + `q?` (no generated tokens):
              0.5 * acc_id (k 1..6) + 0.5 * acc_depth (k 7..10); items = one 0/1 per program (4000)
  reported    acc per k (1..12), acc_ext (k 11..12), cf_both (counterfactual pairs), heuristic floors,
              infer_s (timed forward over every eval program)
"""
import argparse
import json
import math
import os
import sys
import traceback

import numpy as np
import torch

from arlab.lib.lm import causality_check, score_bpb
from common import VOCAB, count_params, file_sha256, tensor_hash
from evalcore import accuracy_report, predict, probe

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


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg[:2000]})
    print("INVALID:", msg[:2000])
    sys.exit(0)


# ------------------------------------------------------------------ run outputs and integrity
try:
    budget = json.load(open(f"{a.run}/budget.json"))
    stats = json.load(open(f"{a.run}/stats.json"))
    ts = float(budget["train_seconds"])
except Exception as e:
    invalid(f"missing or unreadable run outputs: {e!r}")
if not math.isfinite(ts) or ts > a.max_train_seconds:
    invalid(f"train_seconds {ts} > {a.max_train_seconds}")
if stats.get("nan_at") is not None:
    invalid(f"training diverged (non-finite loss at step {stats['nan_at']})")
if not os.path.isfile(f"{a.run}/model.pt") or file_sha256(f"{a.run}/model.pt") != stats.get("model_sha256"):
    invalid("model.pt changed after it was saved at the deadline (sha256 mismatch)")
sys.path.insert(0, a.work)
try:
    import train as surface
    model = surface.load(f"{a.run}/model.pt", dev)
    model.eval()
except Exception:
    invalid("could not load the checkpoint with the surface's load(): " + traceback.format_exc()[-800:])
if tensor_hash(model) != stats.get("tensor_hash"):
    invalid("the loaded weights differ from the weights RUN loaded after the deadline (tensor hash mismatch)")
params_m = count_params(model) / 1e6
if dev == "cpu":
    model.float()  # CPU smoke tests: no bf16 autocast on CPU, so run everything in fp32

# ------------------------------------------------------------------ data
d = dict(np.load(f"{a.data}/programs.npz"))
rows = np.load(f"{a.data}/text_rows.npy")
token_bytes = np.load(f"{a.data}/token_bytes.npy")
n_probe_rows, n_probe_prog = 8, 256
idx = np.arange(len(d["group"]))
if a.limit:  # first N of each group; counterfactual twins whose originals are kept
    keep = np.zeros(len(idx), bool)
    for g in (0, 1, 2):
        keep[np.flatnonzero(d["group"] == g)[:a.limit]] = True
    keep |= (d["group"] == 3) & np.isin(d["cf_of"], np.flatnonzero(keep))
    idx = idx[keep]
    rows, n_probe_rows, n_probe_prog = rows[:max(4, a.limit // 16)], 2, 16
probe_prog = np.random.default_rng(0).permutation(idx[d["group"][idx] < 3])[:n_probe_prog]

# ------------------------------------------------------------------ measurements
try:
    pr = probe(model, [rows[:n_probe_rows, :-1], d["tokens"][probe_prog]], dev)
    if pr["bad_ops"]:
        invalid(f"ops outside aten/prims in the forward pass (custom kernels are not allowed): {pr['bad_ops'][:5]}")
    c_text = causality_check(model, rows, VOCAB, device=dev)
    prog_rows = np.concatenate([d["tokens"][idx[:4]], np.zeros((min(4, len(idx)), 1), np.int64)], axis=1)
    c_prog = causality_check(model, prog_rows, VOCAB, device=dev)
    for name, c in (("text", c_text), ("program", c_prog)):
        if not c["causal"]:
            invalid(f"non-causal model on {name} rows: future tokens changed past log-probs by {c['max_diff']:.3g}")
    bpb = score_bpb(model, rows, token_bytes, device=dev)
    if not bpb["valid"]:
        invalid(bpb["message"])
    pred, infer_s = predict(model, d["tokens"][idx], dev)
except ValueError as e:
    invalid(str(e))
rep = accuracy_report(pred, d, idx)
m = rep["metrics"]
m.update(val_bpb=bpb["val_bpb"], params_m=params_m, infer_flops_tok=pr["flops_per_token"], infer_s=infer_s,
         probe_tokens=pr["tokens"], causal_max_diff_text=c_text["max_diff"], causal_max_diff_prog=c_prog["max_diff"],
         n_items=len(rep["items"]))
for k in ("train_steps", "tokens_seen", "program_tokens_seen", "build_s", "first_step_s", "train_loss_last50"):
    v = stats.get(k)  # RUN-reported, informational only (never gated)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        m["run_" + k] = float(v)
if not all(math.isfinite(v) for v in m.values() if isinstance(v, float)):
    invalid(f"non-finite metric: {m}")
write({"valid": True, "primary": m["accuracy"], "metrics": m, "items": rep["items"], "message": "ok"})
print(json.dumps(m))
