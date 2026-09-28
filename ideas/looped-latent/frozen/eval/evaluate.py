"""Frozen EVALUATE for looped-latent: exact-match accuracy of the greedy answers against private answers, plus the
guard metrics. Scores come only from raw outputs (generated token ids, per-token NLLs from frozen code).

Primary: accuracy on the in-distribution set (items = one 0/1 per problem). Guards (pack.yaml): hard_acc (longer
programs), loop_gain (accuracy minus the same trained model with the loop disabled), text_nll (OASST2 retention),
depth_ratio (effective depth, inference cost), trainable_m. Integrity failures, divergence, non-causal models and
malformed outputs make the run invalid.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from common import MAX_NEW, MODEL_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/data")
ap.add_argument("--tokenizer", default=MODEL_DIR)
ap.add_argument("--limit", type=int, default=0, help="smoke tests only (never in pack.yaml): score the first N items")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg})
    sys.exit(0)


def answer_text(tok, ids) -> str:
    """Decoded answer up to the first stop token; whitespace-stripped. Anything else (words, hedges) stays in."""
    s = tok.decode(ids, skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|endoftext|>"):
        s = s.split(stop)[0]
    return s.strip()


def score(pred: str, gold: str) -> float:
    return float(pred == gold)


def stat(stats, key, lo, hi, integer=False) -> float:
    v = stats.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"stats[{key!r}] = {v!r} is not a finite number in [{lo}, {hi}]")
    if integer and v != int(v):
        raise ValueError(f"stats[{key!r}] = {v!r} is not an integer")
    return float(v)


gold = json.load(open(f"{a.data}/private/answers.json"))
text_rows = np.load(f"{a.data}/public/text.npy", mmap_mode="r")
n_text, t_text = text_rows.shape[0], text_rows.shape[1] - 1
if a.limit:  # smoke runs only; taken from this frozen command line, never from the run outputs
    gold = {k: dict(list(v.items())[:a.limit]) for k, v in gold.items()}
    n_text = max(4, a.limit // 8)
gold["main_noloop"] = gold["main"]
try:
    stats = json.load(open(f"{a.run}/stats.json"))
    preds = json.load(open(f"{a.run}/preds.json"))
    text_nll = np.asarray(np.load(f"{a.run}/text_nll.npy", allow_pickle=False), dtype=np.float64)
    base_nll = np.asarray(np.load(f"{a.run}/base_nll.npy", allow_pickle=False), dtype=np.float64)
except Exception as e:  # missing, wrong format, non-numeric
    invalid(f"missing or unreadable run outputs: {e!r}"[:500])

try:
    if not isinstance(stats, dict) or not isinstance(preds, dict):
        invalid("stats.json and preds.json must be objects")
    if stats.get("base_unchanged") is not True:
        invalid("base weights changed during the run (only LoRA and loop parameters may train)")
    if stats.get("unchanged_after_train") is not True:
        invalid("parameters or buffers changed after the training deadline")
    if stats.get("nan_at") is not None:
        invalid(f"training diverged (non-finite loss at step {stats['nan_at']})")
    if stats.get("nonfinite_decode_calls") != 0:
        invalid("non-finite logits during decoding")
    if stats.get("causal") is not True:
        invalid(f"non-causal model: perturbing future tokens changed past log-probs by {stats.get('causal_max_diff')}")
    for name, arr in (("text_nll", text_nll), ("base_nll", base_nll)):
        if arr.shape != (n_text, t_text) or arr.size == 0:
            invalid(f"{name} shape {arr.shape}, expected ({n_text}, {t_text})")
        if not np.isfinite(arr).all() or arr.min() < -1e-6:
            invalid(f"{name} must be finite and non-negative")
    if not base_nll.mean() > 0:
        invalid(f"base NLL mean {base_nll.mean()} is not positive")
    for kind in ("main", "hard", "main_noloop"):
        got = preds.get(kind)
        if not isinstance(got, dict) or set(got) != set(gold[kind]) or not got:
            invalid(f"preds[{kind}] must have exactly the {len(gold[kind])} expected ids")
        for v in got.values():
            if not (isinstance(v, list) and len(v) <= MAX_NEW and all(type(t) is int and 0 <= t < 200_000 for t in v)):
                invalid(f"preds[{kind}]: each answer must be a list of <= {MAX_NEW} token ids")
    guard = {"depth_ratio": stat(stats, "depth_ratio", 0.5, 100.0), "trainable_m": stat(stats, "trainable_m", 0.0, 1e4),
             "gen_s": stat(stats, "gen_s", 0.0, 1e6), "train_steps": stat(stats, "train_steps", 0, 1e9, integer=True),
             "examples_seen": stat(stats, "examples_seen", 0, 1e12, integer=True)}
    loss50 = stats.get("train_loss_last50")
    guard["train_loss_last50"] = float(loss50) if isinstance(loss50, (int, float)) and math.isfinite(loss50) else -1.0
except SystemExit:
    raise
except Exception as e:
    invalid(f"malformed run outputs: {e!r}"[:500])

from transformers import AutoTokenizer  # noqa: E402  (only needed once the outputs are well-formed)

tok = AutoTokenizer.from_pretrained(a.tokenizer)
items, acc, by_steps = {}, {}, defaultdict(list)
for kind in ("main", "hard", "main_noloop"):
    s = {i: score(answer_text(tok, preds[kind][i]), g["answer"]) for i, g in gold[kind].items()}
    acc[kind] = float(np.mean(list(s.values())))
    if kind != "main_noloop":
        for i, g in gold[kind].items():
            by_steps[g["steps"]].append(s[i])
    if kind == "main":
        items = s
text, base = float(text_nll.mean()), float(base_nll.mean())
metrics = {"accuracy": acc["main"], "hard_acc": acc["hard"], "acc_noloop": acc["main_noloop"],
           "loop_gain": acc["main"] - acc["main_noloop"], "text_nll": text, "base_text_nll": base,
           "text_nll_vs_base": text / base} | guard
metrics |= {f"acc_s{k}": float(np.mean(v)) for k, v in sorted(by_steps.items())}
write({"valid": True, "primary": acc["main"], "metrics": metrics, "items": items, "message": "ok"})
print(json.dumps(metrics))
