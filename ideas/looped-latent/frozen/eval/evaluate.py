"""Frozen EVALUATE for looped-latent: exact-match accuracy of the greedy answers against private answers, plus the
guard metrics. Scores come only from raw outputs (generated token ids, per-token NLLs from frozen code).

Primary: accuracy on the ID-step set (progen.ID_STEPS) (items = one 0/1 per problem). Guards (pack.yaml): hard_acc (progen.HARD_STEPS),
text_nll (OASST2 retention), depth_ratio (effective depth, inference cost), trainable_m.
"""
import argparse
import json
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


gold = json.load(open(f"{a.data}/private/answers.json"))
n_text = np.load(f"{a.data}/public/text.npy", mmap_mode="r").shape[0]
try:
    stats = json.load(open(f"{a.run}/stats.json"))
    preds = json.load(open(f"{a.run}/preds.json"))
    text_nll = np.asarray(np.load(f"{a.run}/text_nll.npy", allow_pickle=False), dtype=np.float64)
    base_nll = np.asarray(np.load(f"{a.run}/base_nll.npy", allow_pickle=False), dtype=np.float64)
except Exception as e:  # missing, wrong format, non-numeric
    invalid(f"missing or unreadable run outputs: {e!r}"[:500])

try:
    if a.limit:  # smoke runs only; taken from this frozen command line, never from the run outputs
        gold = {k: dict(list(v.items())[:a.limit]) for k, v in gold.items()}
        n_text = max(4, a.limit // 8)
    if stats.get("nan_at") is not None:
        invalid(f"training diverged (non-finite loss at step {stats['nan_at']})")
    if stats.get("nonfinite_decode_calls", 1) != 0:
        invalid("non-finite logits during decoding")
    if stats.get("causal") is not True:
        invalid(f"non-causal model: perturbing future tokens changed past log-probs by {stats.get('causal_max_diff')}")
    if text_nll.ndim != 2 or text_nll.shape[0] != n_text or base_nll.shape != text_nll.shape:
        invalid(f"text_nll shape {text_nll.shape} / base {base_nll.shape}, expected ({n_text}, T)")
    if not (np.isfinite(text_nll).all() and np.isfinite(base_nll).all()):
        invalid("non-finite text NLL")
    for kind in ("main", "hard"):
        got = preds.get(kind)
        if not isinstance(got, dict) or set(got) != set(gold[kind]):
            invalid(f"preds[{kind}] must have exactly the {len(gold[kind])} expected ids")
        for v in got.values():
            if not (isinstance(v, list) and len(v) <= MAX_NEW and all(isinstance(t, int) and 0 <= t < 200_000 for t in v)):
                invalid(f"preds[{kind}]: each answer must be a list of <= {MAX_NEW} token ids")
    for k in ("depth_ratio", "trainable_m", "gen_s", "train_steps", "examples_seen"):
        float(stats[k])
except SystemExit:
    raise
except Exception as e:
    invalid(f"malformed run outputs: {e!r}"[:500])

from transformers import AutoTokenizer  # noqa: E402  (only needed once the outputs are well-formed)

tok = AutoTokenizer.from_pretrained(a.tokenizer)
items, acc, by_steps = {}, {}, defaultdict(list)
for kind in ("main", "hard"):
    s = {i: score(answer_text(tok, preds[kind][i]), g["answer"]) for i, g in gold[kind].items()}
    acc[kind] = float(np.mean(list(s.values())))
    for i, g in gold[kind].items():
        by_steps[g["steps"]].append(s[i])
    if kind == "main":
        items = s
text, base = float(text_nll.mean()), float(base_nll.mean())
metrics = {"accuracy": acc["main"], "hard_acc": acc["hard"], "text_nll": text, "base_text_nll": base,
           "text_nll_vs_base": text / base, "depth_ratio": float(stats["depth_ratio"]),
           "trainable_m": float(stats["trainable_m"]), "extra_params_m": float(stats.get("extra_params_m", 0.0)),
           "gen_s": float(stats["gen_s"]), "train_steps": float(stats["train_steps"]),
           "examples_seen": float(stats["examples_seen"]), "train_loss_last50": float(stats.get("train_loss_last50", 0.0))}
metrics |= {f"acc_s{k}": float(np.mean(v)) for k, v in sorted(by_steps.items())}
write({"valid": True, "primary": acc["main"], "metrics": metrics, "items": items, "message": "ok"})
print(json.dumps(metrics))
