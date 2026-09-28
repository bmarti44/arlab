"""Frozen EVALUATE for ttt-context: exact-match accuracy of the greedy answers against the private answers.

Primary: accuracy over all items (items = one 0/1 per question). An item whose TTT time exceeded the per-item
budget (--ttt-seconds, from this frozen command line) scores 0. The run is invalid if any weight reset failed,
the weight SHA-256 at the end differs from the one at load, logits were non-finite, or outputs are malformed.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from common import MAX_NEW, MODEL_DIR  # frozen/run, on PYTHONPATH
from scoring import answer_text, score

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ttt-seconds", type=float, required=True, help="per-item TTT budget; must match the RUN command")
ap.add_argument("--data", default="/data")
ap.add_argument("--tokenizer", default=MODEL_DIR)
ap.add_argument("--limit", type=int, default=0, help="pilot/smoke only (never in pack.yaml): score the first N items")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg})
    sys.exit(0)


gold = json.load(open(f"{a.data}/private/answers.json"))
if a.limit:
    gold = dict(list(gold.items())[:a.limit])
try:
    stats = json.load(open(f"{a.run}/stats.json"))
    preds = json.load(open(f"{a.run}/preds.json"))
    per = json.load(open(f"{a.run}/items.json"))
except Exception as e:  # missing or unreadable
    invalid(f"missing or unreadable run outputs: {e!r}"[:500])

try:
    if int(stats.get("limit", 0)) != a.limit:
        invalid(f"run used --limit {stats.get('limit')} but evaluation --limit {a.limit}")
    if float(stats["ttt_seconds_limit"]) != a.ttt_seconds:
        invalid(f"run used --ttt-seconds {stats['ttt_seconds_limit']} but evaluation {a.ttt_seconds}")
    if not stats.get("sha_start") or not (stats["sha_start"] == stats.get("sha_end") == stats.get("sha_end_work")):
        invalid("model weights at the end differ from the weights at load (SHA-256)")
    if set(preds) != set(gold) or set(per) != set(gold):
        invalid(f"preds/items must have exactly the {len(gold)} expected ids")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    vocab = len(tok)
    for i in gold:
        v, r = preds[i], per[i]
        if not (type(v) is list and len(v) <= MAX_NEW and all(type(t) is int and 0 <= t < vocab for t in v)):
            invalid(f"{i}: an answer must be a list of <= {MAX_NEW} token ids in [0, {vocab})")
        if r.get("reset_ok") is not True:
            invalid(f"{i}: weight reset failed (probe diff {r.get('probe_diff')})")
        if type(r.get("nonfinite")) is not int or r["nonfinite"] != 0:
            invalid(f"{i}: non-finite logits while answering (or a malformed count)")
        for k in ("ttt_s", "prefill_s", "answer_s"):
            if not (type(r.get(k)) in (int, float) and math.isfinite(r[k]) and r[k] >= 0):
                invalid(f"{i}: bad timing {k}={r.get(k)!r}")
    texts = {i: answer_text(tok.decode(preds[i], skip_special_tokens=False)) for i in gold}
except SystemExit:
    raise
except Exception as e:
    invalid(f"malformed run outputs: {e!r}"[:500])

items, by_kind, over = {}, defaultdict(list), 0
for i, g in gold.items():
    s = score(texts[i], g["aliases"])
    if per[i]["ttt_s"] > a.ttt_seconds:
        s, over = 0.0, over + 1
    items[i] = s
    by_kind[g["kind"]].append(s)
tt = np.array([per[i]["ttt_s"] for i in gold])
metrics = {"accuracy": float(np.mean(list(items.values()))), "over_budget_items": float(over),
           "ttt_s_mean": float(tt.mean()), "ttt_s_max": float(tt.max()),
           "prefill_s_mean": float(np.mean([per[i]["prefill_s"] for i in gold])),
           "answer_s_mean": float(np.mean([per[i]["answer_s"] for i in gold])),
           "changed_tensors_mean": float(np.mean([per[i].get("changed_tensors", 0) for i in gold])),
           "doc_in_context_frac": float(np.mean([bool(per[i].get("doc_in_context")) for i in gold])),
           "probe_diff_max": float(max(per[i].get("probe_diff", 0.0) for i in gold))}
metrics |= {f"acc_{k}": float(np.mean(v)) for k, v in sorted(by_kind.items())}
write({"valid": True, "primary": metrics["accuracy"], "metrics": metrics, "items": items, "message": "ok"})
print(json.dumps(metrics))
