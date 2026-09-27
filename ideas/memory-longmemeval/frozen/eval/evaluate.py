"""Frozen EVALUATE for memory-longmemeval: deterministic normalized exact match, one item per question. No LLM judge."""
import argparse
import json
import os

from normalize import score

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gold", default="/data/private/gold.jsonl")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


gold = {g["id"]: g for g in map(json.loads, open(a.gold))}
try:
    rows = list(map(json.loads, open(f"{a.run}/answers.jsonl")))
    answers = {r["id"]: str(r["answer"]) if r.get("status") == "ok" else "" for r in rows}  # over budget / crashed → 0
except (OSError, ValueError, KeyError) as e:
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"answers.jsonl unreadable: {e}"})
    raise SystemExit(0)
items = {qid: score(answers.get(qid, ""), g) for qid, g in gold.items()}
acc = sum(items.values()) / len(items)
by_type = {}
for qid, g in gold.items():
    by_type.setdefault(g["type"], []).append(items[qid])
metrics = {"accuracy": acc, "answered": sum(1 for q in gold if answers.get(q, "").strip()) / len(gold)}
metrics.update({f"acc_{t}": sum(v) / len(v) for t, v in sorted(by_type.items())})
write({"valid": True, "primary": acc, "metrics": metrics, "items": items, "message": "ok"})
print(json.dumps(metrics))
