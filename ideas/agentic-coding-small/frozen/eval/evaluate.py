"""Frozen EVALUATE for agentic-coding-small: each task's hidden tests run in the arlab clean room on the declared
source files from /run_out/ws/<id>. Item = 1 iff pytest exits 0 and the junit report lists exactly the expected
test IDs, all passed. Also reports timeout_rate (tasks that hit the wall-clock limit) for the guard."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arlab.lib.cleanroom import run_hidden_tests

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


priv = Path("/data/private")
tasks = [json.loads(line) for line in open(priv / "tasks.jsonl")]
run = Path(a.run)
try:
    recs = {r["id"]: r for r in map(json.loads, open(run / "tasks.jsonl"))}
    status = {i: r["status"] for i, r in recs.items()}
except (OSError, ValueError, KeyError) as e:
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"tasks.jsonl unreadable: {e}"})
    raise SystemExit(0)


SCORED = ("finished", "returned", "step_limit", "time_limit")  # the agent stopped normally or hit a step/time limit


def score(t):
    ws = run / "ws" / t["id"]
    if not ws.is_dir() or status.get(t["id"]) not in SCORED:  # missing record, over budget or crashed → 0
        return t["id"], 0.0
    return t["id"], run_hidden_tests(ws, t["sources"], priv / "tests" / t["id"], t["expected"], timeout_s=120)["score"]


with ThreadPoolExecutor(8) as ex:
    items = dict(ex.map(score, tasks))
n = len(tasks)
metrics = {"pass_rate": sum(items.values()) / n,
           "timeout_rate": sum(bool(recs.get(t["id"], {"timed_out": True}).get("timed_out")) for t in tasks) / n}
for d in sorted({t["difficulty"] for t in tasks}):
    ds = [items[t["id"]] for t in tasks if t["difficulty"] == d]
    metrics[f"pass_{d}"] = sum(ds) / len(ds)
for s in ("finished", "returned", "budget_exceeded", "step_limit", "time_limit"):
    metrics[f"frac_{s}"] = sum(status.get(t["id"]) == s for t in tasks) / n
write({"valid": True, "primary": metrics["pass_rate"], "metrics": metrics, "items": items, "message": "ok"})
print(json.dumps(metrics))
