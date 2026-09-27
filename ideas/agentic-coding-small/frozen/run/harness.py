"""Frozen RUN entry point for agentic-coding-small.

Each task (up to WORKERS concurrently) gets a fresh working directory /out/ws/<id> holding its starter files, a
budgeted client scoped to the task (PER_TASK_TOKENS, completions ≤ MAX_COMPLETION) and frozen Tools (≤ MAX_STEPS
calls, TASK_SECONDS wall clock). The surface's solve(task, llm, tools) works there; EVALUATE later runs the hidden
tests on the declared source files in /out/ws/<id>.
"""
import argparse
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arlab.lib.client import BudgetedClient, BudgetExceeded
from codetools import StepLimit, TimeLimit, Tools

PER_TASK_TOKENS = 400_000   # fixed in IDEA.md before calibration (budget.limit = this × 40 tasks per split)
MAX_COMPLETION = 2048
MAX_STEPS = 30
TASK_SECONDS = 2400  # safety net for runaway tasks; the budgets are MAX_STEPS and PER_TASK_TOKENS
WORKERS = 20  # two waves; keeps per-request latency (~9 tok/s per stream) far below the client timeout

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
a = ap.parse_args()
sys.path.insert(0, "/work")
import agent as surface  # noqa: E402  (the editable surface)

tasks = [json.loads(line) for line in open("/data/public/tasks.jsonl")]
client = BudgetedClient("http://llm:8000", "llm", PER_TASK_TOKENS, MAX_COMPLETION, timeout_s=1800)  # a timed-out request would be re-sent
ws_root = Path(a.out) / "ws"


class TimedClient:
    """The task's scoped client; model calls past the task's wall-clock deadline raise TimeLimit."""

    def __init__(self, scoped, deadline):
        self._s, self._deadline = scoped, deadline

    def chat(self, messages, max_tokens=None, stop=None):
        if time.monotonic() > self._deadline:
            raise TimeLimit("task wall-clock limit reached")
        return self._s.chat(messages, max_tokens, stop)

    @property
    def tokens_left(self):
        return self._s.tokens_left


def one(t):
    ws = ws_root / t["id"]
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    for rel, content in t["starter"].items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(content)
    t0 = time.monotonic()
    tools = Tools(ws, MAX_STEPS, t0 + TASK_SECONDS)
    view = {"id": t["id"], "title": t["title"], "instructions": t["instructions"], "files": sorted(t["starter"])}
    try:
        surface.solve(view, TimedClient(client.scoped(t["id"]), t0 + TASK_SECONDS), tools)
        status = "finished" if tools.done else "returned"
    except BudgetExceeded:
        status = "budget_exceeded"
    except StepLimit:
        status = "step_limit"
    except TimeLimit:
        status = "time_limit"
    except Exception:  # noqa: BLE001  a crashing agent loses the task, not the run
        status = "error: " + traceback.format_exc(limit=2)[-300:]
    tools.kill_all()
    secs = time.monotonic() - t0
    timed_out = status == "time_limit" or secs > TASK_SECONDS
    if client.used(t["id"]) > PER_TASK_TOKENS:  # the surface may have swallowed BudgetExceeded
        status = "budget_exceeded"
    return {"id": t["id"], "status": status, "timed_out": timed_out, "steps": tools.steps, "seconds": round(secs, 1),
            "tokens": client.used(t["id"])}


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(one, tasks))
with open(f"{a.out}/tasks.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
json.dump({"service_tokens": client.total(), "calls": client.calls}, open(f"{a.out}/budget.json", "w"))
st = {}
for r in results:
    k = r["status"].split(":")[0]
    st[k] = st.get(k, 0) + 1
print(f"{len(results)} tasks, {client.calls} calls, {client.total()} tokens, statuses {st}")
