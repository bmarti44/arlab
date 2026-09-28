"""Frozen RUN entry point for stencil-focus.

For each item: the planner subprocess (frozen/run/planner.py) calls the surface's plan(sentences, current, query,
window); this process re-checks the plan, builds the exact prompt, then generates once with Qwen3-4B (vLLM
service `llm`, raw token-id prompt, temperature 0, --max-new cap). The surface never sees the model. A plan that fails the check (or raises) is recorded
as a plan error, which makes the whole run `invalid` in EVALUATE. The holdout also carries the free real
MemoryCode dialogues (real.jsonl), run the same way at W = 1,536 (descriptive only).
Writes /out/outputs.jsonl and /out/budget.json.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import window
from planner import load_items

STOP_IDS = [151645, 151643]            # <|im_end|>, <|endoftext|>
URL = "http://llm:8000/v1/completions"
REQUEST_TIMEOUT = 900

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)        # recorded only: generation is greedy (see IDEA.md)
ap.add_argument("--split", required=True)
ap.add_argument("--max-new", type=int, required=True)     # generation cap, fixed by the pilot
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--plans", default=None, help="build/pilot.sh only: JSON {item_id: plan} replacing the surface")
a = ap.parse_args()


# ---- plan: the surface runs only in a separate planner process (frozen/run/planner.py); this process never
# imports it, re-checks every returned plan against its own Window and builds every prompt itself.
items = load_items()
t0 = time.time()
planner_error = None
if a.plans:
    raw_plans = {k: {"plan": v} for k, v in json.load(open(a.plans)).items()}
else:
    fd, tmp = tempfile.mkstemp(prefix="plans-", suffix=".json", dir="/tmp")
    os.close(fd)
    p = subprocess.run([sys.executable, "/frozen/planner.py", "--out", tmp], cwd="/tmp")
    raw_plans = json.load(open(tmp)) if p.returncode == 0 else {}
    planner_error = None if p.returncode == 0 else f"planner exited with {p.returncode}"
rows = []
for it, w, kind in items:
    row = {"id": it["id"], "kind": kind, "W": w}
    try:
        got = raw_plans.get(it["id"])
        if got is None:
            raise window.PlanError(planner_error or "planner returned no plan")
        if "error" in got:
            raise window.PlanError("plan() raised " + str(got["error"]))
        win = window.Window(it, w)
        plan = win.check(got["plan"])
        built = win.build(plan)
        row.update(plan={"ids": plan.ids, "header": plan.header, "placement": plan.placement, "budget": plan.budget},
                   prompt=built["prompt"], prompt_tokens=built["prompt_tokens"], reminder_tokens=built["reminder_tokens"],
                   n_reminder=len(plan.ids), n_sentences=len(win.sentences))
    except Exception as e:  # noqa: BLE001  (recorded; EVALUATE marks the run invalid)
        row.update(plan_error=f"{type(e).__name__}: {e}"[:1000])
    rows.append(row)
plan_s = time.time() - t0


# ---- generate (concurrent)
def generate(row):
    if "plan_error" in row:
        return row
    ids = window.tokenizer().encode(row["prompt"]).ids
    body = json.dumps({"model": "llm", "prompt": ids, "max_tokens": a.max_new, "temperature": 0.0, "seed": 0,
                       "stop_token_ids": STOP_IDS, "skip_special_tokens": True}).encode()
    t = time.time()
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                out = json.loads(r.read())
            ch, u = out["choices"][0], out.get("usage") or {}
            row.update(text=ch.get("text") or "", finish_reason=ch.get("finish_reason"), timed_out=False,
                       usage_prompt=int(u.get("prompt_tokens", 0)), completion_tokens=int(u.get("completion_tokens", 0)),
                       gen_s=round(time.time() - t, 2))
            return row
        except TimeoutError:
            break
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                break
            if isinstance(e, urllib.error.HTTPError) and e.code == 400:
                row.update(gen_error=f"model rejected the request: {e.read()[:300]!r}")
                return row
            if attempt < 2:
                time.sleep(2 ** attempt)
    row.update(text="", finish_reason=None, timed_out=True, usage_prompt=0, completion_tokens=0, gen_s=round(time.time() - t, 2))
    return row


t1 = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    rows = list(ex.map(generate, rows))
gen_s = time.time() - t1

os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/outputs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
used = sum(r.get("usage_prompt", 0) + r.get("completion_tokens", 0) for r in rows)
json.dump({"service_tokens": used}, open(f"{a.out}/budget.json", "w"))
json.dump({"seed": a.seed, "split": a.split, "max_new": a.max_new, "plan_s": round(plan_s, 1), "gen_s": round(gen_s, 1),
           "n": len(rows), "plan_errors": sum("plan_error" in r for r in rows),
           "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows)}, open(f"{a.out}/run_info.json", "w"))
print(f"{len(rows)} items, plan {plan_s:.0f}s, generate {gen_s:.0f}s, {used} service tokens, "
      f"{sum('plan_error' in r for r in rows)} plan errors, {sum(bool(r.get('timed_out')) for r in rows)} timeouts")
