"""Frozen RUN entry point for memory-longmemeval.

For each question (concurrently): a fresh surface Memory gets a budgeted client scoped to that question and the
frozen tools, ingests every haystack session in order, then answers. Answers are cut to 32 words.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from arlab.lib.client import BudgetedClient, BudgetExceeded
from memtools import Tools

PER_QUESTION_TOKENS = 160_000   # fixed in IDEA.md before calibration (budget.limit = this × questions per split)
MAX_COMPLETION = 512
ANSWER_WORDS = 32
WORKERS = 16

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
a = ap.parse_args()
sys.path.insert(0, "/work")
import memory as surface  # noqa: E402  (the editable surface)

questions = [json.loads(line) for line in open("/data/public/questions.jsonl")]
client = BudgetedClient("http://llm:8000", "llm", PER_QUESTION_TOKENS, MAX_COMPLETION)


def one(q):
    try:
        mem = surface.Memory(client.scoped(q["id"]), Tools)
        for s in q["sessions"]:
            mem.ingest(s)
        ans = mem.answer(q["question"], q["question_date"])
        status = "ok"
    except BudgetExceeded:
        ans, status = "", "budget_exceeded"
    return {"id": q["id"], "answer": " ".join(str(ans).split()[:ANSWER_WORDS]), "status": status, "tokens": client.used(q["id"])}


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(one, questions))
with open(f"{a.out}/answers.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
json.dump({"service_tokens": client.total(), "calls": client.calls}, open(f"{a.out}/budget.json", "w"))
n_exc = sum(r["status"] != "ok" for r in results)
print(f"answered {len(results)} questions, {client.calls} calls, {client.total()} tokens, {n_exc} over budget")
