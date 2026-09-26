"""PREPARE for agentic-coding-small: split the validated tasks in /prepare/tasks (frozen/prepare, never mounted in
RUN) into equal validation / holdout halves, stratified by difficulty with a fixed seed.

/data/<split>/public/tasks.jsonl     {id, title, difficulty, instructions, starter: {path: content}}
/data/<split>/private/tasks.jsonl    {id, difficulty, sources, expected}
/data/<split>/private/tests/<id>/    the hidden test file
"""
import argparse
import json
import random
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
src, out = Path("/prepare/tasks"), Path(a.out)

tasks = []
for td in sorted(p for p in src.iterdir() if p.is_dir()):
    t = json.loads((td / "task.json").read_text())
    t["starter"] = {p.relative_to(td / "starter").as_posix(): p.read_text() for p in sorted((td / "starter").rglob("*")) if p.is_file()}
    t["dir"] = td
    tasks.append(t)

by = {}
for t in tasks:
    by.setdefault(t["difficulty"], []).append(t)
rng = random.Random(20260926)
split = {"validation": [], "holdout": []}
flip = 0
for d in sorted(by):
    group = sorted(by[d], key=lambda t: t["id"])
    rng.shuffle(group)
    for t in group:  # alternate, carrying the odd one over between strata so the halves stay equal
        split[("validation", "holdout")[flip]].append(t)
        flip ^= 1
n = min(len(v) for v in split.values())
for name, ts in split.items():
    ts = sorted(ts, key=lambda t: t["id"])[:n]
    (out / name / "public").mkdir(parents=True)
    (out / name / "private" / "tests").mkdir(parents=True)
    with open(out / name / "public" / "tasks.jsonl", "w") as fp, open(out / name / "private" / "tasks.jsonl", "w") as fq:
        for t in ts:
            fp.write(json.dumps({k: t[k] for k in ("id", "title", "difficulty", "instructions", "starter")}) + "\n")
            fq.write(json.dumps({k: t[k] for k in ("id", "difficulty", "sources", "expected")}) + "\n")
            shutil.copytree(t["dir"] / "tests", out / name / "private" / "tests" / t["id"])
json.dump({"validation": n, "holdout": n}, open(out / "splits.json", "w"))
print(f"{len(tasks)} tasks -> {n} validation / {n} holdout; difficulties { {d: len(v) for d, v in by.items()} }")
