"""PREPARE for memory-longmemeval: LongMemEval _s (cleaned, pinned revision), short-span questions only.

Drops single-session-preference (rubric-style gold) and every question whose gold has an alias longer than
5 normalized words, records what was dropped, then splits the rest stratified by type (fixed seed) into equal
validation / holdout halves. Haystacks go to public/ (RUN may read them), gold aliases to private/.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import urllib.request

from normalize import aliases

REV = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
URL = f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{REV}/longmemeval_s_cleaned.json"
MAX_WORDS = 5

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
out = ap.parse_args().out
urllib.request.urlretrieve(URL, "/tmp/s.json")
data = json.load(open("/tmp/s.json"))
kept, dropped = [], collections.Counter()
for x in data:
    abst = x["question_id"].endswith("_abs")
    stratum = x["question_type"] + ("_abs" if abst else "")
    if x["question_type"] == "single-session-preference":
        dropped[stratum + ":rubric_gold"] += 1
        continue
    al = [] if abst else aliases(x["answer"])
    if not abst and (not al or any(len(a.split()) > MAX_WORDS for a in al)):
        dropped[stratum + ":long_gold"] += 1
        continue
    kept.append((stratum, x, al, abst))
by = collections.defaultdict(list)
for k in kept:
    by[k[0]].append(k)
rng = random.Random(0)
split = {"validation": [], "holdout": []}
for stratum in sorted(by):
    items = sorted(by[stratum], key=lambda k: k[1]["question_id"])
    rng.shuffle(items)
    for i, k in enumerate(items):
        split["validation" if i % 2 == 0 else "holdout"].append(k)
n = min(len(v) for v in split.values())
for name in split:                       # equal sizes so one absolute token budget fits both splits
    split[name] = sorted(split[name], key=lambda k: k[1]["question_id"])[:n]
for name, items in split.items():
    os.makedirs(f"{out}/{name}/public")
    os.makedirs(f"{out}/{name}/private")
    with open(f"{out}/{name}/public/questions.jsonl", "w") as fq, open(f"{out}/{name}/private/gold.jsonl", "w") as fg:
        for stratum, x, al, abst in items:
            # public turns carry only role + content (the dataset also marks answer-bearing turns: has_answer)
            sessions = [{"session_id": f"s{i}", "date": date, "turns": [{"role": t["role"], "content": t["content"]} for t in turns]}
                        for i, (date, turns) in enumerate(zip(x["haystack_dates"], x["haystack_sessions"]))]
            qid = hashlib.sha256(x["question_id"].encode()).hexdigest()[:12]   # opaque: original ids encode abstention (_abs)
            fq.write(json.dumps({"id": qid, "question": x["question"], "question_date": x["question_date"],
                                 "sessions": sessions}) + "\n")
            fg.write(json.dumps({"id": qid, "source_id": x["question_id"], "type": stratum, "aliases": al, "abstention": abst,
                                 "answer": str(x["answer"])}) + "\n")
json.dump({"validation": n, "holdout": n}, open(f"{out}/splits.json", "w"))
json.dump({"revision": REV, "kept": len(kept), "per_split": n, "dropped": dict(dropped),
           "strata": {s: len(v) for s, v in by.items()}}, open(f"{out}/prepare_report.json", "w"), indent=1)
print(f"kept {len(kept)}, {n} per split, dropped {dict(dropped)}")
