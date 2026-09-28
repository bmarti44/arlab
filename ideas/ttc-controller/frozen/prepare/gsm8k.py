"""Frozen GSM8K problem lists (pinned snapshot, offline) shared by the sampler and PREPARE.

splits(hf) -> {"train": [...], "validation": [...], "holdout": [...]}, each a list of {pid, question, gold}:
  holdout    = the full GSM8K test set (1,319), in file order;
  validation = the first N_VAL problems of the deduplicated GSM8K-train pool ordered by sha256(f"{SPLIT_SEED}:{pid}");
  train      = the next N_TRAIN problems of that order.
A train-pool problem is dropped first if its normalized question's word 8-gram Jaccard with any test question is > 0.8.
Cuts (IDEA: train -> 400, validation -> 800) take prefixes of these lists, so a cut never changes membership rules.

  python gsm8k.py --list --out problems.jsonl [--hf /hf]   writes the sampling order: train, validation, holdout
"""
import argparse
import hashlib
import json
import os
import re

from answers import normalize

SNAPSHOT = "740312add88f781978c0658806c59bc2815b9866"
SPLIT_SEED = 20260928
N_VAL, N_TRAIN = 1000, 700
JACCARD, NGRAM = 0.8, 8
ORDER = ("train", "validation", "holdout")


def _parquet(hf: str, split: str) -> str:
    return f"{hf}/hub/datasets--openai--gsm8k/snapshots/{SNAPSHOT}/main/{split}-00000-of-00001.parquet"


def load(hf: str) -> dict:
    import pyarrow.parquet as pq
    out = {}
    for split in ("train", "test"):
        t = pq.read_table(_parquet(hf, split)).to_pydict()
        out[split] = [{"pid": f"{split}-{i:04d}", "question": q, "gold": normalize(a.split("####")[-1])}
                      for i, (q, a) in enumerate(zip(t["question"], t["answer"]))]
        assert all(p["gold"] and re.fullmatch(r"-?\d+", p["gold"]) for p in out[split]), "GSM8K gold must be integers"
    return out


def shingles(q: str) -> set:
    w = re.findall(r"[a-z0-9]+", q.lower())
    n = min(NGRAM, len(w))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def near_duplicates(pool: list, test: list) -> set:
    """pids in pool whose question has 8-gram Jaccard > JACCARD with some test question."""
    index, sh = {}, [shingles(p["question"]) for p in test]
    for j, s in enumerate(sh):
        for g in s:
            index.setdefault(g, set()).add(j)
    drop = set()
    for p in pool:
        s = shingles(p["question"])
        for j in set().union(*(index.get(g, set()) for g in s)) if s else ():
            if len(s & sh[j]) / len(s | sh[j]) > JACCARD:
                drop.add(p["pid"])
                break
    return drop


def splits(hf: str = "/hf") -> dict:
    d = load(hf)
    drop = near_duplicates(d["train"], d["test"])
    pool = sorted((p for p in d["train"] if p["pid"] not in drop),
                  key=lambda p: hashlib.sha256(f"{SPLIT_SEED}:{p['pid']}".encode()).hexdigest())
    return {"train": pool[N_VAL:N_VAL + N_TRAIN], "validation": pool[:N_VAL], "holdout": d["test"], "dropped": sorted(drop)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf", default=os.environ.get("HF_HOME", "/hf"))
    a = ap.parse_args()
    sp = splits(a.hf)
    with open(a.out + ".tmp", "w") as f:
        for split in ORDER:
            for p in sp[split]:
                f.write(json.dumps({"split": split, **p}) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print({s: len(sp[s]) for s in ORDER}, "dropped near-duplicates:", len(sp["dropped"]))
