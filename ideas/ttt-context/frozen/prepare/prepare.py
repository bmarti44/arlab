"""PREPARE for ttt-context (CPU, offline, deterministic; mounted at /prepare in PREPARE only, so RUN can neither
regenerate the documents nor learn the split seeds): N_ITEMS long documents + one question each, per split.

Layout written under --out:
  {validation,holdout}/public/docs.npy, offsets.npy   document prompt tokens (chat head + log), concatenated, int32
  {validation,holdout}/public/items.json              [{"id", "q"}]: q = question prompt tokens (question + assistant
                                                      header); item i uses document i. No answers.
  {validation,holdout}/private/answers.json           {id: {answer, aliases, kind, seed, target}}
  splits.json, info.json
Splits come from disjoint generator seed ranges (validation 1,000,000+i, holdout 2,000,000+i); kinds cycle
state/hop2/count/kv so every prefix of 4k items is balanced (the pilot's --limit relies on it).
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from common import MODEL_DIR, QUESTION
from docgen import CONTEXT_TOKENS, DIFFICULTY, KINDS, N_ITEMS, build

SPLIT_SEEDS = {"validation": 1_000_000, "holdout": 2_000_000}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--small", action="store_true", help="tiny sizes for a local smoke test only")
a = ap.parse_args()
ctx_tokens, n_items = (CONTEXT_TOKENS, N_ITEMS) if not a.small else (1024, 16)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def count(text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


info = {"context_tokens": ctx_tokens, "n_items": n_items, "difficulty": DIFFICULTY, "split_seeds": SPLIT_SEEDS,
        "model_dir": MODEL_DIR, "splits": {}}
for split, base in SPLIT_SEEDS.items():
    docs, pub, priv = [], [], {}
    for i in range(n_items):
        it = build(base + i, KINDS[i % len(KINDS)], ctx_tokens, count)
        ids = tok(it["text"], add_special_tokens=False)["input_ids"]
        assert len(ids) == it["n_tokens"] <= ctx_tokens, (split, i, len(ids))
        iid = f"{split[0]}{i:04d}"
        docs.append(np.asarray(ids, dtype=np.int32))
        pub.append({"id": iid, "q": tok(QUESTION.format(q=it["question"]), add_special_tokens=False)["input_ids"]})
        priv[iid] = {k: it[k] for k in ("answer", "aliases", "kind", "seed", "target")}
    os.makedirs(f"{a.out}/{split}/public", exist_ok=True)
    np.save(f"{a.out}/{split}/public/docs.npy", np.concatenate(docs))
    np.save(f"{a.out}/{split}/public/offsets.npy", np.cumsum([0] + [len(d) for d in docs]).astype(np.int64))
    write_json(f"{a.out}/{split}/public/items.json", pub)
    write_json(f"{a.out}/{split}/private/answers.json", priv)
    lens = [len(d) for d in docs]
    info["splits"][split] = {"n": n_items, "doc_tokens": {"min": min(lens), "max": max(lens), "mean": float(np.mean(lens))},
                             "kinds": Counter(v["kind"] for v in priv.values()),
                             "top_answers": Counter(v["answer"] for v in priv.values()).most_common(5)}
    print(split, json.dumps(info["splits"][split]), flush=True)

write_json(f"{a.out}/splits.json", {"validation": n_items, "holdout": n_items})
write_json(f"{a.out}/info.json", info)
