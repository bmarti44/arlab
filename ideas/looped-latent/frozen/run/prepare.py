"""PREPARE for looped-latent (CPU, offline, deterministic): program-tracing problems from progen + an OASST2 text sample.

Layout written under --out:
  train/tokens.npy, offsets.npy, prompt_len.npy   N_TRAIN problems (steps 2-8), prompt + answer tokens
  {validation,holdout}/public/items.json          {"main": [{id, prompt}], "hard": [...]}: prompt token ids only
  {validation,holdout}/public/text.npy            N_TEXT OASST2 rows of TEXT_LEN+1 tokens (retention guard)
  {validation,holdout}/private/answers.json       {"main": {id: {answer, steps, code}}, "hard": {...}}
  splits.json, info.json
Every split has its own generator seed; eval programs are hash-deduplicated against each other and train excludes
all of them. OASST2 (Apache-2.0) is read from the pinned HF-cache snapshot; nothing is downloaded.
"""
import argparse
import gzip
import json
import os
import random
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from common import MODEL_DIR, OASST_FILE, PROMPT, TARGET
from progen import DIFFICULTY, HARD_STEPS, ID_STEPS, TRAIN_STEPS, code_hash, generate, question

N_TRAIN = 100_000     # far more than one 11-min run sees (~40k); the harness samples a seeded permutation
N_EVAL = 2_000        # per split (validation, holdout); power arithmetic in pack.yaml
N_HARD = 1_000        # per split, steps 9-12 (guard only)
N_TEXT, TEXT_LEN = 256, 256
SEEDS = {"validation": 2001, "holdout": 3001, "hard_validation": 4001, "hard_holdout": 5001, "train": 1001, "text": 7}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--small", action="store_true", help="tiny sizes for a local smoke test only")
a = ap.parse_args()
if a.small:
    N_TRAIN, N_EVAL, N_HARD, N_TEXT = 2_000, 100, 50, 16
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
seen: set[str] = set()


def fresh(seed: int, n: int, steps: tuple[int, int]) -> list[dict]:
    """n problems from `seed` whose program hash is new (dedup within and across splits)."""
    out, k = [], 0
    while len(out) < n:
        for p in generate(seed * 1000 + k, 2 * n, steps):
            h = code_hash(p["code"])
            if h not in seen and len(out) < n:
                seen.add(h)
                out.append({**p, "hash": h})
        k += 1
    return out


def encode(texts: list[str]) -> list[list[int]]:
    return tok(texts, add_special_tokens=False)["input_ids"]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


sets = {}
for split in ("validation", "holdout"):
    sets[split] = fresh(SEEDS[split], N_EVAL, ID_STEPS)
for split in ("validation", "holdout"):
    sets["hard_" + split] = fresh(SEEDS["hard_" + split], N_HARD, HARD_STEPS)
train = fresh(SEEDS["train"], N_TRAIN, TRAIN_STEPS)

# ---- eval splits: prompts public, answers private
for split in ("validation", "holdout"):
    pub, priv = {}, {}
    for kind, key in (("main", split), ("hard", "hard_" + split)):
        probs = sets[key]
        ids = [f"{split[0]}{kind[0]}{i:05d}" for i in range(len(probs))]
        prompts = encode([PROMPT.format(q=question(p["code"])) for p in probs])
        pub[kind] = [{"id": i, "prompt": pr} for i, pr in zip(ids, prompts)]
        priv[kind] = {i: {"answer": p["answer"], "steps": p["steps"], "code": p["code"]} for i, p in zip(ids, probs)}
    write_json(f"{a.out}/{split}/public/items.json", pub)
    write_json(f"{a.out}/{split}/private/answers.json", priv)

# ---- train: prompt + answer tokens, loss mask from prompt_len
prompts = encode([PROMPT.format(q=question(p["code"])) for p in train])
answers = encode([TARGET.format(a=p["answer"]) for p in train])
seqs = [p + t for p, t in zip(prompts, answers)]
os.makedirs(f"{a.out}/train", exist_ok=True)
np.save(f"{a.out}/train/tokens.npy", np.fromiter((t for s in seqs for t in s), dtype=np.int32))
np.save(f"{a.out}/train/offsets.npy", np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64))
np.save(f"{a.out}/train/prompt_len.npy", np.array([len(p) for p in prompts], dtype=np.int32))

# ---- retention-guard text: English OASST2 messages, first TEXT_LEN+1 tokens of long ones, disjoint halves
msgs = []


def walk(node):
    if node.get("lang") == "en" and node.get("text"):
        msgs.append((node["message_id"], node["text"]))
    for r in node.get("replies") or []:
        walk(r)


with gzip.open(OASST_FILE, "rt") as f:
    for line in f:
        walk(json.loads(line)["prompt"])
msgs.sort()
random.Random(SEEDS["text"]).shuffle(msgs)
rows = []
for i in range(0, len(msgs), 512):
    for ids in encode([t for _, t in msgs[i:i + 512]]):
        if len(ids) >= TEXT_LEN + 1 and len(rows) < 2 * N_TEXT:
            rows.append(ids[:TEXT_LEN + 1])
    if len(rows) >= 2 * N_TEXT:
        break
rows = np.array(rows, dtype=np.int64)
assert rows.shape == (2 * N_TEXT, TEXT_LEN + 1), rows.shape
np.save(f"{a.out}/validation/public/text.npy", rows[:N_TEXT])
np.save(f"{a.out}/holdout/public/text.npy", rows[N_TEXT:])

write_json(f"{a.out}/splits.json", {"validation": N_EVAL, "holdout": N_EVAL})
lens = np.diff(np.load(f"{a.out}/train/offsets.npy"))
info = {"difficulty": DIFFICULTY, "train_steps": TRAIN_STEPS, "id_steps": ID_STEPS, "hard_steps": HARD_STEPS,
        "n": {k: len(v) for k, v in sets.items()} | {"train": len(train), "text_rows_per_split": N_TEXT},
        "seeds": SEEDS, "model_dir": MODEL_DIR, "oasst_file": OASST_FILE,
        "train_tokens": int(lens.sum()), "train_seq_len": {"mean": float(lens.mean()), "max": int(lens.max())},
        "top_answers_validation": Counter(p["answer"] for p in sets["validation"]).most_common(5)}
write_json(f"{a.out}/info.json", info)
print(json.dumps(info, indent=1))
