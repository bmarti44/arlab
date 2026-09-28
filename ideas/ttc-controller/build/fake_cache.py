#!/usr/bin/env python3
"""Write a small SYNTHETIC trace cache with the real schema (cache.py) so PREPARE, RUN, EVALUATE and the pack tests run
on the CPU before the GPU sampling. Uses the real GSM8K split (pids, gold) but invented traces:
per problem a difficulty d; a trace is correct with probability 1 - d, else it gives one of three wrong answers
(the first one is an attractor); 2 % are truncated at 1024 tokens and 1 % stop without a box; lengths ~ 300 tokens;
token signals are slightly better on correct traces, so confidence-aware controllers have something to find.
The manifest says "synthetic": true; build/sample.sh deletes these files before real sampling.

  python3 build/fake_cache.py [--out frozen/prepare/traces] [--hf ~/.cache/huggingface] [--sizes 40,60,60]
(needs numpy and pyarrow; the host's system python3 has both)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frozen", "prepare"))
import cache  # noqa: E402
import gsm8k  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "..", "frozen", "prepare", "traces"))
ap.add_argument("--hf", default=os.path.expanduser("~/.cache/huggingface"))
ap.add_argument("--sizes", default="40,60,60", help="train,validation,holdout problems")
ap.add_argument("--traces", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sizes = dict(zip(gsm8k.ORDER, map(int, a.sizes.split(","))))
os.makedirs(a.out, exist_ok=True)
if os.path.exists(f"{a.out}/MANIFEST.json"):
    import json
    if not json.load(open(f"{a.out}/MANIFEST.json")).get("synthetic"):
        sys.exit(f"{a.out} holds a REAL cache; refusing to overwrite it")
for f in os.listdir(a.out):
    os.remove(os.path.join(a.out, f))
sp = gsm8k.splits(a.hf)
rng = np.random.default_rng(a.seed)


def trace(gold: str, d: float, wrong: list, mu: float, T_max: int = 1024) -> dict:
    ok = rng.random() > d
    ans = gold if ok else wrong[min(int(rng.exponential(0.8)), 2)]
    L = int(np.clip(rng.normal(mu, 70), 40, T_max))
    finish, text_end = "stop", f" So the answer is \\boxed{{{ans}}}."
    u = rng.random()
    if u < 0.02:
        L, finish, text_end = T_max, "length", " and then"
    elif u < 0.03:
        text_end, ans = " So the answer is unclear.", None
    ans = ans if finish == "stop" else None
    c0 = 2.3 + (0.12 if ok else 0.0) - 0.3 * d + rng.normal(0, 0.3)       # weak signal, trace-level noise
    conf = rng.normal(c0, 0.7, L).clip(0.05, 9)
    return {"logprob": -np.abs(rng.normal(0.28 if ok else 0.32, 0.4, L)), "conf": conf,
            "ent": np.abs(rng.normal(0.66 if ok else 0.72, 0.3, L)), "finish": finish, "answer": ans,
            "text": f"Let me think step by step ({L} tokens).{text_end}"}


man = {"version": 1, "synthetic": True, "model": "synthetic", "snapshot": None, "image": None,
       "sampling": {"generator": "build/fake_cache.py", "seed": a.seed}, "n_traces": a.traces, "sizes": sizes, "files": {}}
for split in gsm8k.ORDER:
    probs = sp[split][:sizes[split]]
    for k in range(0, len(probs), cache.SHARD):
        chunk, traces = probs[k:k + cache.SHARD], []
        for p in chunk:
            g = int(p["gold"])
            d = float(rng.beta(1.2, 2.0))
            wrong = [str(g + int(rng.integers(1, 20))), str(g * 2), str(max(g - 7, 0))]
            mu = 220 + 200 * d
            traces.append([trace(p["gold"], d, wrong, mu) for _ in range(a.traces)])
        man["files"].update(cache.write_shard(a.out, cache.shard_name(split, k // cache.SHARD), [p["pid"] for p in chunk], traces))
cache.write_manifest(a.out, man)
print("synthetic cache:", sizes, "->", os.path.abspath(a.out))
