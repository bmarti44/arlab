"""GPU trace sampler (vLLM offline), run by build/sample.sh inside nvcr.io/nvidia/vllm:26.04-py3 with the arlab GPU
lock held. Never part of arlab's PREPARE (no GPU there); PREPARE only verifies what this wrote.

For every problem of problems.jsonl (gsm8k.py --list; order train, validation, holdout) it samples --n traces in one
request (shared prefix) and writes resumable 100-problem shards (cache.py) plus MANIFEST.json (sha256 of every
file, model snapshot, vLLM image, sampling parameters). Existing shards listed in the manifest are skipped, so a
killed job resumes. vLLM batching is not bitwise deterministic: the cache, not this script, is the frozen artifact.

  --pilot 100                            sample the first 100 train problems only and print the pilot gate table
  --splits train:400,validation:800,holdout    cuts (prefixes) if the pilot projects > 6 GPU h; never cut the holdout
  --model Qwen/Qwen3.5-2B@15852e8c --n 16 --splits train:300,holdout    the transfer cache (with another --out)
"""
import argparse
import glob
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np

import cache
from answers import extract

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
GATES = {"trunc_max": 0.03, "nobox_max": 0.02, "pass1": (0.50, 0.88), "headroom_min": 0.05, "hours_max": 6.0}
TOTAL_PROBLEMS = 3019

ap = argparse.ArgumentParser()
ap.add_argument("--problems", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-1.7B@70d244cc")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--splits", default="train,validation,holdout", help="split[:n] list, in sampling order")
ap.add_argument("--pilot", type=int, default=0, help="sample only the first N train problems and print the gate table")
ap.add_argument("--gpu-mem", type=float, default=0.25)
ap.add_argument("--max-num-seqs", type=int, default=128)
a = ap.parse_args()

name, rev = a.model.split("@")
snap = glob.glob(f"{os.environ.get('HF_HOME', '/hf')}/hub/models--{name.replace('/', '--')}/snapshots/{rev}*")
assert len(snap) == 1, f"model snapshot {a.model} not found (or ambiguous) in the HF cache"
SAMPLING = {"n": a.n, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": a.max_tokens, "logprobs": 5,
            "logprobs_mode": "raw_logprobs", "enable_thinking": False, "suffix": SUFFIX}

probs = [json.loads(line) for line in open(a.problems)]
want = {}
for s in a.splits.split(","):
    sp, _, k = s.partition(":")
    rows = [p for p in probs if p["split"] == sp]
    want[sp] = rows[:int(k)] if k else rows
if a.pilot:
    want = {"train": want.get("train", [p for p in probs if p["split"] == "train"])[:a.pilot]}

man_path = f"{a.out}/MANIFEST.json"
man = json.load(open(man_path)) if os.path.exists(man_path) else None
meta = {"version": 1, "synthetic": False, "model": name, "snapshot": os.path.basename(snap[0]),
        "image": os.environ.get("VLLM_IMAGE"), "sampling": SAMPLING, "n_traces": a.n,
        "problems_sha256": hashlib.sha256(open(a.problems, "rb").read()).hexdigest()}
if man is None:
    man = {**meta, "sizes": {}, "files": {}, "throughput": []}
elif man.get("synthetic") or any(man.get(k) != v for k, v in meta.items() if k != "image"):
    raise SystemExit("MANIFEST.json in --out was written with other settings (or is synthetic): use another --out")

from vllm import LLM, SamplingParams  # noqa: E402  (after the cheap checks)

llm = LLM(model=snap[0], dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
          enable_prefix_caching=True, max_model_len=2048, max_logprobs=5, logprobs_mode="raw_logprobs")
tok = llm.get_tokenizer()


def convert(o) -> dict:
    top = np.zeros((len(o.token_ids), 5))
    lp = np.zeros(len(o.token_ids))
    for i, (tid, d) in enumerate(zip(o.token_ids, o.logprobs)):
        lp[i] = d[tid].logprob
        x = [v for _, v in sorted((v.rank, v.logprob) for v in d.values() if v.rank is not None and v.rank <= 5)][:5]
        top[i] = x + [x[-1] - 20.0] * (5 - len(x))          # vLLM returns exactly 5; pad defensively
    top, lp = np.maximum(top, -100.0), np.maximum(lp, -100.0)   # keep fp16-safe and finite
    p = np.exp(top - top.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    finish = "stop" if o.finish_reason == "stop" else "length"
    return {"logprob": lp, "conf": -top.mean(1), "ent": -(p * np.log(np.maximum(p, 1e-30))).sum(1), "finish": finish,
            "answer": extract(o.text, finish), "text": o.text}


new = []
for split, rows in want.items():
    for k in range(0, len(rows), cache.SHARD):
        sname, chunk = cache.shard_name(split, k // cache.SHARD), rows[k:k + cache.SHARD]
        f = f"{sname}.npz"
        if f in man["files"] and os.path.exists(f"{a.out}/{f}") and cache.sha256(f"{a.out}/{f}") == man["files"][f]:
            continue
        prompts = [tok.apply_chat_template([{"role": "user", "content": p["question"] + SUFFIX}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for p in chunk]
        params = [SamplingParams(n=a.n, temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens, logprobs=5,
                                 seed=int(hashlib.sha256(p["pid"].encode()).hexdigest()[:8], 16)) for p in chunk]
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t0
        traces = [[convert(o) for o in out.outputs] for out in outs]
        man["files"].update(cache.write_shard(a.out, sname, [p["pid"] for p in chunk], traces))
        man["sizes"][split] = max(man["sizes"].get(split, 0), k + len(chunk))
        gen = sum(len(t["logprob"]) for tr in traces for t in tr)
        man["throughput"].append({"shard": sname, "tokens": gen, "seconds": round(dt, 1)})
        cache.write_manifest(a.out, man)
        new.append((chunk, traces, gen, dt))
        print(f"{time.strftime('%H:%M:%S')} {sname}: {gen} tokens in {dt:.0f} s ({gen / dt:.0f} tok/s)", flush=True)

if a.pilot and new:
    gold = [p["gold"] for chunk, *_ in new for p in chunk]
    ans = [[t["answer"] for t in tr] for _, traces, *_ in new for tr in traces]
    fin = [t["finish"] for _, traces, *_ in new for tr in traces for t in tr]
    tokens, secs = sum(x[2] for x in new), sum(x[3] for x in new)
    mean_len = tokens / len(fin)
    trunc = fin.count("length") / len(fin)
    nobox = sum(f == "stop" and x is None for f, x in zip(fin, (x for r in ans for x in r))) / len(fin)
    pass1 = float(np.mean([x == g for r, g in zip(ans, gold) for x in r]))
    maj = float(np.mean([bool(c) and c[0][0] == g for c, g in ((Counter(x for x in r if x).most_common(1), g) for r, g in zip(ans, gold))]))
    hours = TOTAL_PROBLEMS * a.n * mean_len / (tokens / secs) / 3600
    rows = [("truncation", trunc, trunc <= GATES["trunc_max"]), ("no box", nobox, nobox <= GATES["nobox_max"]),
            ("pass@1", pass1, GATES["pass1"][0] <= pass1 <= GATES["pass1"][1]),
            (f"maj@{a.n} - pass@1", maj - pass1, maj - pass1 >= GATES["headroom_min"]),
            ("projected GPU h", hours, hours <= GATES["hours_max"])]
    print(f"\nPILOT GATE ({len(gold)} problems x {a.n}; mean length {mean_len:.0f}, {tokens / secs:.0f} tok/s, "
          f"suggested B = {int(round(4 * mean_len / 50) * 50)})")
    for k, v, ok in rows:
        print(f"  {k:22s} {v:8.3f}  {'PASS' if ok else 'FAIL'}")
    print("PILOT", "PASS" if all(ok for *_, ok in rows) else "FAIL")
    if pass1 > GATES["pass1"][1]:
        print("pass@1 > 0.88: switch to Qwen3-0.6B (--model Qwen/Qwen3-0.6B@c1899de2; delete the 1.7B pilot files in --out first) and record it in DECISIONS.md")
