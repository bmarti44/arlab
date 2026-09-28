"""PREPARE for ttc-controller (CPU, offline). The traces were sampled before the build (build/sample.sh) into
/prepare/traces/ (part of data_hash); this step verifies them and writes the replay data. Layout under --out:

  train/traces.npz                         the cache arrays (see cache.py) + correct (n, T) bool; answer VALUES stay
                                           here, the harness hands fit() opaque labels only
  {validation,holdout}/public/traces.npz   pids, lengths, finish, answers, offsets, logprob, conf, ent (no gold)
  {validation,holdout}/private/gold.json   {pid: gold}; private/texts.json.gz {pid: [T texts]} (audit only)
  splits.json {"validation": n, "holdout": n};  info.json (stats; suggested budget B = 4 x mean train length)

Checks: every shard's sha256 against MANIFEST.json; each split's pids are exactly the frozen GSM8K split (a prefix of
it for cuts; the holdout must be complete unless the cache is synthetic); every stored answer equals the frozen
extraction of its text; signals finite.
"""
import argparse
import gzip
import json
import os
from collections import Counter

import numpy as np

import cache
import gsm8k
from answers import extract

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--traces", default="/prepare/traces")
ap.add_argument("--hf", default=os.environ.get("HF_HOME", "/hf"))
a = ap.parse_args()

man = json.load(open(f"{a.traces}/MANIFEST.json"))
T = int(man["n_traces"])
sp = gsm8k.splits(a.hf)
info = {"manifest": {k: man.get(k) for k in ("synthetic", "model", "snapshot", "image", "sampling", "sizes", "n_traces")},
        "splits": {}}
if man.get("synthetic"):
    print("WARNING: SYNTHETIC trace cache (build/fake_cache.py): for CPU checks only, never for a campaign", flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


def maj(ans_row):
    c = Counter(x for x in ans_row if x)
    return c.most_common(1)[0][0] if c else None


for split, n in man["sizes"].items():
    if n == 0:
        continue
    arr, texts = cache.read_split(a.traces, split, man)
    want = sp[split][:n]
    if split == "holdout" and n != len(sp["holdout"]) and not man.get("synthetic"):
        raise SystemExit(f"holdout must be the full GSM8K test set ({len(sp['holdout'])}), cache has {n}")
    if [str(p) for p in arr["pids"]] != [p["pid"] for p in want]:
        raise SystemExit(f"{split}: cached problems differ from the frozen GSM8K split (prefix of {len(sp[split])})")
    assert arr["lengths"].shape == (n, T) and len(texts) == n and all(len(t) == T for t in texts)
    fin = [cache.FINISH[f] for f in arr["finish"].reshape(-1)]
    redo = np.array([extract(t, f) or "" for t, f in zip((t for row in texts for t in row), fin)]).reshape(n, T)
    if not np.array_equal(redo.astype(str), arr["answers"].astype(str)):
        bad = np.argwhere(redo.astype(str) != arr["answers"].astype(str))[0]
        raise SystemExit(f"{split}: stored answer differs from the frozen extraction at {bad.tolist()}")
    for s in cache.SIGNALS:
        if not np.isfinite(arr[s].astype(np.float32)).all():
            raise SystemExit(f"{split}: non-finite {s}")
    gold = np.array([p["gold"] for p in want])
    correct = arr["answers"] == gold[:, None]
    arrays = {k: arr[k] for k in ("pids", "lengths", "finish", "answers", "offsets", *cache.SIGNALS)}
    if split == "train":
        os.makedirs(f"{a.out}/train", exist_ok=True)
        np.savez(f"{a.out}/train/traces.npz", **arrays, correct=correct)
    else:
        os.makedirs(f"{a.out}/{split}/public", exist_ok=True)
        np.savez(f"{a.out}/{split}/public/traces.npz", **arrays)
        write_json(f"{a.out}/{split}/private/gold.json", {p["pid"]: p["gold"] for p in want})
        write_json(f"{a.out}/{split}/private/provenance.json", {"synthetic": bool(man.get("synthetic")), "model": man.get("model")})
        with gzip.open(f"{a.out}/{split}/private/texts.json.gz", "wt") as f:
            json.dump({p["pid"]: t for p, t in zip(want, texts)}, f)
    L = arr["lengths"]
    info["splits"][split] = {
        "n": n, "mean_len": float(L.mean()), "trunc": float((arr["finish"] == 1).mean()),
        "no_answer": float((arr["answers"] == "").mean()), "pass1": float(correct.mean()),
        "maj_all": float(np.mean([maj(r) == g for r, g in zip(arr["answers"], gold)])),
        "maj4": float(np.mean([maj(r[:4]) == g for r, g in zip(arr["answers"], gold)]))}
    print(split, json.dumps(info["splits"][split]), flush=True)

if "train" in info["splits"]:
    info["suggested_budget_tokens"] = int(round(4 * info["splits"]["train"]["mean_len"] / 50) * 50)
write_json(f"{a.out}/splits.json", {s: info["splits"].get(s, {}).get("n", 0) for s in ("validation", "holdout")})
write_json(f"{a.out}/info.json", info)
print("suggested --budget-tokens (4 x mean train length):", info.get("suggested_budget_tokens"))
