"""Trace-cache shard format (written by sample.py or build/fake_cache.py, read by prepare.py).

A shard <split>-<k:03d>.npz holds up to SHARD problems x T traces:
  pids (n,) str; lengths (n, T) int32; finish (n, T) uint8 (0 = stop, 1 = length); answers (n, T) str ('' = none);
  offsets (n*T + 1,) int64 into the flat per-token fp16 arrays logprob (sampled token), conf (DeepConf token
  confidence = -mean of the top-5 logprobs) and ent (entropy of the renormalized top-5).
Its texts go to <split>-<k:03d>.texts.json.gz ([[T strings] per problem]).
MANIFEST.json: sampling metadata, "sizes" {split: n problems}, "n_traces", "synthetic", and "files" {name: sha256}.
"""
import gzip
import hashlib
import json
import os

import numpy as np

SHARD = 100
FINISH = ("stop", "length")
SIGNALS = ("logprob", "conf", "ent")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shard_name(split: str, k: int) -> str:
    return f"{split}-{k:03d}"


def write_shard(out: str, name: str, pids: list, traces: list) -> dict:
    """traces[i][j] = {"logprob", "conf", "ent": float arrays, "finish": "stop"|"length", "answer": str|None, "text"}.
    Returns {file name: sha256} of the two files written (atomically)."""
    T = len(traces[0])
    assert all(len(tr) == T for tr in traces)
    flat = [t for tr in traces for t in tr]
    lens = np.array([len(t["logprob"]) for t in flat], dtype=np.int64)
    arrays = {"pids": np.array(pids), "lengths": lens.reshape(len(pids), T).astype(np.int32),
              "finish": np.array([FINISH.index(t["finish"]) for t in flat], dtype=np.uint8).reshape(len(pids), T),
              "answers": np.array([t["answer"] or "" for t in flat]).reshape(len(pids), T),
              "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)}
    for s in SIGNALS:
        arrays[s] = np.concatenate([np.asarray(t[s], dtype=np.float16) for t in flat])
    files = {}
    for fn, write in ((f"{name}.npz", lambda f: np.savez(f, **arrays)),
                      (f"{name}.texts.json.gz", lambda f: f.write(gzip.compress(json.dumps(
                          [[t["text"] for t in tr] for tr in traces]).encode(), mtime=0)))):
        tmp = os.path.join(out, fn + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, os.path.join(out, fn))
        files[fn] = sha256(os.path.join(out, fn))
    return files


def write_manifest(out: str, man: dict):
    tmp = os.path.join(out, "MANIFEST.json.tmp")
    json.dump(man, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(out, "MANIFEST.json"))


def read_split(traces_dir: str, split: str, man: dict) -> tuple[dict, list]:
    """Concatenate the shards of one split (sha256-verified against the manifest). Returns (arrays, texts)."""
    names = sorted(f[:-4] for f in man["files"] if f.startswith(split + "-") and f.endswith(".npz"))
    parts, texts = [], []
    for n in names:
        for fn in (f"{n}.npz", f"{n}.texts.json.gz"):
            if sha256(os.path.join(traces_dir, fn)) != man["files"][fn]:
                raise ValueError(f"{fn}: sha256 differs from MANIFEST.json")
        z = np.load(os.path.join(traces_dir, f"{n}.npz"))
        parts.append({k: z[k] for k in z.files})
        texts += json.loads(gzip.decompress(open(os.path.join(traces_dir, f"{n}.texts.json.gz"), "rb").read()))
    if not parts:
        return {}, []
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("pids", "lengths", "finish", "answers", *SIGNALS)}
    lens = out["lengths"].reshape(-1).astype(np.int64)
    out["offsets"] = np.concatenate([[0], np.cumsum(lens)])
    for p in parts:
        assert p["offsets"][-1] == len(p["logprob"]) and np.array_equal(np.diff(p["offsets"]), p["lengths"].reshape(-1))
    return out, texts
