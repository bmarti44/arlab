"""Frozen RUN entry point for latent-arch: owns the data mixture, batch shape, seeded order and the wall-clock budget.

The timer starts BEFORE the surface (/work/train.py + /work/model.py) is imported: import, build, torch.compile, every
train_step and save all count. Before each deadline check the GPU is synchronized, so queued work cannot hide.
After the deadline no batch is handed out; the surface's save() is timed too.

Each batch is 64 rows x 1024 tokens: TEXT_ROWS random windows of the relabelled climbmix stream + PROG_ROWS rows of
packed programs (a random program start, BOS-aligned, ~12 programs per row), in a seeded random row order.
Plain (x, y) next-token pairs; the surface cannot tell rows apart except by learning to.

Outputs in --out:
  model.pt       whatever surface.save() writes (the evaluator loads it with the surface's load())
  budget.json    {"train_seconds": t}   (runner: invalid if > budget.limit)
  stats.json     steps, tokens seen, timings, nan_at, checkpoint sha256 and the tensor hash of load(model.pt)
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

from common import PROG_ROWS, SEQ_LEN, TEXT_ROWS, VOCAB, file_sha256, tensor_hash

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)          # unused: RUN never sees eval data
ap.add_argument("--train-seconds", type=float, required=True)
ap.add_argument("--data", default="/data/train")
ap.add_argument("--work", default="/work")
ap.add_argument("--smoke", action="store_true", help="tests only: CPU, 3 text + 1 program rows of 128 tokens")
a = ap.parse_args()
dev = "cpu" if a.smoke else "cuda"
text_rows, prog_rows, seq_len = (3, 1, 128) if a.smoke else (TEXT_ROWS, PROG_ROWS, SEQ_LEN)
batch = text_rows + prog_rows


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


text = np.memmap(f"{a.data}/tokens.bin", dtype=np.uint16, mode="r")
progs = np.memmap(f"{a.data}/programs.bin", dtype=np.uint16, mode="r")
starts = np.load(f"{a.data}/program_starts.npy")
starts = starts[starts <= len(progs) - seq_len - 1]
rng = np.random.default_rng(a.seed)
torch.manual_seed(a.seed)
if dev == "cuda":
    torch.cuda.manual_seed(a.seed)
buf = torch.empty((batch, seq_len + 1), dtype=torch.long)
if dev == "cuda":
    buf = buf.pin_memory()


def next_batch():
    t0s = rng.integers(0, len(text) - seq_len - 1, text_rows)
    p0s = starts[rng.integers(0, len(starts), prog_rows)]
    rows = [text[s:s + seq_len + 1] for s in t0s] + [progs[s:s + seq_len + 1] for s in p0s]
    order = rng.permutation(batch)
    buf.copy_(torch.from_numpy(np.stack([rows[i] for i in order]).astype(np.int64)))
    xy = buf.to(dev, non_blocking=True)
    return xy[:, :-1], xy[:, 1:]


# ------------------------------------------------------------------ the clock starts here, before the surface import
t0 = time.time()
sys.path.insert(0, a.work)
import train as surface  # noqa: E402  (the editable surface)

state = surface.build({"vocab_size": VOCAB, "seq_len": seq_len, "batch": batch, "device": dev, "seed": a.seed,
                       "train_seconds": a.train_seconds})
sync()
build_s = time.time() - t0
step, nan_at, first_step_s, recent = 0, None, None, []
while True:
    sync()
    el = time.time() - t0
    if el >= a.train_seconds:
        break
    x, y = next_batch()
    loss = surface.train_step(state, (x, y), step, el / a.train_seconds)
    step += 1
    lv = float(loss)
    if first_step_s is None:
        first_step_s = time.time() - t0
    recent = (recent + [lv])[-50:]
    if not math.isfinite(lv):
        nan_at = step
        print(f"non-finite loss at step {step}; stopping", flush=True)
        break
    if step % 50 == 0:
        print(f"step {step} {el:.0f}s loss {sum(recent) / len(recent):.4f}", flush=True)
sync()
loop_end_s = time.time() - t0
surface.save(state, f"{a.out}/model.pt")
sync()
train_s = time.time() - t0
json.dump({"train_seconds": train_s}, open(f"{a.out}/budget.json", "w"))
tokens = step * batch * seq_len
print(f"trained {step} steps ({tokens} tokens) in {train_s:.1f}s (build {build_s:.1f}s, first step at "
      f"{first_step_s or 0:.1f}s, {tokens / max(loop_end_s - (first_step_s or 0), 1e-9):.0f} tok/s after it)", flush=True)

# ------------------------------------------------------------------ after the deadline: integrity fingerprints only
sha = file_sha256(f"{a.out}/model.pt")
del state
model = surface.load(f"{a.out}/model.pt", dev)
stats = {"train_steps": step, "tokens_seen": tokens, "program_tokens_seen": step * prog_rows * seq_len,
         "train_seconds": train_s, "build_s": build_s, "first_step_s": first_step_s, "loop_end_s": loop_end_s,
         "save_s": train_s - loop_end_s, "overrun_s": max(0.0, loop_end_s - a.train_seconds), "nan_at": nan_at,
         "train_loss_last50": sum(recent) / max(1, len(recent)), "model_sha256": sha, "tensor_hash": tensor_hash(model),
         "batch": [text_rows, prog_rows, seq_len], "seed": a.seed}
json.dump(stats, open(f"{a.out}/stats.json", "w"))
print(json.dumps(stats), flush=True)
