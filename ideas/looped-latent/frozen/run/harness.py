"""Frozen RUN entry point for looped-latent. It owns the base model, the data stream, the wall-clock training budget,
greedy decoding and every measurement; the surface (/work/loop.py) only supplies build / train_step / logits.

Outputs in --out (all scored by the frozen evaluator, which never trusts a surface number):
  preds.json      {"main": {id: [generated token ids]}, "hard": {...}}
  text_nll.npy    per-token NLL of the trained model on the OASST2 guard rows (computed here from raw logits)
  base_nll.npy    the same for the untouched base model (before build)
  stats.json      steps, examples, timings, effective depth (hooked layer calls), trainable params, causality
  budget.json     {"train_seconds": wall-clock of build + training}
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
from transformers import AutoModelForCausalLM

from arlab.lib.lm import causality_check
from common import MODEL_DIR, PAD_ID, LayerCounter, greedy_decode, token_nll

BATCH = 32            # training micro-batch (problems per train_step call)
EVAL_BATCH = 32       # decode batch
TEXT_BATCH = 8

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--train-seconds", type=float, required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=0, help="smoke tests only: first N eval items / text rows")
ap.add_argument("--data", default="/data")
a = ap.parse_args()
dev = a.device
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
torch.manual_seed(a.seed)
if dev == "cuda":
    torch.cuda.manual_seed(a.seed)
sys.path.insert(0, "/work")
import loop as surface  # noqa: E402  (the editable surface)

items = json.load(open(f"{a.data}/public/items.json"))
text = np.load(f"{a.data}/public/text.npy")
if a.limit:
    items = {k: v[:a.limit] for k, v in items.items()}
    text = text[:max(4, a.limit // 8)]
tokens = np.load(f"{a.data}/train/tokens.npy")
offsets = np.load(f"{a.data}/train/offsets.npy")
plen = np.load(f"{a.data}/train/prompt_len.npy")

base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, attn_implementation="sdpa").to(dev)
base.eval()
n_layers = base.config.num_hidden_layers
base_params = sum(p.numel() for p in base.parameters())
base_nll = token_nll(lambda x: base(input_ids=x, use_cache=False).logits, text, TEXT_BATCH, dev)

# ---------------------------------------------------------------- training (harness-owned wall clock)
rng = np.random.default_rng(a.seed)
perm, ptr = rng.permutation(len(plen)), 0


def next_batch():
    global perm, ptr
    if ptr + BATCH > len(perm):
        perm, ptr = rng.permutation(len(plen)), 0
    idx = perm[ptr:ptr + BATCH]
    ptr += BATCH
    seqs = [tokens[offsets[i]:offsets[i + 1]] for i in idx]
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), PAD_ID, dtype=np.int64)
    tgt = np.full((len(seqs), T), -100, dtype=np.int64)
    for r, (s, i) in enumerate(zip(seqs, idx)):
        ids[r, :len(s)] = s
        tgt[r, plen[i] - 1:len(s) - 1] = s[plen[i]:]      # predict the answer tokens and <|im_end|> only
    return torch.from_numpy(ids).to(dev), torch.from_numpy(tgt).to(dev)


t0 = time.time()
state = surface.build(base, {"device": dev, "seed": a.seed, "dtype": dtype})  # build time counts toward the budget
steps, nan_at, recent = 0, None, []
while (el := time.time() - t0) < a.train_seconds:
    ids, tgt = next_batch()
    lv = float(surface.train_step(state, ids, tgt, el / a.train_seconds))
    steps += 1
    recent = (recent + [lv])[-50:]
    if not math.isfinite(lv):
        nan_at = steps
        print(f"non-finite loss at step {steps}; stopping", flush=True)
        break
    if steps % 50 == 0:
        print(f"step {steps} {el:.0f}s loss {sum(recent) / len(recent):.4f}", flush=True)
if dev == "cuda":
    torch.cuda.synchronize()
train_time = time.time() - t0
print(f"trained {steps} steps ({steps * BATCH} problems) in {train_time:.1f}s", flush=True)
json.dump({"train_seconds": train_time}, open(f"{a.out}/budget.json", "w"))

# ---------------------------------------------------------------- evaluation outputs (frozen decode + measurements)
model = state["model"]
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
fwd = [0]


def logits_fn(x):
    fwd[0] += 1
    return surface.logits(state, x)


counter = LayerCounter(base.model.layers)
t1 = time.time()
preds, bad = {}, 0
for kind in ("main", "hard"):
    gen, b = greedy_decode(logits_fn, [it["prompt"] for it in items[kind]], EVAL_BATCH, dev)
    preds[kind] = {it["id"]: g for it, g in zip(items[kind], gen)}
    bad += b
if dev == "cuda":
    torch.cuda.synchronize()
gen_s = time.time() - t1
depth_ratio = counter.calls / max(1, fwd[0] * n_layers)
counter.remove()
text_nll = token_nll(logits_fn, text, TEXT_BATCH, dev)
caus = causality_check(logits_fn, text, 151643, device=dev)
json.dump(preds, open(f"{a.out}/preds.json", "w"))
np.save(f"{a.out}/text_nll.npy", text_nll)
np.save(f"{a.out}/base_nll.npy", base_nll)
stats = {"train_steps": steps, "examples_seen": steps * BATCH, "train_time": train_time, "nan_at": nan_at,
         "train_loss_last50": sum(recent) / max(1, len(recent)), "gen_s": gen_s, "depth_ratio": depth_ratio,
         "nonfinite_decode_calls": bad, "trainable_m": trainable / 1e6, "extra_params_m": (total - base_params) / 1e6,
         "causal": caus["causal"], "causal_max_diff": caus["max_diff"], "limit": a.limit}
json.dump(stats, open(f"{a.out}/stats.json", "w"))
print(json.dumps(stats), flush=True)
