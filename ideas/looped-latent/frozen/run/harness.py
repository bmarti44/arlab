"""Frozen RUN entry point for looped-latent. It owns the base model, LoRA, optimizer, schedule, data stream, the
wall-clock budget, greedy decoding and every measurement (frozen/run/looprt.py); the surface (/work/loop.py)
supplies only the loop module.

Outputs in --out (all scored by the frozen evaluator, which never trusts a surface number):
  preds.json      {"main": {id: [ids]}, "hard": {...}, "main_noloop": {...}}  greedy answers; main_noloop is the same
                  trained model with the loop disabled (frozen single pass through the block)
  text_nll.npy    per-token NLL of the trained model on the OASST2 guard rows (computed here from raw logits)
  base_nll.npy    the same for the untouched base model
  stats.json      steps, timings, effective depth (frozen block counter), trainable params, causality, integrity
  budget.json     {"train_seconds": wall-clock from before the surface import to the end of training}
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
from common import MODEL_DIR, PAD_ID, greedy_decode, token_nll
from looprt import LoopedModel, Trainer, add_lora, base_params, tensor_hash

BATCH = 32            # training batch (problems per optimizer step), fixed for every arm
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

text = np.load(f"{a.data}/public/text.npy")
if a.limit:
    text = text[:max(4, a.limit // 8)]
tokens = np.load(f"{a.data}/train/tokens.npy")
offsets = np.load(f"{a.data}/train/offsets.npy")
plen = np.load(f"{a.data}/train/prompt_len.npy")

base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, attn_implementation="sdpa").to(dev)
base.eval()
base_nll = token_nll(lambda x: base(input_ids=x, use_cache=False).logits, text, TEXT_BATCH, dev)
lora = add_lora(base)                                  # frozen adapters, seeded init, identity at init
base_hash_load = tensor_hash(base_params(base))

# ---------------------------------------------------------------- data stream (frozen order per seed)
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


# ---------------------------------------------------------------- training: the timer starts before the surface import
t0 = time.time()
sys.path.insert(0, "/work")
import loop as surface  # noqa: E402  (the editable surface: loop machinery only)

loop = surface.build(base.config.hidden_size, base.config.num_hidden_layers, a.seed).to(dev)
model = LoopedModel(base, loop, surface.LOOP_START, surface.LOOP_END)
trainer = Trainer(model, lora, surface.LOOP_LR)
steps, nan_at, recent = 0, None, []
while (el := time.time() - t0) < a.train_seconds:
    ids, tgt = next_batch()
    lv = trainer.step(ids, tgt, el / a.train_seconds)
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
trainer.check_trainable()


# ---------------------------------------------------------------- evaluation outputs (no learning after the deadline)
def all_state():
    return list(model.named_parameters()) + list(model.named_buffers())


hash_after_train = tensor_hash(all_state())
trainable = sum(p.numel() for p in lora) + sum(p.numel() for p in trainer.loop_params)
model.eval()
items = json.load(open(f"{a.data}/public/items.json"))
if a.limit:
    items = {k: v[:a.limit] for k, v in items.items()}
model.block_tokens = model.fwd_tokens = 0
t1 = time.time()
preds, bad = {}, 0
for kind, off in (("main", False), ("hard", False), ("main_noloop", True)):
    if kind == "main_noloop":
        depth_ratio = model.depth_ratio()                 # loop-on decoding only (main + hard)
        if dev == "cuda":
            torch.cuda.synchronize()
        gen_s = time.time() - t1
    src = items[kind.removesuffix("_noloop")]
    gen, b = greedy_decode(lambda x, off=off: model(x, loop_off=off), [it["prompt"] for it in src], EVAL_BATCH, dev)
    preds[kind] = {it["id"]: g for it, g in zip(src, gen)}
    bad += b
with torch.no_grad():
    text_nll = token_nll(model, text, TEXT_BATCH, dev)
    caus = causality_check(model, text, 151643, device=dev)
hash_after_eval = tensor_hash(all_state())
base_hash_end = tensor_hash(base_params(base))
json.dump(preds, open(f"{a.out}/preds.json", "w"))
np.save(f"{a.out}/text_nll.npy", text_nll)
np.save(f"{a.out}/base_nll.npy", base_nll)
stats = {"train_steps": steps, "examples_seen": steps * BATCH, "train_time": train_time, "nan_at": nan_at,
         "train_loss_last50": sum(recent) / max(1, len(recent)), "gen_s": gen_s, "depth_ratio": depth_ratio,
         "nonfinite_decode_calls": bad, "trainable_m": trainable / 1e6, "causal": caus["causal"],
         "causal_max_diff": caus["max_diff"], "base_unchanged": base_hash_end == base_hash_load,
         "unchanged_after_train": hash_after_eval == hash_after_train,
         "loop_window": [surface.LOOP_START, surface.LOOP_END]}
json.dump(stats, open(f"{a.out}/stats.json", "w"))
print(json.dumps(stats), flush=True)
