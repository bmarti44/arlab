"""Frozen RUN entry point for plastic-agent.

For every world of the split (in a fixed order) the harness hands the surface ONLY that world's frozen exploration
transcript and tool names, plus `gen` (budgeted base-model generation, engine.Gen) and `train` (the frozen LoRA
trainer, trainer.Trainer.train). adapt() returns an adapter from train() or None. The harness then saves the adapter,
detaches everything and checks that the base weights are bit-for-bit unchanged before the next world (per-world
adapters, reset between worlds). No task, goal, answer or simulator is reachable from RUN: they live in private/.

Outputs in --out (scored only by the frozen evaluator):
  adapters/<world_id>/adapter.{safetensors,json}   one per world whose adapt() returned an adapter
  stats.json     arm, per-world timings / tokens / trainer stats (reported, never used for scoring)
  budget.json    {"adapt_s_max": the longest per-world adapt() wall time}
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import copy
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lora
from common import MODEL_DIR
from engine import Budget, BudgetExceeded, Engine, Gen
from trainer import Trainer

ARMS = ("adapter", "icl", "placebo")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--adapt-seconds", type=float, required=True, help="per-world wall-clock budget of adapt()")
ap.add_argument("--gen-tokens", type=int, required=True, help="per-world gen budget (tokens processed)")
ap.add_argument("--train-tokens", type=int, required=True, help="per-world train budget (tokens processed)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=MODEL_DIR, help="tests only: a tiny random model")
ap.add_argument("--data", default="/data")
ap.add_argument("--work", default="/work")
ap.add_argument("--limit-worlds", type=int, default=0, help="tests only")
a = ap.parse_args()
dev = a.device
dtype = torch.bfloat16 if dev == "cuda" else torch.float32

sys.path.insert(0, a.work)
import adapt as surface  # noqa: E402  (the editable surface)

arm = getattr(surface, "ARM", "adapter")
if arm not in ARMS:
    raise SystemExit(f"surface ARM must be one of {ARMS}")
worlds = json.load(open(f"{a.data}/public/worlds.json"))
if a.limit_worlds:
    worlds = worlds[:a.limit_worlds]
replay = np.load(f"{a.data}/train/replay.npy")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(dev)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
engine = Engine(model, tok, dev)
fp0 = lora.fingerprint(model)
os.makedirs(f"{a.out}/adapters", exist_ok=True)

per_world, t_all = [], time.time()
for wi, w in enumerate(worlds):
    if lora.n_wrapped(model) or lora.fingerprint(model) != fp0:
        raise RuntimeError(f"base model changed before world {w['id']}: adapters must not leak between worlds")
    s = a.seed * 1000 + wi
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    budget = Budget(a.adapt_seconds, a.gen_tokens, a.train_tokens)
    gen = Gen(engine, w["tools"], copy.deepcopy(w["transcript"]), budget, s)
    trainer = Trainer(model, engine, w["tools"], replay, budget, s, dev)
    t0 = time.time()
    hit = None
    try:
        res = surface.adapt(copy.deepcopy(w["transcript"]), list(w["tools"]), gen, trainer.train)
    except BudgetExceeded as e:     # out of time or tokens: keep the last adapter trained in this world, if any
        res, hit = (trainer.produced[-1] if trainer.produced else None), str(e)
    adapt_s = time.time() - t0
    lora.detach(model)
    model.eval()
    if res is not None and not any(res is x for x in trainer.produced):
        raise TypeError("adapt() must return an adapter produced by this world's train() call, or None")
    if res is not None:
        lora.save(f"{a.out}/adapters/{w['id']}", res.cfg, res.tensors)
    del gen, trainer
    if dev == "cuda":
        torch.cuda.empty_cache()
    rec = {"id": w["id"], "adapter": res is not None, "adapt_s": adapt_s, "gen_tokens": budget.gen_used,
           "train_tokens": budget.train_used, "budget_hit": hit, "adapter_params_m": lora.n_params(res.tensors) / 1e6 if res else 0.0,
           "train": res.stats if res else None}
    per_world.append(rec)
    print(json.dumps(rec), flush=True)
if lora.n_wrapped(model) or lora.fingerprint(model) != fp0:
    raise RuntimeError("base model changed")

json.dump({"adapt_s_max": max((r["adapt_s"] for r in per_world), default=0.0)}, open(f"{a.out}/budget.json", "w"))
stats = {"arm": arm, "split": a.split, "seed": a.seed, "worlds": per_world, "run_s": time.time() - t_all,
         "nan": any(r["train"] and r["train"]["nan"] for r in per_world)}
json.dump(stats, open(f"{a.out}/stats.json", "w"))
print(json.dumps({k: v for k, v in stats.items() if k != "worlds"}), flush=True)
