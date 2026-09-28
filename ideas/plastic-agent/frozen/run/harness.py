"""Frozen RUN entry point for plastic-agent.

For every world of the split (in a fixed order) the harness loads ONLY that world's public file and hands the surface
that world's frozen exploration transcript and tool names, plus `gen` (budgeted base-model generation, engine.Gen) and
`train` (the frozen LoRA trainer, trainer.Trainer.train). adapt() returns an adapter from train() or None. The harness
saves the trainer's private copy of that adapter, detaches everything and checks that the base weights are
byte-for-byte unchanged (sha256 over every parameter and buffer) at load, between worlds and at the end (per-world
adapters, reset between worlds). No task, goal, answer or simulator is reachable from RUN: they live in private/.

The arm is chosen by frozen code (common.arm_of: the sha256 of /work/adapt.py against the frozen references), never by
the surface. Only the placebo reference changes what adapt() receives: for world i it gets the transcript and tool
names of world i+1 (mod n), so the evaluator's own-world score of that adapter is the placebo control. A candidate
always adapts to, and is scored on, its own world.

The per-world clock starts before the surface module is imported: the import time is charged to world 0.
Any non-finite loss/gradient in any train() call makes the run invalid (stats.nan), even if a finite adapter is returned.

Outputs in --out (scored only by the frozen evaluator):
  adapters/<world_id>/adapter.{safetensors,json}   one per world whose adapt() returned an adapter
  stats.json     arm, per-world timings / tokens / trainer stats (reported, never used for scoring)
  budget.json    {"adapt_s_max": the longest per-world adapt() wall time}
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import copy
import importlib
import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lora
from common import MODEL_DIR, arm_of
from engine import Budget, BudgetExceeded, Engine, Gen
from trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--adapt-seconds", type=float, required=True, help="per-world wall-clock budget of adapt()")
ap.add_argument("--gen-tokens", type=int, required=True, help="per-world gen budget (positions processed)")
ap.add_argument("--train-tokens", type=int, required=True, help="per-world train budget (positions processed)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=MODEL_DIR, help="tests only: a tiny random model")
ap.add_argument("--data", default="/data")
ap.add_argument("--work", default="/work")
ap.add_argument("--limit-worlds", type=int, default=0, help="tests only")
a = ap.parse_args()
dev = a.device
dtype = torch.bfloat16 if dev == "cuda" else torch.float32

arm = arm_of(a.work, HERE)
order = json.load(open(f"{a.data}/public/order.json"))
if a.limit_worlds:
    order = order[:a.limit_worlds]
if arm == "placebo" and len(order) < 2:
    raise SystemExit("the placebo arm needs at least two worlds")
replay = np.load(f"{a.data}/train/replay.npy")


def load_world(wid: str) -> dict:
    return json.load(open(f"{a.data}/public/worlds/{wid}.json"))


tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(dev)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
engine = Engine(model, tok, dev)
fp0 = lora.fingerprint(model)
os.makedirs(f"{a.out}/adapters", exist_ok=True)


def check_base(where: str):
    if lora.n_wrapped(model) or lora.fingerprint(model) != fp0:
        raise RuntimeError(f"base model changed {where}: adapters must not leak between worlds")


per_world, t_all, nan_any = [], time.time(), False
t_import = time.time()                  # world 0's clock starts before the surface is imported
sys.path.insert(0, a.work)
surface = importlib.import_module("adapt")      # the editable surface
import_s = time.time() - t_import
for wi, wid in enumerate(order):
    check_base(f"before world {wid}")
    donor = order[(wi + 1) % len(order)] if arm == "placebo" else wid
    w = load_world(donor)
    s = a.seed * 1000 + wi
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    t0 = t_import if wi == 0 else time.time()
    budget = Budget(a.adapt_seconds, a.gen_tokens, a.train_tokens, start=t0)
    gen = Gen(engine, w["tools"], copy.deepcopy(w["transcript"]), budget, s)
    trainer = Trainer(model, engine, w["tools"], replay, budget, s, dev)
    hit = None
    try:
        res = surface.adapt(copy.deepcopy(w["transcript"]), list(w["tools"]), gen, trainer.train)
    except BudgetExceeded as e:     # out of time or tokens: keep the last adapter trained in this world, if any
        res, hit = (trainer.produced[-1] if trainer.produced else None), str(e)
    adapt_s = time.time() - t0
    lora.detach(model)
    model.eval()
    if res is not None:
        cfg, tens = trainer.saved(res)      # the trainer's private copy; raises if res is not from this world's train()
        lora.save(f"{a.out}/adapters/{wid}", cfg, tens)
    nan_any |= trainer.nan_seen
    rec = {"id": wid, "donor": donor, "adapter": res is not None, "adapt_s": adapt_s,
           "import_s": import_s if wi == 0 else 0.0, "gen_tokens": budget.gen_used, "train_tokens": budget.train_used,
           "budget_hit": hit, "adapter_params_m": lora.n_params(tens) / 1e6 if res is not None else 0.0,
           "train_calls": trainer.calls, "nan_seen": trainer.nan_seen,
           "train": dict(trainer.saved_stats(res)) if res is not None else None}
    del gen, trainer, w, res
    if dev == "cuda":
        torch.cuda.empty_cache()
    check_base(f"after world {wid}")
    per_world.append(rec)
    print(json.dumps(rec), flush=True)
check_base("at the end")

json.dump({"adapt_s_max": max((r["adapt_s"] for r in per_world), default=0.0)}, open(f"{a.out}/budget.json", "w"))
stats = {"arm": arm, "split": a.split, "seed": a.seed, "worlds": per_world, "run_s": time.time() - t_all,
         "base_sha256": fp0, "nan": bool(nan_any)}
json.dump(stats, open(f"{a.out}/stats.json", "w"))
print(json.dumps({k: v for k, v in stats.items() if k != "worlds"}), flush=True)
