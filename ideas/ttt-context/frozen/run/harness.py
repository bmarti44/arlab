"""Frozen RUN entry point for ttt-context. It owns the models, the item loop, the per-item TTT deadline, the state
restore and verification, and greedy answering (all in tttlib.run_items); the surface (/work/ttt.py) only supplies
adapt(model, ctx) -> {"weights": {name: tensor}, "doc_in_context": bool}.

Outputs in --out (scored by the frozen evaluator, which never trusts a surface number):
  preds.json    {id: [generated token ids]}
  items.json    {id: {ttt_s, prefill_s, answer_s, changed_tensors, doc_in_context, nonfinite, reset_ok, probe_diff}}
  stats.json    SHA-256 of the weights at load and of both models at the end, restore/probe results, timings
  budget.json   {"ttt_seconds": total TTT seconds over all items}
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from common import MODEL_DIR  # noqa: E402
from tttlib import fresh_import, run_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--ttt-seconds", type=float, required=True, help="per-item wall-clock TTT budget")
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=0, help="pilot/smoke only: the first N items")
ap.add_argument("--data", default="/data")
ap.add_argument("--model", default=MODEL_DIR, help="tests only: a tiny local model")
a = ap.parse_args()
dev = a.device
torch.manual_seed(a.seed)

items = json.load(open(f"{a.data}/public/items.json"))
flat, off = np.load(f"{a.data}/public/docs.npy"), np.load(f"{a.data}/public/offsets.npy")
docs = [flat[off[i]:off[i + 1]] for i in range(len(items))]
if a.limit:
    items, docs = items[:a.limit], docs[:a.limit]

sys.path.insert(0, "/work")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                             attn_implementation="sdpa").to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f"model loaded in {time.time() - t0:.1f}s; {len(items)} items, split {a.split}, seed {a.seed}, "
      f"ttt-seconds {a.ttt_seconds}", flush=True)
dump = json.dump      # bound before any surface code runs
preds, per, stats = run_items(model, lambda: fresh_import("ttt"), docs, items, ttt_seconds=a.ttt_seconds,
                              seed=a.seed, device=dev, log=lambda s: print(s, flush=True))
stats |= {"split": a.split, "limit": a.limit, "load_model_s": time.time() - t0 - stats["wall_s"],
          "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0}
dump(preds, open(f"{a.out}/preds.json", "w"))
dump(per, open(f"{a.out}/items.json", "w"))
dump(stats, open(f"{a.out}/stats.json", "w"))
dump({"ttt_seconds": sum(v["ttt_s"] for v in per.values())}, open(f"{a.out}/budget.json", "w"))
print(json.dumps(stats), flush=True)
