"""Frozen RUN entry point for nanochat-lite: owns the token budget and batch loop (arlab.lib.lm)."""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import sys

from arlab.lib.lm import token_budget_loop

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--budget", type=int, required=True)
a = ap.parse_args()
sys.path.insert(0, "/work")
import train as surface  # noqa: E402  (the editable surface)

BATCH, SEQ_LEN, VOCAB = 64, 1024, 8192
token_budget_loop(surface, "/data/train/tokens.bin", a.budget, BATCH, SEQ_LEN, a.seed, a.out, {"vocab_size": VOCAB, "device": "cuda"})
