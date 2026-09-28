"""Editable surface: the loop machinery of a looped mid-block retrofit of Qwen3-0.6B, and nothing else.

The frozen runtime (frozen/run/looprt.py) owns the base model, its LoRA adapters (r16 on all projections), the
forward, the answer loss, AdamW, the LR schedule and the training data; it is identical for every arm. It runs
    p = prelude(embed(ids))                     layers[:LOOP_START]
    h = loop(p, block, ctx)                     <- this file; block(x) runs layers[LOOP_START:LOOP_END] once
    logits = lm_head(norm(coda(h)))             layers[LOOP_END:]
BASELINE STATE = the no-loop arm (LOOPS_TRAIN = (1,), LOOPS_EVAL = 1): h = block(p), exactly the fine-tuned model.
Loop arm: h = block(p); then for j = 2..K: h = h + gate_j * (block(h + inj(p)) - h), gates and inj zero at init,
so any K reproduces the base model at init (identity-at-init).
ctx.training / ctx.progress (fraction of the budget) are available; in training ctx.readout(x) gives the frozen
answer loss of an intermediate state x through the coda, and ctx.aux_loss (a scalar tensor) is added to the loss.
"""
import random

import torch
import torch.nn as nn

LOOP_START, LOOP_END = 12, 16      # loop block = layers 12..15 of 28 (frozen limits: 1 <= start < end <= 27, size <= 6)
LOOPS_TRAIN = (1,)                 # K sampled uniformly from this tuple each training step (e.g. (1, 2, 3, 4))
LOOPS_EVAL = 1                     # K at evaluation (inference cost guard: effective depth <= 2x)
K_MAX = 8                          # gates exist for K <= K_MAX
LOOP_LR = 1e-2                     # peak LR of the loop's own parameters (the frozen schedule shape applies)


class Loop(nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        self.gates = nn.Parameter(torch.zeros(K_MAX - 1, d))  # gate_j for passes j = 2..K_MAX
        self.inj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.inj.weight)
        self.rng = random.Random(seed)
        self.k_train, self.k_eval = tuple(LOOPS_TRAIN), LOOPS_EVAL

    def forward(self, p, block, ctx):
        k = self.rng.choice(self.k_train) if ctx.training else self.k_eval
        h = block(p)
        for j in range(1, k):
            g = block(h + self.inj(p.to(self.inj.weight.dtype)).to(h.dtype))
            h = h + self.gates[j - 1].to(h.dtype) * (g - h)
        return h


def build(d_model, n_layers, seed):
    return Loop(d_model, seed)
