"""Deterministic CPU checks of nanochat-lite data and the frozen LM scorer (run by `arlab check`)."""
import json
import math

import numpy as np
import torch

from arlab.lib.lm import causality_check, score_bpb

D = "/data"


def test_data_layout_and_disjoint_eval_sets():
    info = json.load(open(f"{D}/info.json"))
    assert info["vocab_size"] == 8192 and info["seq_len"] == 1024
    assert np.memmap(f"{D}/train/tokens.bin", dtype=np.uint16, mode="r").size >= 200_000_000
    v = np.load(f"{D}/validation/private/rows.npy")
    h = np.load(f"{D}/holdout/private/rows.npy")
    assert v.shape == h.shape == (2048, 1025)
    assert not {r.tobytes() for r in v[:, :64]} & {r.tobytes() for r in h[:, :64]}
    tb = np.load(f"{D}/validation/private/token_bytes.npy")
    assert tb.shape == (8192,) and (tb == 0).sum() == 4 and tb.max() <= 64


class Uniform(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(*x.shape, 8192)


class Peeking(torch.nn.Module):
    """Non-causal: position t sees the mean of all tokens."""
    def forward(self, x):
        z = torch.zeros(*x.shape, 8192)
        z[..., 0] = x.float().mean(1, keepdim=True)
        return z


def test_scorer_uniform_model_matches_closed_form():
    rows = np.load(f"{D}/validation/private/rows.npy")[:4]
    tb = np.load(f"{D}/validation/private/token_bytes.npy")
    r = score_bpb(Uniform(), rows, tb, device="cpu")
    y = rows[:, 1:]
    b = tb[y]
    expect = math.log(8192) * (b > 0).sum() / (math.log(2) * b.sum())
    assert abs(r["val_bpb"] - expect) < 1e-6


def test_causality_check():
    rows = np.load(f"{D}/validation/private/rows.npy")[:4]
    assert causality_check(Uniform(), rows, 8192, device="cpu")["causal"]
    assert not causality_check(Peeking(), rows, 8192, device="cpu")["causal"]
