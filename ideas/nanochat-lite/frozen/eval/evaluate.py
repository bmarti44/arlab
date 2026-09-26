"""Frozen EVALUATE for nanochat-lite: val_bpb from the model's raw logits + causality check + params."""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

from arlab.lib.lm import causality_check, score_bpb

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/data/private")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg})
    sys.exit(0)


sys.path.insert(0, "/work")
try:
    import train as surface
    model = surface.load(f"{a.run}/model.pt", "cuda")
    model.eval()
except Exception:
    invalid("could not load the checkpoint with the surface's model: " + traceback.format_exc()[-800:])
rows = np.load(f"{a.data}/rows.npy")
token_bytes = np.load(f"{a.data}/token_bytes.npy")
try:
    caus = causality_check(model, rows, len(token_bytes))
    if not caus["causal"]:
        invalid(f"non-causal model: perturbing future tokens changed past log-probs by {caus['max_diff']:.3g}")
    r = score_bpb(model, rows, token_bytes)
except ValueError as e:
    invalid(str(e))
if not r["valid"]:
    invalid(r["message"])
params_m = sum(p.numel() for p in model.parameters()) / 1e6
write({"valid": True, "primary": r["val_bpb"], "items": None, "message": "ok",
       "metrics": {"val_bpb": r["val_bpb"], "params_m": params_m, "eval_tokens": r["tokens"], "causal_max_diff": caus["max_diff"]}})
print(f"val_bpb {r['val_bpb']:.6f} params_m {params_m:.2f} causal_max_diff {caus['max_diff']:.2g}")
