"""EVALUATE (frozen): accuracy of the raw logits against private labels; one item per example."""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="/data/private/y.npy")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


y = np.load(a.labels)
try:
    logits = np.load(f"{a.run}/logits.npy")
except Exception as e:
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"no logits: {e}"})
    raise SystemExit(0)
if logits.shape != (len(y), 10) or not np.isfinite(logits).all():
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"bad logits {logits.shape} or non-finite"})
    raise SystemExit(0)
correct = (logits.argmax(1) == y).astype(float)
write({"valid": True, "primary": float(correct.mean()), "metrics": {"accuracy": float(correct.mean())},
       "items": {str(i): float(c) for i, c in enumerate(correct)}, "message": "ok"})
