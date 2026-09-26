"""Frozen evaluator: scores raw logits from RUN against private labels. Never uses surface-computed numbers."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="/data/private/y.npy"); a = ap.parse_args()
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
y = np.load(a.labels)

def write(obj):
    tmp = a.out + ".tmp"; json.dump(obj, open(tmp, "w")); os.replace(tmp, a.out)

try:
    logits = np.load(f"{a.run}/logits.npy")
except Exception as e:
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"no logits: {e}"}); raise SystemExit(0)
if logits.shape != (len(y), int(y.max()) + 1) or not np.isfinite(logits).all():
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": f"bad logits shape {logits.shape} or non-finite"})
    raise SystemExit(0)
correct = (logits.argmax(1) == y).astype(float)
items = {str(i): float(c) for i, c in enumerate(correct)} if cfg["items"] else None
write({"valid": True, "primary": float(correct.mean()), "metrics": {"accuracy": float(correct.mean())}, "items": items, "message": "ok"})
