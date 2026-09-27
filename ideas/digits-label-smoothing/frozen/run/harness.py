"""RUN entry point (frozen): exactly 1,500 optimizer steps of batch 64; the surface supplies model/optimizer/step."""
import argparse
import json
import sys

import numpy as np
import torch

STEPS, BATCH = 1500, 64
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
torch.manual_seed(a.seed)
sys.modules["sklearn.datasets"] = None  # the eval images ship with sklearn: the surface must learn, not look labels up
sys.path.insert(0, "/work")
import model as surface  # noqa: E402  (the editable surface)

X, y = torch.from_numpy(np.load("/data/train/X.npy")), torch.from_numpy(np.load("/data/train/y.npy")).long()
state = surface.build({"n_in": X.shape[1], "n_out": 10, "total_steps": STEPS})
g = torch.Generator().manual_seed(a.seed)
for step in range(STEPS):
    idx = torch.randint(0, len(X), (BATCH,), generator=g)
    surface.train_step(state, X[idx], y[idx], step, STEPS)
with torch.no_grad():
    logits = surface.predict(state, torch.from_numpy(np.load("/data/public/X.npy")))
np.save(f"{a.out}/logits.npy", np.asarray(logits, dtype=np.float64))
json.dump({"steps": STEPS}, open(f"{a.out}/budget.json", "w"))
print("done")
