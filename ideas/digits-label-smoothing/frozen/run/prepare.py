"""PREPARE: sklearn digits (bundled, no download), fixed-seed 50/25/25 split into train / validation / holdout."""
import argparse
import json
import os

import numpy as np
from sklearn.datasets import load_digits

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
out = ap.parse_args().out
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
perm = np.random.RandomState(0).permutation(len(X))
n_tr, n_va = len(X) // 2, len(X) // 4
parts = {"train": perm[:n_tr], "validation": perm[n_tr:n_tr + n_va], "holdout": perm[n_tr + n_va:]}
os.makedirs(f"{out}/train")
np.save(f"{out}/train/X.npy", X[parts["train"]])
np.save(f"{out}/train/y.npy", y[parts["train"]])
for split in ("validation", "holdout"):
    os.makedirs(f"{out}/{split}/public")
    os.makedirs(f"{out}/{split}/private")
    np.save(f"{out}/{split}/public/X.npy", X[parts[split]])
    np.save(f"{out}/{split}/private/y.npy", y[parts[split]])
json.dump({"validation": len(parts["validation"]), "holdout": len(parts["holdout"])}, open(f"{out}/splits.json", "w"))
print({k: len(v) for k, v in parts.items()})
