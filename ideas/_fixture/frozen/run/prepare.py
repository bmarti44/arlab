"""Synthetic 10-class data from a fixed random teacher. Deterministic."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = np.random.default_rng(12345)
D, K = 32, 10
W1, W2 = rng.normal(size=(D, 48)), rng.normal(size=(48, K))

def make(n):
    X = rng.normal(size=(n, D)).astype(np.float32)
    logits = np.tanh(X @ W1) @ W2 + 1.2 * rng.normal(size=(n, K))
    return X, logits.argmax(1).astype(np.int64)

for split, n in [("train", 20000), ("validation", 2000), ("holdout", 2000)]:
    X, y = make(n)
    if split == "train":
        os.makedirs(f"{a.out}/train"); np.save(f"{a.out}/train/X.npy", X); np.save(f"{a.out}/train/y.npy", y)
    else:
        os.makedirs(f"{a.out}/{split}/public"); os.makedirs(f"{a.out}/{split}/private")
        np.save(f"{a.out}/{split}/public/X.npy", X); np.save(f"{a.out}/{split}/private/y.npy", y)
json.dump({"validation": 2000, "holdout": 2000}, open(f"{a.out}/splits.json", "w"))
print("prepared")
