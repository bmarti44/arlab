"""Frozen RUN entry point: owns the step budget and the batch loop; the surface supplies the model."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import argparse, json, sys, time
import numpy as np

STEPS, BATCH = 1500, 2048
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--split", required=True)
a = ap.parse_args()
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
sys.path.insert(0, "/work")
import model  # the surface

rng = np.random.default_rng(a.seed if cfg["seed_noise"] else 0)
X, y = np.load("/data/train/X.npy"), np.load("/data/train/y.npy")
state = model.build({"n_in": X.shape[1], "n_out": int(y.max()) + 1, "rng": rng})
t0 = time.time()
for step in range(STEPS):
    idx = rng.integers(0, len(X), BATCH)
    model.train_step(state, (X[idx], y[idx]), step, STEPS)
np.save(f"{a.out}/logits.npy", np.asarray(model.predict(state, np.load("/data/public/X.npy")), dtype=np.float64))
model.save(state, f"{a.out}/model.npz")
json.dump({"steps": STEPS}, open(f"{a.out}/budget.json", "w"))
print(f"trained {STEPS} steps in {time.time() - t0:.2f}s")
