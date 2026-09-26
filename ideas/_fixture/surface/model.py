"""Editable surface: a 2-layer numpy MLP trained with plain SGD."""
import numpy as np

CONFIG = {"hidden": 128, "lr": 0.002}


def build(config):
    rng, h = config["rng"], CONFIG["hidden"]
    return {"W1": rng.normal(size=(config["n_in"], h)) / np.sqrt(config["n_in"]), "b1": np.zeros(h),
            "W2": rng.normal(size=(h, config["n_out"])) / np.sqrt(h), "b2": np.zeros(config["n_out"])}


def forward(s, X):
    H = np.maximum(X @ s["W1"] + s["b1"], 0)
    return H, H @ s["W2"] + s["b2"]


def train_step(s, batch, step, total_steps):
    X, y = batch
    H, Z = forward(s, X)
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    G = P.copy(); G[np.arange(len(y)), y] -= 1; G /= len(y)
    dH = (G @ s["W2"].T) * (H > 0)
    for k, g in (("W2", H.T @ G), ("b2", G.sum(0)), ("W1", X.T @ dH), ("b1", dH.sum(0))):
        s[k] -= CONFIG["lr"] * g
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def predict(s, X):
    return forward(s, X)[1]


def save(s, path):
    np.savez(path, **s)
