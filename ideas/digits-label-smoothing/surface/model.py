"""Editable surface: a 2-layer MLP, its optimizer and training step (the frozen harness runs 1,500 steps of 64)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 64
LR = 1e-3


def build(config):
    model = nn.Sequential(nn.Linear(config["n_in"], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, config["n_out"]))
    return {"model": model, "opt": torch.optim.Adam(model.parameters(), lr=LR)}


def train_step(state, x, y, step, total_steps):
    state["model"].train()
    loss = F.cross_entropy(state["model"](x), y)
    state["opt"].zero_grad()
    loss.backward()
    state["opt"].step()
    return loss.item()


def predict(state, x):
    state["model"].eval()
    return state["model"](x)
