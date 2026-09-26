"""Deterministic checks of the fixture pack (run by `arlab check`)."""
import json
import subprocess
import sys

import numpy as np

LABELS = "/data/validation/private/y.npy"


def evaluate(tmp_path, logits):
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "logits.npy", logits)
    out = tmp_path / "m.json"
    subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", str(out), "--labels", LABELS], check=True)
    return json.load(open(out))


def test_data_shapes():
    assert np.load("/data/train/X.npy").shape == (20000, 32)
    assert np.load(LABELS).shape == (2000,)
    assert json.load(open("/data/splits.json"))["holdout"] == 2000


def test_evaluator_rejects_nan(tmp_path):
    m = evaluate(tmp_path, np.full((2000, 10), np.nan))
    assert m["valid"] is False and m["primary"] is None


def test_evaluator_scores_perfect_logits(tmp_path):
    y = np.load(LABELS)
    m = evaluate(tmp_path, np.eye(10)[y])
    assert m["valid"] and m["primary"] == 1.0
