"""Deterministic checks: splits and the scorer."""
import json
import subprocess
import sys

import numpy as np


def test_splits_disjoint_and_sized():
    sp = json.load(open("/data/splits.json"))
    assert (sp["validation"], sp["holdout"]) == (449, 450)
    tr, va, ho = (np.load(p) for p in ("/data/train/X.npy", "/data/validation/public/X.npy", "/data/holdout/public/X.npy"))
    assert len(tr) + len(va) + len(ho) == 1797
    rows = lambda a: {r.tobytes() for r in a}
    assert len(rows(va) & rows(ho)) <= 5  # digits has a few exact-duplicate images; indices are disjoint by construction


def _eval(tmp_path, logits):
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "logits.npy", logits)
    subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", str(tmp_path / "m.json"),
                    "--labels", "/data/validation/private/y.npy"], check=True)
    return json.load(open(tmp_path / "m.json"))


def test_scorer(tmp_path):
    y = np.load("/data/validation/private/y.npy")
    m = _eval(tmp_path, np.eye(10)[y])
    assert m["valid"] and m["primary"] == 1.0 and len(m["items"]) == 449


def test_scorer_rejects_nan(tmp_path):
    assert _eval(tmp_path, np.full((449, 10), np.nan))["valid"] is False
