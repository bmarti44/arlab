"""Frozen planner subprocess for stencil-focus: the ONLY place the surface runs during RUN.

The harness starts `python /frozen/planner.py --out <file>`; this process imports /work/stencil.py, calls
plan(sentences, current, query, window) per item and writes {item_id: {"plan": {...}} | {"error": "..."}} as JSON.
The harness (a separate, trusted process that never imports the surface) re-checks every plan, builds the prompts,
calls the model and records the outputs, so surface code cannot touch generation or the recorded outputs.
"""
import argparse
import dataclasses
import importlib.util
import json
import os

import window

REAL_W = 1536


def read(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def load_items(pub="/data/public"):
    return [(it, window.W, "main") for it in read(f"{pub}/items.jsonl")] + [(it, REAL_W, "real") for it in read(f"{pub}/real.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("surface_stencil", "/work/stencil.py")
    surface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(surface)
    out = {}
    for it, w, _kind in load_items():
        try:
            win = window.Window(it, w)
            p = surface.plan([dict(r) for r in win.sentences], it["sessions"][-1], it["query"], win)
            p = dataclasses.asdict(p) if dataclasses.is_dataclass(p) and not isinstance(p, type) else p
            out[it["id"]] = {"plan": json.loads(json.dumps(p))}
        except Exception as e:  # noqa: BLE001  (recorded; the run becomes invalid in EVALUATE)
            out[it["id"]] = {"error": f"{type(e).__name__}: {e}"[:1000]}
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
