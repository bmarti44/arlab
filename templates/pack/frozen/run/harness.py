"""RUN entry point (frozen). Owns the budget: drives the surface, writes raw outputs + /out/budget.json."""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", required=True)
a = ap.parse_args()
sys.path.insert(0, "/work")
import surface_module  # noqa: E402  TODO: the editable surface
raise SystemExit("TODO: run the surface under the budget, save raw outputs to /out, then write budget.json")
json.dump({"steps": 0}, open(f"{a.out}/budget.json", "w"))
