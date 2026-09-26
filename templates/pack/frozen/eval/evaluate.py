"""EVALUATE (frozen, never visible to the agent). Scores raw RUN outputs in /run_out against /data/private.
Writes /result/metrics.json atomically: {"valid", "primary", "metrics", "items", "message"}."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": "TODO: score"})
