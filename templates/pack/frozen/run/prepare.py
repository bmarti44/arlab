"""PREPARE: write immutable data into --out. Layout (see docs/PACK-AUTHORING.md):
train/  validation/{public,private}/  holdout/{public,private}/  [+ splits.json for item packs]
Must be deterministic: pin dataset revisions / repo SHAs / seeds here (this file is part of data_hash)."""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
raise SystemExit("TODO: write the data")
