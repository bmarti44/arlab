"""Transfer check (secondary, after FINALIZE, never the verdict): replay the kept controller, the baseline and the
references on a second cache (another model: IDEA.md "Transfer check") at B' = 4 x its mean train length, and report
each arm's accuracy and its paired difference to the baseline (mean over seeds per problem, normal 95 % CI).

The transfer cache is sampled by build/sample.sh (OUT=..., --model ..., --n 16, --splits train:300,holdout) and
prepared with /prepare/prepare.py --traces <it> --out <data>. Run in the pack image, CPU only, e.g.:
  docker run --rm --network none --user 1000:1000 -e PYTHONPATH=/frozen -v <sealed>/frozen/run:/frozen:ro \
    -v <sealed>/frozen/eval:/eval:ro -v <data>:/tdata:ro -v <kept surface>:/arms/kept:ro -v <sealed>/surface:/arms/baseline:ro \
    -v <out>:/out IMG python /frozen/transfer.py --data /tdata --out /out --arm kept=/arms/kept \
    --arm baseline=/arms/baseline --arm conf_vote=/frozen/ref_conf_vote --arm asc=/frozen/ref_asc
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="prepared transfer data root (train/, holdout/public, holdout/private)")
ap.add_argument("--out", required=True)
ap.add_argument("--arm", action="append", required=True, help="name=controller dir; one must be named baseline")
ap.add_argument("--seeds", default="101,102,103")
ap.add_argument("--replicates", type=int, default=8)
ap.add_argument("--budget-tokens", type=int, default=0, help="default: suggested_budget_tokens from info.json")
ap.add_argument("--evaluate", default="/eval/evaluate.py")
a = ap.parse_args()

B = a.budget_tokens or json.load(open(f"{a.data}/info.json"))["suggested_budget_tokens"]
view = f"{a.out}/data"                       # the harness/evaluator layout: public, private, train
os.makedirs(view, exist_ok=True)
for name, src in (("public", "holdout/public"), ("private", "holdout/private"), ("train", "train")):
    if not os.path.lexists(f"{view}/{name}"):
        os.symlink(f"{a.data}/{src}", f"{view}/{name}")
arms = dict(x.split("=", 1) for x in a.arm)
assert "baseline" in arms, "one --arm must be named baseline"
items = {}
for name, work in arms.items():
    for seed in map(int, a.seeds.split(",")):
        d = f"{a.out}/{name}-s{seed}"
        os.makedirs(d, exist_ok=True)
        run = [sys.executable, f"{HERE}/harness.py", "--out", d, "--seed", str(seed), "--split", "transfer",
               "--budget-tokens", str(B), "--replicates", str(a.replicates), "--data", view, "--work", work]
        ev = [sys.executable, a.evaluate, "--run", d, "--out", f"{d}/metrics.json", "--budget-tokens", str(B),
              "--replicates", str(a.replicates), "--data", view]
        for cmd in (run, ev):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        m = json.load(open(f"{d}/metrics.json"))
        if not m["valid"]:
            sys.exit(f"{name} seed {seed}: invalid: {m['message']}")
        items.setdefault(name, []).append(m["items"])
pids = sorted(items["baseline"][0])
acc = {k: np.array([np.mean([s[p] for s in v]) for p in pids]) for k, v in items.items()}
report = {"budget_tokens": B, "n_problems": len(pids), "seeds": a.seeds, "arms": {}}
for k, v in acc.items():
    d = v - acc["baseline"]
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    report["arms"][k] = {"accuracy": float(v.mean()), "d_vs_baseline": float(d.mean()), "se": se,
                         "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
json.dump(report, open(f"{a.out}/transfer.json", "w"), indent=1)
print(json.dumps(report, indent=1))
