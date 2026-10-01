"""Paired search blocks and the pre-registered analysis (docs/DREAM-PROTOCOL.md).

A block = one shared opening tree + one tree per arm grown from it (--opening-from), each finalized as usual. Then
every arm's chosen node and the common root are scored on the block's own fresh holdout seeds (shared across arms), so
the root cancels and arms are compared directly. The unit of replication is the block.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from ..record import read_json, write_json
from .model import Tree

MARGIN = 0.2      # MES units: "beats" means by more than this (pre-registered)
ALPHA = 0.025     # one-sided, per comparator
PRIMARY = ("P", "G")  # D must beat both fixed policies; S (stop_first_win) is secondary


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile (Cornish-Fisher from the normal quantile; error < 1e-3 for df >= 5)."""
    z = statistics.NormalDist().inv_cdf(p)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def block_eval(campaigns: dict, seeds: list[int], out: Path) -> dict:
    """campaigns: arm -> finalized TreeCampaign (setup() done). Scores each arm's choice (or the root) on `seeds`."""
    from ..experiment import guard_failures
    first = next(iter(campaigns.values()))
    p, base = first.pack, first.state["baseline_commit"]
    sgn = 1 if p.metric.direction == "maximize" else -1

    def score(c, commit, name):
        vals = {}
        for seed in seeds:
            tdir = c.dir / "block-eval" / out.name / name / f"s{seed}"  # trials must live under the run's own dir
            res = read_json(tdir / "result.json")
            if res is None:
                res = c.settled(lambda: c.trial(c.export(commit, tdir / "surface"), seed, "holdout", tdir))
                c.clean_trial(res, tdir)
                write_json(tdir / "result.json", res)
            ok = res["status"] == "ok" and not guard_failures(c, res)
            vals[seed] = res["primary"] if ok else None
        return vals

    root = score(first, base, "root")
    if any(v is None for v in root.values()):
        raise RuntimeError(f"root failed on block seeds: {root}")
    arms = {}
    for arm, c in campaigns.items():
        ch = c.state.get("tree_choice") or {}
        commit = ch.get("commit") or base
        vals = score(c, commit, arm) if commit != base else dict(root)
        d = [0.0 if vals[s] is None else sgn * (vals[s] - root[s]) / p.metric.mes for s in seeds]  # failure = root
        nodes = [n for n in c.tree.nodes if n["id"] != "root"]
        arms[arm] = {"tag": c.tag, "policy": c.tree.meta.get("policy"), "policy_sha": c.tree.meta.get("policy_sha"),
                     "node": ch.get("node"), "commit": commit, "vals": vals, "q": sum(d) / len(d),
                     "fallback": [s for s in seeds if vals[s] is None], "nodes": c.tree.used(),
                     "calls": sum(int(n.get("agent_calls") or 1) for n in nodes if n["status"] != "pending"),
                     "stop": c.state.get("stop_reason")}
    rec = {"seeds": seeds, "root": root, "mes": p.metric.mes, "arms": arms}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "block.json", rec)
    return rec


def contrast(diffs: list[float]) -> dict:
    n = len(diffs)
    m = sum(diffs) / n
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = t_quantile(1 - ALPHA, n - 1) if n > 1 else float("nan")
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(10000))
    return {"n": n, "mean": m, "sd": sd, "lower": m - t * se, "upper": m + t * se,
            "boot_lower": boots[int(ALPHA * 10000)], "boot_upper": boots[int((1 - ALPHA) * 10000) - 1]}


def analyze(block_files: list[Path]) -> dict:
    blocks = [json.loads(Path(f).read_text()) for f in block_files]
    blocks = [b for b in blocks if all(a in b["arms"] for a in ("D", *PRIMARY))]
    out = {"blocks": len(blocks), "margin_mes": MARGIN, "alpha_one_sided": ALPHA, "quality": {}, "cost_nodes": {}, "cost_calls": {}}
    if len(blocks) < 2:
        return {**out, "verdict": "insufficient blocks"}
    for x in [a for a in ("P", "G", "S") if all(a in b["arms"] for b in blocks)]:
        out["quality"][f"D-{x}"] = contrast([b["arms"]["D"]["q"] - b["arms"][x]["q"] for b in blocks])
        out["cost_nodes"][f"D-{x}"] = contrast([b["arms"]["D"]["nodes"] - b["arms"][x]["nodes"] for b in blocks])
        out["cost_calls"][f"D-{x}"] = contrast([b["arms"]["D"]["calls"] - b["arms"][x]["calls"] for b in blocks])
    out["mean_q"] = {a: sum(b["arms"][a]["q"] for b in blocks) / len(blocks) for a in blocks[0]["arms"]}
    q, c = out["quality"], out["cost_calls"]
    if all(q[f"D-{x}"]["lower"] > MARGIN and c[f"D-{x}"]["upper"] <= 0 for x in PRIMARY):
        v = "proven"
    elif any(q[f"D-{x}"]["upper"] < MARGIN for x in PRIMARY):
        v = "denied"
    else:
        v = "inconclusive"
    out["verdict"] = v
    out["rule"] = (f"proven: for both P and G, the one-sided {1 - ALPHA:.1%} lower bound of mean(q_D - q_X) > {MARGIN} MES "
                   f"and the upper bound of mean(calls_D - calls_X) <= 0; denied: either quality upper bound < {MARGIN}; "
                   "else inconclusive (t intervals over blocks; bootstrap bounds reported as sensitivity)")
    return out


def load_campaign(pack_dir: Path, tag: str):
    from .online import TreeCampaign
    c = TreeCampaign(pack_dir, tag)
    if c.state.get("phase") != "finalized":
        raise RuntimeError(f"{tag} is not finalized")
    c.setup()
    c.tree = Tree.load(c.dir / "tree.json")
    return c
