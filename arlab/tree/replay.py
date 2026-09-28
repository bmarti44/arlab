"""Trusted replay simulator (Dream-RSI §3.2) and the objective V.

Semantics: selecting a revealed node reveals its next *recorded* child in recorded order; a node with no recorded
child left reveals nothing (that branch is exhausted). Nothing unrevealed is ever shown to the policy, and revealed
nodes are relabelled r0001, r0002, ... in reveal order so recorded ids/orders cannot leak the recording's shape.
"""
from __future__ import annotations

from .model import DROPPED, children, clean_selection, public_view

BETA = 0.5
BETAS = (0.0, 0.5, 1.0)


class Replay:
    def __init__(self, recorded: list[dict]):
        rec = [n for n in recorded if n["status"] not in DROPPED]
        self.rec = {n["id"]: n for n in rec}
        self.kids = children(rec)
        self.ptr: dict[str, int] = {}
        self.label = {"root": "root"}       # recorded id -> public id
        self.back = {"root": "root"}        # public id -> recorded id
        root = self.rec["root"]
        self.nodes = [{**root, "order": 0}]
        self.root_score = root.get("score") or 0.0

    def expand(self, parents: list[str]) -> list[dict]:
        out = []
        for pub in parents:
            rid = self.back[pub]
            ks, i = self.kids[rid], self.ptr.get(rid, 0)
            if i >= len(ks):
                continue                    # exhausted: the recording has no further child here
            self.ptr[rid] = i + 1
            child = self.rec[ks[i]]
            new = f"r{len(self.nodes):04d}"
            self.label[child["id"]], self.back[new] = new, child["id"]
            n = {k: child.get(k) for k in ("status", "score", "cost_s", "depth")}
            n.update(id=new, parent=pub, order=len(self.nodes))
            self.nodes.append(n)
            out.append(n)
        return out


def gain(n: dict, root_score: float) -> float | None:
    return None if n.get("score") is None or n["status"] != "ok" else n["score"] - root_score


def replay(policy, recorded: list[dict], W: int, budget: int, max_stalls: int = 3) -> dict:
    """Run one policy on one recorded tree. `policy` has select(view, budget_left, W) -> list[id].

    `budget` is the run's public node budget (tree.meta["budget"]), never the recording's size (that would leak it).
    """
    sim = Replay(recorded)
    used = stalls = invalid = batches = 0
    stop = "budget"
    curve, trace, best = [], [], 0.0
    while used < budget:
        sel, bad = clean_selection(policy.select(public_view(sim.nodes), budget - used, W), sim.nodes, W, budget - used)
        invalid += bad
        if not sel:
            stop = "policy"
            break
        batches += 1
        trace.append([sim.back[x] for x in sel])
        new = sim.expand(sel)
        if not new:
            stalls += 1
            if stalls >= max_stalls:
                stop = "stalled"
                break
            continue
        stalls = 0
        for n in new:
            used += 1
            g = gain(n, sim.root_score)
            best = max(best, g) if g is not None else best
            curve.append(round(best, 6))
    n_ref = max(1, budget)
    return {"N": used, "N_ref": n_ref, "best": best, "batches": batches, "invalid": invalid, "stop": stop, "curve": curve,
            "V": {str(b): objective(best, used, n_ref, b) for b in BETAS}, "revealed": [sim.back[n["id"]] for n in sim.nodes[1:]], "trace": trace}


def objective(best_gain: float, n: int, n_ref: int, beta: float = BETA) -> float:
    """V = max(0, best gain over the root, in MES units) − β·N/N_ref."""
    return max(0.0, best_gain) - beta * n / max(1, n_ref)


def mean_v(results: list[dict], beta: float = BETA) -> float:
    return sum(r["V"][str(beta)] for r in results) / len(results) if results else float("-inf")
