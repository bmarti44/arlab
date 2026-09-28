"""Closest to arlab's greedy runner: spend the whole batch refining the best-scoring leaf (the root until something wins)."""


def select(nodes, budget_left, W):
    scored = [n for n in nodes if n["leaf"] and n["status"] == "ok" and n["score"] is not None and n["score"] > 0]
    best = max(scored, key=lambda n: (n["score"], -n["order"]))["id"] if scored else "root"
    return [best] * min(W, budget_left)
