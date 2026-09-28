"""π₀ / the fixed control (Dream-RSI's parallel refinement): open W chains from the root, then refine each chain's leaf."""


def select(nodes, budget_left, W):
    chains = [n for n in nodes if n["parent"] == "root"]
    if len(chains) < W:
        return ["root"] * min(W - len(chains), budget_left)
    by_id = {n["id"]: n for n in nodes}
    out = []
    for c in chains:
        n = c
        while n["children"]:
            n = by_id[n["children"][-1]]
        out.append(n["id"])
    return out[:budget_left]
