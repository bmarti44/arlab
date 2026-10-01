"""Handwritten control (DREAM-PROTOCOL.md): parallel_refine that stops once any node reaches a full MES (score >= 1)."""


def select(nodes, budget_left, W):
    if any(n["score"] is not None and n["score"] >= 1.0 for n in nodes if n["id"] != "root"):
        return []
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
