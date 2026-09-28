"""A simple bandit over chains (subtrees of the root): UCB on each chain's best score; a new chain is one more arm."""
import math

C = 0.5          # exploration weight, in MES units
NEW_ARM = 0.3    # optimistic value of an untried chain


def select(nodes, budget_left, W):
    by_id = {n["id"]: n for n in nodes}
    chains = {}
    for n in nodes:
        if n["id"] == "root":
            continue
        top = n
        while top["parent"] != "root":
            top = by_id[top["parent"]]
        chains.setdefault(top["id"], []).append(n)
    total = max(1, len(nodes) - 1)
    out, pulls = [], {}
    for _ in range(min(W, budget_left)):
        best_arm, best_u = "root", NEW_ARM + C * math.sqrt(math.log(total + 1))
        for cid, ms in chains.items():
            k = len(ms) + pulls.get(cid, 0)
            vals = [m["score"] for m in ms if m["score"] is not None]
            u = (max(vals) if vals else -1.0) + C * math.sqrt(math.log(total + 1) / k)
            if u > best_u:
                best_arm, best_u = cid, u
        if best_arm == "root":
            out.append("root")
            continue
        pulls[best_arm] = pulls.get(best_arm, 0) + 1
        leaves = [m for m in chains[best_arm] if m["leaf"]]
        ok = [m for m in leaves if m["score"] is not None]
        pick = max(ok, key=lambda m: m["score"]) if ok else max(leaves, key=lambda m: m["order"])
        out.append(pick["id"])
    return out
