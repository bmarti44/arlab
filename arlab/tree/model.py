"""tree.json: one node per attempt. Written atomically after every change (kill -9 safe)."""
from __future__ import annotations

from pathlib import Path

from ..record import read_json, write_json

# The only node fields a policy ever sees (no descriptions, commits or recorded ids).
PUBLIC = ("id", "parent", "depth", "order", "status", "score", "cost_s")  # no tags/descriptions: agent-written free text
# can carry information about other (in replay: unrevealed) branches
FAILED = ("crash", "timeout", "oom", "invalid", "guard_fail", "no_op", "skip")   # real attempts that produced no score
DROPPED = ("infra_error", "contended", "pending")                                # not attempts: never kept in a tree


class Tree:
    def __init__(self, path: Path, meta: dict | None = None, nodes: list[dict] | None = None):
        self.path = Path(path)
        self.meta = meta or {}
        self.nodes = nodes or [{"id": "root", "parent": None, "depth": 0, "order": 0, "status": "ok", "score": 0.0, "tag": "root", "cost_s": 0.0}]

    @classmethod
    def load(cls, path: Path) -> "Tree | None":
        d = read_json(path)
        return cls(path, d["meta"], d["nodes"]) if d else None

    def save(self):
        write_json(self.path, {"meta": self.meta, "nodes": self.nodes})

    def get(self, nid: str) -> dict:
        return next(n for n in self.nodes if n["id"] == nid)

    def new_id(self) -> str:
        return f"n{max([int(n['id'][1:]) for n in self.nodes if n['id'] != 'root'] + [0]) + 1:04d}"

    def used(self) -> int:
        return sum(1 for n in self.nodes if n["id"] != "root" and n["status"] not in DROPPED)

    def path_to(self, nid: str) -> list[dict]:
        out = []
        while nid:
            n = self.get(nid)
            out.append(n)
            nid = n["parent"]
        return out[::-1]


def children(nodes: list[dict]) -> dict[str, list[str]]:
    kids: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for n in sorted(nodes, key=lambda n: n["order"]):
        if n["parent"] is not None:
            kids[n["parent"]].append(n["id"])
    return kids


def public_view(nodes: list[dict]) -> list[dict]:
    kids = children(nodes)
    return [{**{k: n.get(k) for k in PUBLIC}, "children": kids[n["id"]], "leaf": n["id"] != "root" and not kids[n["id"]]}
            for n in sorted(nodes, key=lambda n: n["order"])]


def frontier(nodes: list[dict]) -> set[str]:
    kids = children(nodes)
    return {"root"} | {n["id"] for n in nodes if n["id"] != "root" and not kids[n["id"]]}


def clean_selection(sel, nodes: list[dict], W: int, budget_left: int) -> tuple[list[str], int]:
    """Keep ids in A(T) (repeats allowed: several children of one node in a batch), at most min(W, budget_left)."""
    if not isinstance(sel, list):
        return [], 1
    ok = frontier(nodes)
    good = [str(s) for s in sel if str(s) in ok]
    return good[:max(0, min(W, budget_left))], len(sel) - len(good)
