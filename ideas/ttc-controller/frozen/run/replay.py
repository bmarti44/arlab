"""The replay environment (harness side): cache loading, seeded orders and one budgeted episode.

An episode = one replicate r at one budget level: every problem of the split once, in a seeded order, with a pooled
token allowance pool = budget x n_problems. The controller can only open the NEXT trace of the problem's seeded
permutation and reveal it in chunks of 32 tokens. A read reveals min(ceil32(n), tokens left in the trace, pool left)
tokens and charges exactly that; with pool left 0 it is refused. A trace read to its end reveals its length, finish
reason and an opaque label (a0, a1, ... in first-reveal order within this problem and episode; None = no answer).
Orders depend only on (seed, replicate) and the traces' permutation on (seed, replicate, pid).
"""
import base64
import hashlib

import numpy as np

CHUNK = 32
SIGNALS = ("logprob", "conf", "ent")
LEVELS = (1.0, 0.5, 2.0)          # budget multipliers: the primary level first, then the two guard levels


def load_cache(path: str, limit: int = 0) -> dict:
    z = np.load(path)
    c = {k: z[k] for k in z.files}
    n = len(c["pids"]) if not limit else min(limit, len(c["pids"]))
    T = c["lengths"].shape[1]
    c.update(pids=c["pids"][:n].astype(str), lengths=c["lengths"][:n].astype(np.int64), answers=c["answers"][:n].astype(str),
             finish=c["finish"][:n], n=n, T=T, sig=np.stack([c[s] for s in SIGNALS]))   # (3, total) float16
    if "correct" in c:
        c["correct"] = c["correct"][:n]
    return c


def _rng(*key) -> np.random.Generator:
    h = hashlib.sha256(":".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def problem_order(seed: int, r: int, n: int) -> np.ndarray:
    return _rng("order", seed, r).permutation(n)


def trace_perm(seed: int, r: int, pid: str, T: int) -> np.ndarray:
    return _rng("traces", seed, r, pid).permutation(T)


def controller_seed(seed: int, r: int, level: float) -> int:
    return int(_rng("controller", seed, r, level).integers(2**31))


def charge(n: int, left_in_trace: int, pool_left: int) -> int:
    return int(min(-(-n // CHUNK) * CHUNK, left_in_trace, pool_left))


def encode(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode()


def decode(s: str, k: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).reshape(len(SIGNALS), k).astype(np.float32)


class Episode:
    """Serves one controller for one (seed, replicate, level); records the read log the evaluator replays."""

    def __init__(self, c: dict, seed: int, r: int, budget: int):
        self.c, self.seed, self.r, self.budget = c, seed, r, int(budget)
        self.pool = self.pool_left = self.budget * c["n"]
        self.order = problem_order(seed, r, c["n"])
        self.problems, self.cur = [], None

    def begin(self, i: int) -> dict:
        q = int(self.order[i])
        pid = str(self.c["pids"][q])
        self.cur = {"q": q, "pid": pid, "perm": trace_perm(self.seed, self.r, pid, self.c["T"]), "pos": [], "labels": {},
                    "events": []}
        return {"index": i, "problems_left": self.c["n"] - i, "pool_left": self.pool_left, "budget": self.budget,
                "max_traces": self.c["T"]}

    def handle(self, msg: dict) -> dict:
        p, op = self.cur, msg.get("op")
        if op == "open":
            if len(p["pos"]) >= self.c["T"]:
                return {"ok": False, "err": "no more traces"}
            p["pos"].append(0)
            p["events"].append(["o"])
            return {"ok": True, "t": len(p["pos"]) - 1}
        if op == "read":
            t, n = msg.get("t"), msg.get("n")
            if type(t) is not int or not 0 <= t < len(p["pos"]) or type(n) is not int or n < 1:
                return {"ok": False, "err": "bad read (t must be an opened trace, n an int >= 1)"}
            j = int(p["perm"][t])
            L = int(self.c["lengths"][p["q"], j])
            if p["pos"][t] >= L:
                return {"ok": False, "err": "trace already read to its end"}
            if self.pool_left <= 0:
                return {"ok": False, "err": "budget"}
            k = charge(n, L - p["pos"][t], self.pool_left)
            start = int(self.c["offsets"][p["q"] * self.c["T"] + j]) + p["pos"][t]
            p["pos"][t] += k
            self.pool_left -= k
            p["events"].append(["r", t, n, k])
            rep = {"ok": True, "k": k, "sig": encode(self.c["sig"][:, start:start + k]), "pool_left": self.pool_left,
                   "done": p["pos"][t] == L}
            if rep["done"]:
                v = str(self.c["answers"][p["q"], j])
                rep.update(length=L, finish=("stop", "length")[int(self.c["finish"][p["q"], j])],
                           label=p["labels"].setdefault(v, f"a{len(p['labels'])}") if v else None)
            return rep
        return {"ok": False, "err": f"unknown op {op!r}"}

    def finish(self, label, solve_s: float = 0.0):
        p = self.cur
        self.problems.append({"pid": p["pid"], "events": p["events"], "answer": label, "solve_s": round(solve_s, 4)})
        self.cur = None

    @property
    def charged(self) -> int:
        return self.pool - self.pool_left


def train_problems(c: dict) -> list:
    """What fit() receives: per train problem, its traces with full signals, opaque labels and correct flags."""
    sig = c["sig"].astype(np.float32)
    out = []
    for q in range(c["n"]):
        labels, traces = {}, []
        for j in range(c["T"]):
            a, b = int(c["offsets"][q * c["T"] + j]), int(c["offsets"][q * c["T"] + j + 1])
            v = str(c["answers"][q, j])
            traces.append({"logprob": sig[0, a:b], "conf": sig[1, a:b], "ent": sig[2, a:b], "length": b - a,
                           "finish": ("stop", "length")[int(c["finish"][q, j])],
                           "label": labels.setdefault(v, f"a{len(labels)}") if v else None, "correct": bool(c["correct"][q, j])})
        out.append(traces)
    return out
