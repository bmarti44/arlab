"""Controller-side view of a problem (see api.md). Every call goes through rpc(msg) -> reply to the harness, which
owns the cache, the budget and the log; these objects only hold what has been revealed."""
import numpy as np

from replay import decode


class BudgetExhausted(Exception):
    """The pool is empty: nothing more can be read in this episode."""


class Trace:
    def __init__(self, problem, index: int):
        self._p, self.index = problem, index
        self.pos, self.done = 0, False
        self.length = self.finish = self.answer = None
        self._chunks = []
        self._sig = np.zeros((3, 0), np.float32)

    def read(self, n: int | None = None) -> int:
        """Reveal up to n more tokens (rounded up to a multiple of 32; None = to the end). Returns tokens revealed.
        The pool may cut a read short; with the pool empty it raises BudgetExhausted."""
        if self.done:
            return 0
        rep = self._p._rpc({"op": "read", "t": self.index, "n": int(n) if n is not None else 1 << 30})
        if not rep["ok"]:
            raise BudgetExhausted() if rep["err"] == "budget" else RuntimeError(rep["err"])
        k = rep["k"]
        self._chunks.append(decode(rep["sig"], k))
        self._sig = None
        self.pos += k
        self._p.pool_left, self._p.spent = rep["pool_left"], self._p.spent + k
        if rep["done"]:
            self.done, self.length, self.finish, self.answer = True, rep["length"], rep["finish"], rep["label"]
        return k

    def _signals(self) -> np.ndarray:
        if self._sig is None:
            self._sig = np.concatenate(self._chunks, axis=1)
        return self._sig

    @property
    def logprob(self) -> np.ndarray:
        return self._signals()[0]

    @property
    def conf(self) -> np.ndarray:
        return self._signals()[1]

    @property
    def ent(self) -> np.ndarray:
        return self._signals()[2]


class Problem:
    def __init__(self, rpc, info: dict):
        self._rpc = rpc
        self.index, self.problems_left = info["index"], info["problems_left"]
        self.pool_left, self.budget, self.max_traces = info["pool_left"], info["budget"], info["max_traces"]
        self.spent, self.traces = 0, []

    def open(self) -> Trace | None:
        """The next trace of this problem's fixed order, or None when all max_traces are open. Opening is free."""
        rep = self._rpc({"op": "open"})
        if not rep["ok"]:
            return None
        t = Trace(self, rep["t"])
        self.traces.append(t)
        return t
