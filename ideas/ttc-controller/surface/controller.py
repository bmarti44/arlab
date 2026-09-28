"""Baseline controller: fixed-budget majority vote ("majority@k at the same budget", k set by the budget).

For each problem take the fair share f = pool_left / problems_left, read whole traces in order until the spend on
this problem reaches f (the last trace may go over; later problems absorb the difference), then return the plurality
label of the traces read to their end, ties broken by first reveal. The API is in frozen/run/api.md.
"""
from collections import Counter

from ttc_api import BudgetExhausted


class Controller:
    def __init__(self, cfg: dict):
        self.cfg = cfg          # budget, n_problems, pool, max_traces, seed

    def fit(self, train: list) -> None:
        """train: list of problems, each a list of trace dicts with full signals, 'label' and 'correct'. Unused."""

    def solve(self, problem):
        share = problem.pool_left / problem.problems_left
        while problem.spent < share:
            t = problem.open()
            if t is None:
                break
            try:
                t.read()                                     # to the end (or until the pool runs out)
            except BudgetExhausted:
                break
        votes, first = Counter(), {}
        for i, t in enumerate(problem.traces):
            if t.done and t.answer is not None:
                votes[t.answer] += 1
                first.setdefault(t.answer, i)
        if not votes:
            return None
        return max(votes, key=lambda a: (votes[a], -first[a]))
