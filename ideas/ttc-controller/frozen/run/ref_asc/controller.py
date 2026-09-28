"""Reference asc (Adaptive-Consistency, Beta stopping rule): read whole traces one at a time and stop once
P(p1 > p2 | votes) >= threshold under a uniform Beta prior on the top two answers' shares, or once the spend reaches
3x the fair share. fit() sets the threshold on the train problems: the largest one whose mean spend there is <= the
budget (cfg["budget"]), so the pool rarely runs dry. Answer: plurality vote, ties broken by first reveal."""
from collections import Counter
from math import comb

from ttc_api import BudgetExhausted

CAP = 3.0
GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.01]


def p_top(labels):
    """P(p1 > p2) with v1, v2 the two largest vote counts = P(Binomial(v1 + v2 + 1, 1/2) <= v1)."""
    c = sorted(Counter(x for x in labels if x is not None).values(), reverse=True) + [0, 0]
    v1, v2 = c[0], c[1]
    n = v1 + v2 + 1
    return sum(comb(n, i) for i in range(v1 + 1)) / 2 ** n if v1 else 0.0


def vote(pairs):
    votes, first = Counter(), {}
    for i, a in pairs:
        if a is not None:
            votes[a] += 1
            first.setdefault(a, i)
    return max(votes, key=lambda a: (votes[a], -first[a])) if votes else None


class Controller:
    def __init__(self, cfg):
        self.cfg, self.thr = cfg, 0.95

    def fit(self, train):
        B = self.cfg["budget"]
        for thr in GRID:
            spend = 0
            for traces in train:
                s, labels = 0, []
                for t in traces:
                    if s >= CAP * B:
                        break
                    s += t["length"]
                    labels.append(t["label"])
                    if p_top(labels) >= thr:
                        break
                spend += s
            if train and spend / len(train) <= B:
                self.thr = thr

    def solve(self, problem):
        cap = CAP * problem.pool_left / problem.problems_left
        while problem.spent < cap:
            t = problem.open()
            if t is None:
                break
            try:
                t.read()
            except BudgetExhausted:
                break
            if p_top([x.answer for x in problem.traces if x.done]) >= self.thr:
                break
        return vote([(t.index, t.answer) for t in problem.traces if t.done])
