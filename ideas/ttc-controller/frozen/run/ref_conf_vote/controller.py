"""Reference conf_vote (DeepConf-offline): the baseline's reading rule, then keep the top 50 % of finished traces by
lowest 64-token group confidence (the minimum over sliding 64-token windows of the mean token confidence) and take
a vote weighted by that confidence; ties broken by first reveal."""
import numpy as np

from ttc_api import BudgetExhausted

WINDOW, KEEP = 64, 0.5


def read_fair_share(problem):
    share = problem.pool_left / problem.problems_left
    while problem.spent < share:
        t = problem.open()
        if t is None:
            return
        try:
            t.read()
        except BudgetExhausted:
            return


def lowest_group_conf(conf):
    if len(conf) <= WINDOW:
        return float(conf.mean())
    c = np.concatenate([[0.0], np.cumsum(conf, dtype=np.float64)])
    return float(((c[WINDOW:] - c[:-WINDOW]) / WINDOW).min())


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg

    def solve(self, problem):
        read_fair_share(problem)
        done = [(lowest_group_conf(t.conf), t.index, t.answer) for t in problem.traces if t.done and t.answer is not None]
        if not done:
            return None
        kept = sorted(done, key=lambda x: (-x[0], x[1]))[:max(1, int(np.ceil(KEEP * len(done))))]
        score, first = {}, {}
        for g, i, a in kept:
            score[a] = score.get(a, 0.0) + g
            first[a] = min(first.get(a, i), i)
        return max(score, key=lambda a: (score[a], -first[a]))
