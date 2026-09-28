"""Reference conf_best (self-certainty best-of-n style): the baseline's reading rule (whole traces up to the fair
share), then the answer of the finished trace with the highest mean token confidence."""
from ttc_api import BudgetExhausted


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


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg

    def solve(self, problem):
        read_fair_share(problem)
        done = [t for t in problem.traces if t.done and t.answer is not None]
        if not done:
            return None
        return max(done, key=lambda t: (float(t.conf.mean()), -t.index)).answer
