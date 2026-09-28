"""Frozen EVALUATE for ttc-controller. Trusts nothing the controller did: it replays the harness's read log against
the split's cache and the private gold and recomputes every score.

Primary: accuracy at budget B = mean over problems of the problem's accuracy averaged over the replicates (items).
Invalid if: the settings differ from this command line; an episode is missing or its sandbox self-check failed or it
left an IPC object behind; a problem order or event is not what the seeded replay allows; a charge differs from
min(ceil32(n), tokens left in the trace, pool left) or a read happened with the pool empty; the charged total
differs from the sum of reads or exceeds the pool; the answer count is wrong; or a returned label was never
produced by a trace read to its end in that problem and episode.
Secondary: acc_half / acc_double (0.5 B and 2 B), tokens per problem, pool fraction used, abstain rate, traces
opened/finished per problem (all at B), and the split's pass@1 and majority over all traces for context.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from replay import LEVELS, charge, problem_order, trace_perm  # frozen/run, on PYTHONPATH

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--budget-tokens", type=int, required=True)
ap.add_argument("--replicates", type=int, required=True)
ap.add_argument("--data", default="/data", help="reads <data>/public/traces.npz and <data>/private/gold.json")
ap.add_argument("--limit", type=int, default=0, help="tests/pilot only: must match the RUN's --limit")
a = ap.parse_args()


def write(obj):
    json.dump(obj, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": str(msg)[:1000]})
    sys.exit(0)


z = np.load(f"{a.data}/public/traces.npz")
gold_all = json.load(open(f"{a.data}/private/gold.json"))
n_all = len(z["pids"])
n = min(a.limit, n_all) if a.limit else n_all
pids, lengths, answers = z["pids"][:n].astype(str), z["lengths"][:n].astype(int), z["answers"][:n].astype(str)
T = lengths.shape[1]
gold = [gold_all[p] for p in pids]
try:
    log = json.load(open(f"{a.run}/log.json"))
except Exception as e:
    invalid(f"missing or unreadable log.json: {e!r}")


def replay_episode(ep, level, r):
    """Returns per-problem (score, tokens, opened, finished, abstained) keyed by pid; raises ValueError if invalid."""
    budget = round(level * a.budget_tokens)
    if ep.get("budget") != budget or ep.get("pool") != budget * n:
        raise ValueError(f"episode {level}/{r}: budget or pool differs from --budget-tokens")
    if not isinstance(ep.get("sandbox"), dict) or ep["sandbox"].get("ok") is not True or ep.get("ipc_clean") is not True:
        raise ValueError(f"episode {level}/{r}: sandbox self-check failed or IPC state left behind ({ep.get('sandbox')})")
    probs = ep.get("problems")
    order = problem_order(log["seed"], r, n)
    if not isinstance(probs, list) or [p.get("pid") for p in probs] != [pids[q] for q in order]:
        raise ValueError(f"episode {level}/{r}: wrong answer count or problem order")
    pool_left, out = ep["pool"], {}
    for q, p in zip(order, probs):
        perm, pos, val2lab = trace_perm(log["seed"], r, pids[q], T), [], {}
        spent = 0
        for ev in p.get("events", []):
            if ev == ["o"] and len(pos) < T:
                pos.append(0)
                continue
            if not (isinstance(ev, list) and len(ev) == 4 and ev[0] == "r" and all(type(x) is int for x in ev[1:])):
                raise ValueError(f"{pids[q]}: malformed event {ev}")
            _, t, req, k = ev
            if not 0 <= t < len(pos) or req < 1:
                raise ValueError(f"{pids[q]}: read of an unopened trace or bad size {ev}")
            L = lengths[q, perm[t]]
            if pos[t] >= L or pool_left <= 0 or k != charge(req, L - pos[t], pool_left):
                raise ValueError(f"{pids[q]}: charge {k} differs from the replay (or a read past the end/pool) {ev}")
            pos[t] += k
            pool_left -= k
            spent += k
            if pos[t] == L and answers[q, perm[t]]:
                val2lab.setdefault(answers[q, perm[t]], f"a{len(val2lab)}")   # opaque labels in first-reveal order
        labels = {lab: v for v, lab in val2lab.items()}
        lab = p.get("answer")
        if lab is not None and lab not in labels:
            raise ValueError(f"{pids[q]}: label {lab!r} was never produced by a trace read to its end")
        finished = sum(pos[t] == lengths[q, perm[t]] for t in range(len(pos)))
        out[pids[q]] = (float(lab is not None and labels[lab] == gold[q]), spent, len(pos), finished, lab is None)
    if ep.get("charged") != ep["pool"] - pool_left or pool_left < 0:
        raise ValueError(f"episode {level}/{r}: charged {ep.get('charged')} differs from the sum of reads {ep['pool'] - pool_left}")
    return out


try:
    if (log.get("budget_tokens"), log.get("replicates"), log.get("limit"), log.get("n_problems"), log.get("levels")) != \
            (a.budget_tokens, a.replicates, a.limit, n, list(LEVELS)):
        invalid("run settings (budget, replicates, limit, problems, levels) differ from the evaluation command")
    eps = {(e.get("level"), e.get("replicate")): e for e in log.get("episodes", [])}
    want = [(lv, r) for lv in LEVELS for r in range(a.replicates)]
    if sorted(eps, key=str) != sorted(want, key=str) or len(log["episodes"]) != len(want):
        invalid(f"expected {len(want)} episodes (levels x replicates), got {len(log.get('episodes', []))}")
    res = {k: replay_episode(eps[k], *k) for k in want}
except ValueError as e:
    invalid(e)
except (KeyError, TypeError, AttributeError) as e:
    invalid(f"malformed log: {e!r}")

acc = {}
for lv in LEVELS:
    acc[lv] = {p: float(np.mean([res[lv, r][p][0] for r in range(a.replicates)])) for p in pids}
main = [res[LEVELS[0], r] for r in range(a.replicates)]
col = lambda i: float(np.mean([v[i] for ep in main for v in ep.values()]))  # noqa: E731
maj = [Counter(x for x in row if x).most_common(1) for row in answers]
metrics = {
    "accuracy": float(np.mean(list(acc[1.0].values()))),
    "acc_half": float(np.mean(list(acc[0.5].values()))), "acc_double": float(np.mean(list(acc[2.0].values()))),
    "tokens_per_problem": col(1), "pool_used": col(1) / a.budget_tokens, "abstain_rate": col(4),
    "traces_opened": col(2), "traces_finished": col(3),
    "fit_s_max": max(float(e.get("fit_s") or 0) for e in log["episodes"]),
    "pass1_split": float(np.mean(answers == np.array(gold)[:, None])),
    "maj_all_split": float(np.mean([bool(m) and m[0][0] == g for m, g in zip(maj, gold)])),
}
write({"valid": True, "primary": metrics["accuracy"], "metrics": metrics, "items": acc[1.0],
       "message": f"{n} problems x {a.replicates} replicates at B={a.budget_tokens}: accuracy {metrics['accuracy']:.4f}"})
print(json.dumps(metrics))
