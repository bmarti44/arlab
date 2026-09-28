"""Deterministic CPU checks for ttc-controller (run by `arlab check` as root, no GPU, no network; the whole data dir at
/data, the surface at /work, the sealed pack at /pack, frozen/run at /frozen, frozen/eval at /eval).
They pass on the synthetic cache (build/fake_cache.py) and on the real one."""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path[:0] = ["/pack/frozen/prepare", "/frozen"]
import answers  # noqa: E402
import gsm8k  # noqa: E402
from replay import LEVELS, Episode, load_cache, problem_order, trace_perm  # noqa: E402
from ttc_api import BudgetExhausted, Problem  # noqa: E402

D = "/data"
PY = sys.executable
PACK = open("/pack/pack.yaml").read()
B = int(re.search(r"harness\.py .*--budget-tokens (\d+)", PACK).group(1))
REFS = ("conf_best", "conf_vote", "asc")


def val(limit=0):
    return load_cache(f"{D}/validation/public/traces.npz", limit)


def surface():
    sys.path.insert(0, "/work")
    import controller
    return controller


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, "PYTHONPATH": "/frozen"}, **kw)


def harness(out, seed=1, work="/work", R=2, limit=30, user=None):
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    for d in ("public", "private"):
        if not os.path.lexists(f"{out}/data/{d}"):
            os.symlink(f"{D}/validation/{d}", f"{out}/data/{d}")
    if not os.path.lexists(f"{out}/data/train"):
        os.symlink(f"{D}/train", f"{out}/data/train")
    r = run([PY, "/frozen/harness.py", "--out", out, "--seed", str(seed), "--split", "validation", "--budget-tokens", str(B),
             "--replicates", str(R), "--data", f"{out}/data", "--work", work, "--limit", str(limit)],
            **({"user": user, "group": user, "extra_groups": []} if user is not None else {}))
    return r


def evaluate(out, R=2, limit=30, data=None):
    r = run([PY, "/eval/evaluate.py", "--run", out, "--out", f"{out}/metrics.json", "--budget-tokens", str(B),
             "--replicates", str(R), "--limit", str(limit), "--data", data or f"{out}/data"])
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(f"{out}/metrics.json"))


# ---------------------------------------------------------------- scorer: extraction and normalization
@pytest.mark.parametrize("text,finish,want", [
    ("so \\boxed{1,000}.", "stop", "1000"), ("\\boxed{12.0}", "stop", "12"), ("\\boxed{ $18 }", "stop", "18"),
    ("\\boxed{\\$1,234.50}", "stop", "1234.5"), ("\\boxed{007}", "stop", "7"), ("\\boxed{-0}", "stop", "0"),
    ("the answer is 42", "stop", None), ("\\boxed{3} then \\boxed{4}", "stop", "4"), ("\\boxed{5}", "length", None),
    ("\\boxed{\\frac{1}{2}}", "stop", "\\frac{1}{2}"), ("\\boxed{12 \\text{ apples}}", "stop", "12\\text{apples}"),
    ("\\boxed{}", "stop", None), ("\\boxed{7", "stop", None), ("\\boxed{2} and \\boxed{9", "stop", "2")])
def test_extract_and_normalize(text, finish, want):
    assert answers.extract(text, finish) == want


def test_gold_normalization_and_near_duplicates():
    assert answers.normalize(" 1,000") == "1000" and answers.normalize("-3") == "-3"
    test = [{"pid": "test-0", "question": "Tom has 3 apples and buys 5 more apples at the store today. How many now?"}]
    pool = [{"pid": "train-0", "question": "Tom has 3 apples and buys 5 more apples at the store today. How many now?!"},
            {"pid": "train-1", "question": "A train leaves at 5 pm and travels 60 miles per hour for 3 hours; how far?"}]
    assert gsm8k.near_duplicates(pool, test) == {"train-0"}


# ---------------------------------------------------------------- data: splits, no gold in RUN's view
def test_splits_are_the_frozen_gsm8k_split_and_disjoint():
    sp, ref = json.load(open(f"{D}/splits.json")), gsm8k.splits("/hf")
    got = {s: list(load_cache(f"{D}/{p}/traces.npz")["pids"]) for s, p in
           (("train", "train"), ("validation", "validation/public"), ("holdout", "holdout/public"))}
    assert sp == {"validation": len(got["validation"]), "holdout": len(got["holdout"])}
    for s, pids in got.items():
        assert pids == [p["pid"] for p in ref[s][:len(pids)]] and len(set(pids)) == len(pids), s
    assert all(p.startswith("test-") for p in got["holdout"])
    assert all(p.startswith("train-") for p in got["train"] + got["validation"])
    assert not set(got["train"]) & set(got["validation"])
    q = {p["pid"]: p for s in ("train", "validation", "holdout") for p in ref[s]}
    assert not gsm8k.near_duplicates([q[p] for p in got["train"] + got["validation"]], ref["holdout"])
    info = json.load(open(f"{D}/info.json"))
    if not info["manifest"]["synthetic"]:
        assert len(got["holdout"]) == 1319


def test_run_view_holds_no_gold():
    for s in ("validation", "holdout"):
        assert sorted(os.listdir(f"{D}/{s}/public")) == ["traces.npz"]
        z = np.load(f"{D}/{s}/public/traces.npz")
        assert set(z.files) == {"pids", "lengths", "finish", "answers", "offsets", "logprob", "conf", "ent"}
        assert set(json.load(open(f"{D}/{s}/private/gold.json"))) == set(z["pids"].astype(str))
    t = np.load(f"{D}/train/traces.npz")
    assert "correct" in t.files and not {"gold", "question", "text"} & set(t.files)


# ---------------------------------------------------------------- replay: orders, budget, labels
def test_orders_depend_only_on_seed_replicate_problem():
    c = val()
    n, pid = c["n"], str(c["pids"][3])
    assert (problem_order(1, 0, n) == problem_order(1, 0, n)).all()
    assert not (problem_order(1, 0, n) == problem_order(2, 0, n)).all()
    assert not (problem_order(1, 0, n) == problem_order(1, 1, n)).all()
    p = trace_perm(1, 0, pid, 32)
    assert (p == trace_perm(1, 0, pid, 32)).all() and sorted(p) == list(range(32))
    assert not (p == trace_perm(2, 0, pid, 32)).all() and not (p == trace_perm(1, 1, pid, 32)).all()
    assert not (p == trace_perm(1, 0, str(c["pids"][4]), 32)).all()
    # the trace order of a problem does not depend on which other problems are in the split
    e1, e2 = Episode(c, 1, 0, B), Episode(val(limit=10), 1, 0, B)
    i1 = e1.begin(list(e1.order).index(3))
    i2 = e2.begin(list(e2.order).index(3))
    assert e1.cur["pid"] == e2.cur["pid"] == pid and (e1.cur["perm"] == e2.cur["perm"]).all() and i1["max_traces"] == 32


def test_reads_are_charged_exactly_and_refused_past_the_pool():
    c = val(limit=2)
    for seed in range(50):                          # a first trace longer than the pool
        ep = Episode(c, seed, 0, 40)                # pool = 2 x 40 = 80 tokens
        prob = Problem(ep.handle, ep.begin(0))
        if int(c["lengths"][ep.cur["q"], ep.cur["perm"][0]]) > 80:
            break
    t = prob.open()
    assert t.read(1) == 32 and t.pos == 32 and len(t.conf) == 32 and prob.pool_left == 48
    assert t.read(100) == 48 and prob.pool_left == 0 and not t.done and t.answer is None
    with pytest.raises(BudgetExhausted):
        t.read(1)
    with pytest.raises(BudgetExhausted):
        prob.open().read()
    assert ep.charged == ep.pool == 80
    assert ep.cur["events"] == [["o"], ["r", 0, 1, 32], ["r", 0, 100, 48], ["o"]]


def test_labels_are_opaque_in_first_reveal_order():
    c = val(limit=5)
    ep = Episode(c, 3, 1, 10 ** 6)
    for i in range(c["n"]):
        prob = Problem(ep.handle, ep.begin(i))
        while (t := prob.open()) is not None:
            t.read()
            assert t.done and t.length == len(t.conf) == t.pos and t.finish in ("stop", "length")
        assert len(prob.traces) == 32 and prob.open() is None
        vals = [str(c["answers"][ep.cur["q"], j]) for j in ep.cur["perm"]]
        order = list(dict.fromkeys(v for v in vals if v))
        assert [t.answer for t in prob.traces] == [f"a{order.index(v)}" if v else None for v in vals]
        assert prob.traces[0].read() == 0
        assert prob._rpc({"op": "read", "t": 0, "n": 1}) == {"ok": False, "err": "trace already read to its end"}
        assert prob._rpc({"op": "read", "t": 32, "n": 1})["ok"] is False
        ep.finish(None)


@pytest.mark.parametrize("level", LEVELS)
def test_baseline_spend_within_pool(level):
    c, ctl = val(), surface()
    ep = Episode(c, 1, 0, round(level * B))
    ctrl = ctl.Controller({"budget": ep.budget, "n_problems": c["n"], "pool": ep.pool, "max_traces": 32, "seed": 0})
    opened = []
    for i in range(c["n"]):
        prob = Problem(ep.handle, ep.begin(i))
        ep.finish(ctrl.solve(prob))
        opened.append(len(prob.traces))
    assert 0 <= ep.charged <= ep.pool
    mean_len = float(c["lengths"].mean())
    assert abs(np.mean(opened) - level * B / mean_len) < 1.5, (np.mean(opened), level * B / mean_len)


# ---------------------------------------------------------------- sandbox
def test_sandboxed_child_cannot_open_data():
    f = f"{D}/validation/public/traces.npz"
    code = ("import sys; sys.path.insert(0, '/frozen'); import child; open(sys.argv[1], 'rb').close(); "
            "child.lockdown(child.READ_DIRS + ['/work']); import numpy; open('/work/controller.py').close(); "
            "open(sys.argv[1], 'rb')")
    r = subprocess.run([PY, "-I", "-c", code, f], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "PermissionError" in r.stderr, r.stderr[-1000:]


SPY = r'''
import os, re
LEAKS = ["/data/validation/public/traces.npz", "/data/validation/private/gold.json", "/data/train/traces.npz",
         "/pack/pack.yaml", "/eval/evaluate.py", "/frozen/replay.py", "/proc/%d/mem" % os.getppid(), "/proc/self/environ"]
for p in LEAKS:
    try:
        open(p, "rb").close()
    except OSError:
        continue
    raise RuntimeError("LEAK " + p)
for p in ("/tmp/x", "/work/x"):
    try:
        open(p, "w").close()
    except OSError:
        continue
    raise RuntimeError("WRITE " + p)
if os.getuid() != 0:                 # RLIMIT_NPROC does not bind root (TESTS); RUN is uid 1000
    try:
        os.fork()
        raise RuntimeError("FORK")
    except BlockingIOError:
        pass

class Controller:
    def __init__(self, cfg):
        import sys
        assert set(cfg) == {"budget", "n_problems", "pool", "max_traces", "seed"}
        assert sys.flags.hash_randomization == 0 and not os.environ.get("PYTHONPATH")
    def fit(self, train):
        for traces in train:
            assert len(traces) == 32 and all(t["label"] is None or re.fullmatch(r"a\d+", t["label"]) for t in traces)
            assert all(isinstance(t["correct"], bool) and len(t["conf"]) == t["length"] for t in traces)
    def solve(self, problem):
        assert not {"pid", "question", "gold", "answers"} & set(vars(problem))
        t = problem.open()
        t.read()
        assert t.answer is None or re.fullmatch(r"a\d+", t.answer)
        return t.answer
'''


def test_spy_controller_sees_only_opaque_labels_and_no_files():
    """As in RUN: the harness runs as uid 1000, so the no-fork limit applies too."""
    base = tempfile.mkdtemp(prefix="ttc-spy-", dir="/tmp")
    work, out = f"{base}/spy", f"{base}/out"
    os.makedirs(work)
    open(f"{work}/controller.py", "w").write(SPY)
    os.makedirs(out)
    for p in (base, work, out):
        os.chmod(p, 0o777)
    r = harness(out, work=work, R=1, limit=10, user=1000)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    m = evaluate(out, R=1, limit=10)
    assert m["valid"], m["message"]
    log = json.load(open(f"{out}/log.json"))
    assert all(e["sandbox"]["ok"] and e["sandbox"]["landlock_abi"] >= 4 for e in log["episodes"])


# ---------------------------------------------------------------- harness + evaluator end to end
def _strip(log):
    for e in log["episodes"]:
        for k in ("fit_s", "wall_s"):
            e.pop(k, None)
        for p in e["problems"]:
            p.pop("solve_s")
    log.pop("wall_s")
    return log


def test_harness_and_evaluator_are_deterministic_per_seed(tmp_path):
    logs = {}
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        out = str(tmp_path / name)
        r = harness(out, seed=seed)
        assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
        m = evaluate(out)
        assert m["valid"], m["message"]
        assert len(m["items"]) == 30 and 0 <= m["primary"] <= 1 and m["metrics"]["pool_used"] <= 1.0
        logs[name] = (_strip(json.load(open(f"{out}/log.json"))), m["items"])
        assert json.load(open(f"{out}/budget.json"))["budget_fraction"] <= 1.0
    assert logs["a"] == logs["b"]
    assert logs["a"][0]["episodes"] != logs["c"][0]["episodes"]


@pytest.mark.parametrize("ref", REFS)
def test_references_run_and_score(tmp_path, ref):
    out = str(tmp_path / ref)
    r = harness(out, work=f"/frozen/ref_{ref}", R=1, limit=30)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert evaluate(out, R=1)["valid"]


def _forge(tmp_path, name, fn):
    src = tmp_path / "valid"
    if not (src / "log.json").exists():
        assert harness(str(src), R=1, limit=20).returncode == 0
    out = tmp_path / name
    out.mkdir()
    log = json.load(open(src / "log.json"))
    fn(log)
    json.dump(log, open(out / "log.json", "w"))
    return evaluate(str(out), R=1, limit=20, data=str(src / "data"))


def _first_read(log):
    return next(ev for p in log["episodes"][0]["problems"] for ev in p["events"] if ev[0] == "r")


@pytest.mark.parametrize("name,forge", [
    ("undercharged", lambda log: _first_read(log).__setitem__(3, _first_read(log)[3] - 1)),
    ("unknown_label", lambda log: log["episodes"][0]["problems"][0].__setitem__("answer", "a17")),
    ("over_pool", lambda log: log["episodes"][0].__setitem__("charged", log["episodes"][0]["pool"] + 1)),
    ("sandbox", lambda log: log["episodes"][1]["sandbox"].__setitem__("ok", False)),
    ("order", lambda log: log["episodes"][0]["problems"].reverse()),
    ("missing", lambda log: log["episodes"].pop()),
    ("budget", lambda log: log.__setitem__("budget_tokens", B + 50)),
])
def test_evaluator_rejects_forged_logs(tmp_path, name, forge):
    m = _forge(tmp_path, name, forge)
    assert m["valid"] is False and m["primary"] is None, name


def test_label_from_an_unfinished_trace_is_rejected(tmp_path):
    """A controller that reads 32 tokens of one trace and returns 'a0' (which no finished trace produced)."""
    def script(level, pick_unfinished):
        ep = Episode(c, 1, 0, round(level * B))
        for i in range(c["n"]):
            prob = Problem(ep.handle, ep.begin(i))
            t = prob.open()
            t.read(32 if pick_unfinished and i == 0 else None)
            ep.finish("a0" if pick_unfinished and i == 0 else t.answer)
        return {"level": level, "replicate": 0, "budget": ep.budget, "pool": ep.pool, "charged": ep.charged,
                "sandbox": {"ok": True}, "ipc_clean": True, "fit_s": 0, "problems": ep.problems}
    c = val(limit=8)
    for bad in (False, True):
        out = tmp_path / f"s{int(bad)}"
        out.mkdir()
        log = {"seed": 1, "split": "validation", "budget_tokens": B, "replicates": 1, "levels": list(LEVELS), "limit": 8,
               "n_problems": 8, "n_traces": 32, "episodes": [script(lv, bad and lv == 1.0) for lv in LEVELS]}
        json.dump(log, open(out / "log.json", "w"))
        m = evaluate(str(out), R=1, limit=8, data=f"{D}/validation")
        assert m["valid"] is (not bad), m["message"]
        if bad:
            assert "never produced" in m["message"]


def test_scorer_on_a_hand_made_split(tmp_path):
    """2 problems x 3 traces with known answers and gold; every trace read fully; hand-picked answers."""
    d = tmp_path / "hand"
    (d / "public").mkdir(parents=True)
    (d / "private").mkdir()
    lengths = np.array([[40, 70, 33], [50, 20, 64]])
    ans = np.array([["5", "7", ""], ["3", "4", "3"]])
    tot = int(lengths.sum())
    z = {"pids": np.array(["train-0001", "train-0002"]), "lengths": lengths.astype(np.int32), "answers": ans,
         "finish": np.array([[0, 0, 1], [0, 0, 0]], np.uint8), "offsets": np.concatenate([[0], np.cumsum(lengths.ravel())]),
         "logprob": np.zeros(tot, np.float16), "conf": np.ones(tot, np.float16), "ent": np.zeros(tot, np.float16)}
    np.savez(d / "public" / "traces.npz", **z)
    json.dump({"train-0001": "7", "train-0002": "3"}, open(d / "private" / "gold.json", "w"))
    c = load_cache(str(d / "public" / "traces.npz"))
    pick = {"train-0001": "5", "train-0002": "3"}         # expected scores 0 and 1 at every level
    eps = []
    for lv in LEVELS:
        ep = Episode(c, 7, 0, round(lv * B))
        for i in range(2):
            prob = Problem(ep.handle, ep.begin(i))
            while (t := prob.open()) is not None:
                assert t.read() == t.length
            vals = [str(ans[ep.cur["q"], j]) for j in ep.cur["perm"]]
            order = list(dict.fromkeys(v for v in vals if v))
            ep.finish(f"a{order.index(pick[ep.cur['pid']])}")
        eps.append({"level": lv, "replicate": 0, "budget": ep.budget, "pool": ep.pool, "charged": ep.charged,
                    "sandbox": {"ok": True}, "ipc_clean": True, "fit_s": 0, "problems": ep.problems})
    out = tmp_path / "run"
    out.mkdir()
    json.dump({"seed": 7, "split": "validation", "budget_tokens": B, "replicates": 1, "levels": list(LEVELS), "limit": 0,
               "n_problems": 2, "n_traces": 3, "episodes": eps}, open(out / "log.json", "w"))
    m = evaluate(str(out), R=1, limit=0, data=str(d))
    assert m["valid"], m["message"]
    assert m["items"] == {"train-0001": 0.0, "train-0002": 1.0} and m["primary"] == 0.5
    assert m["metrics"]["acc_half"] == m["metrics"]["acc_double"] == 0.5
    assert m["metrics"]["tokens_per_problem"] == tot / 2 and m["metrics"]["traces_finished"] == 3
    assert m["metrics"]["pass1_split"] == 3 / 6 and m["metrics"]["maj_all_split"] == 0.5
