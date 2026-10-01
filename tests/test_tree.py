"""Discovery-tree search: replay semantics (no leaks), objective, policies, dreaming guard; docker: sandbox + a fixture tree run."""
import json
import subprocess
from pathlib import Path

import pytest

from arlab.agent import Proposal
from arlab.tree import dream as dream_mod
from arlab.tree.model import Tree, clean_selection, frontier, public_view
from arlab.tree.policy import BUILTIN, LocalPolicy, PolicyError, SandboxPolicy
from arlab.tree.replay import Replay, objective, replay
from conftest import ROOT, make_variant, run_arlab

SCRIPTS = ROOT / "tests" / "scripts"


def node(i, parent, score, status="ok", order=None, depth=1, tag="t"):
    return {"id": i, "parent": parent, "depth": depth, "order": order, "status": status, "score": score, "tag": tag, "cost_s": 1.0}


def recorded():
    """root ─ a(0.1) ─ a1(0.5) ─ a11(3.0)      recorded order: a, b, a1, c, a11
            ├ b(crash)
            └ c(0.2)"""
    root = {"id": "root", "parent": None, "depth": 0, "order": 0, "status": "ok", "score": 0.0, "tag": "root", "cost_s": 0}
    return [root, node("a", "root", 0.1, order=1), node("b", "root", None, "crash", order=2), node("a1", "a", 0.5, order=3, depth=2),
            node("c", "root", 0.2, order=4), node("a11", "a1", 3.0, order=5, depth=3)]


class Scripted:
    """A policy that replays a fixed list of selections and records every view it was shown."""

    def __init__(self, picks):
        self.picks, self.seen = list(picks), []

    def select(self, nodes, budget_left, W):
        self.seen.append(json.loads(json.dumps(nodes)))
        return self.picks.pop(0) if self.picks else []


# ------------------------------------------------------------------ replay semantics
def test_root_reveals_children_in_recorded_order():
    sim = Replay(recorded())
    got = [sim.back[n["id"]] for n in sim.expand(["root", "root", "root"])]
    assert got == ["a", "b", "c"]
    assert sim.expand(["root"]) == []                          # root exhausted: nothing, no invention


def test_leaf_reveals_its_recorded_child_and_exhaustion():
    sim = Replay(recorded())
    a = sim.expand(["root"])[0]["id"]
    a1 = sim.expand([a])[0]
    assert sim.back[a1["id"]] == "a1" and a1["parent"] == a and a1["score"] == 0.5
    assert sim.expand([a]) == []                               # a has one recorded child only


def test_no_leak_of_unrevealed_nodes():
    """open-dream-rsi regression: a policy must never see an unreached node, its score, or recorded ids/orders."""
    pol = Scripted([["root"], ["root"]])
    res = replay(pol, recorded(), W=1, budget=9)
    for view in pol.seen:
        ids = {n["id"] for n in view}
        assert ids <= {"root"} | {f"r{i:04d}" for i in range(1, 6)}          # relabelled, no recorded ids
        assert 3.0 not in [n["score"] for n in view] and 0.5 not in [n["score"] for n in view]
        assert all(set(n) == {"id", "parent", "depth", "order", "status", "score", "cost_s", "children", "leaf"} for n in view)
        assert [n["order"] for n in view] == list(range(len(view)))           # reveal order, not recorded order
    assert res["best"] == pytest.approx(0.1) and res["N"] == 2                # the grandchild's 3.0 is never credited
    assert res["revealed"] == ["a", "b"]


def test_deep_chain_is_credited_only_when_reached():
    pol = Scripted([["root"], ["r0001"], ["r0002"]])
    res = replay(pol, recorded(), W=1, budget=5)
    assert res["revealed"] == ["a", "a1", "a11"] and res["best"] == pytest.approx(3.0)
    assert res["curve"] == [0.1, 0.5, 3.0]


def test_invalid_ids_and_width_are_clamped():
    nodes = recorded()
    sel, bad = clean_selection(["a", "zzz", "root", "c", "c"], nodes, W=2, budget_left=5)
    assert frontier(nodes) == {"root", "b", "c", "a11"}
    assert sel == ["root", "c"] and bad == 2                  # 'a' is not a leaf, 'zzz' does not exist
    assert clean_selection("root", nodes, 2, 5) == ([], 1)


def test_stall_ends_replay_and_objective():
    pol = Scripted([["root"], ["root"], ["r0002"], ["r0002"], ["r0002"], ["root"]])   # r0002 = b: no recorded child
    res = replay(pol, recorded(), W=1, budget=5)
    assert res["stop"] == "stalled" and res["N"] == 2 and res["best"] == pytest.approx(0.1)
    assert objective(0.5, 2, 5, 0.5) == pytest.approx(0.3) and objective(-1, 2, 5, 0) == 0.0
    assert res["V"]["0.5"] == pytest.approx(0.1 - 0.5 * 2 / 5)


def test_gain_is_relative_to_root_score():
    rec = recorded()
    rec[0]["score"] = 0.4
    res = replay(Scripted([["root", "root", "root"]]), rec, W=3, budget=5)
    assert res["best"] == 0.0                                  # nothing beat the root


# ------------------------------------------------------------------ built-in policies (trusted, in-process)
def run_policy(name, W, budget):
    return replay(LocalPolicy(BUILTIN / f"{name}.py"), recorded(), W=W, budget=budget)


def test_builtins_are_valid_and_deterministic():
    for name in ("parallel_refine", "greedy_best_leaf", "ucb_chains"):
        a, b = run_policy(name, 2, 5), run_policy(name, 2, 5)
        assert a == b and a["invalid"] == 0, name
    assert run_policy("parallel_refine", 3, 5)["trace"][0] == ["root", "root", "root"]
    assert run_policy("greedy_best_leaf", 1, 5)["revealed"][:1] == ["a"]


def test_public_view_leaf_flags():
    v = {n["id"]: n for n in public_view(recorded())}
    assert v["root"]["leaf"] is False and v["b"]["leaf"] and not v["a"]["leaf"] and v["a"]["children"] == ["a1"]


# ------------------------------------------------------------------ persistence
def test_tree_roundtrip_and_ids(tmp_path):
    t = Tree(tmp_path / "tree.json", {"budget": 3})
    t.nodes.append(node("n0001", "root", 0.2, order=1))
    t.nodes.append(node("n0002", "n0001", None, "pending", order=2, depth=2))
    t.save()
    u = Tree.load(tmp_path / "tree.json")
    assert u.new_id() == "n0003" and u.used() == 1 and [n["id"] for n in u.path_to("n0002")] == ["root", "n0001", "n0002"]


# ------------------------------------------------------------------ dreaming: the held-out guard
class FakeCodex:
    """Writes a fixed sequence of policy files into the view, like a Codex edit."""
    model, effort = "fake", "none"

    def __init__(self, sources):
        self.sources = list(sources)

    def _call(self, view, out, name, model, effort):
        (view / "policy.py").write_text(self.sources.pop(0))
        return Proposal("edit", f"rev {name}", "x", None, "fake", {})


def _save_tree(path, nodes, W=1):
    t = Tree(path, {"pack": "p", "tag": path.stem, "W": W, "budget": len(nodes) - 1}, nodes)
    t.save()
    return str(path)


STOP_AT_ONE = "def select(nodes, budget_left, W):\n    return [] if len(nodes) > 1 else ['root']\n"
DIVE_FIRST = ("def select(nodes, budget_left, W):\n    leaves = [n for n in nodes if n['leaf']]\n"
              "    return [max(leaves, key=lambda n: n['order'])['id']] if leaves else ['root']\n")


def test_dream_accepts_only_if_heldout_does_not_drop(tmp_path, monkeypatch):
    monkeypatch.setenv("ARLAB_RUNS", str(tmp_path))
    train = _save_tree(tmp_path / "train.json", recorded())
    # held-out tree where diving is bad: the first child is a dead end with a costly chain of zeros
    ho_nodes = [recorded()[0], node("x", "root", 0.0, order=1), node("x1", "x", 0.0, order=2, depth=2),
                node("x2", "x1", 0.0, order=3, depth=3), node("y", "root", 0.0, order=4)]
    held = _save_tree(tmp_path / "held.json", ho_nodes)
    base = tmp_path / "stop_at_one.py"
    base.write_text(STOP_AT_ONE)
    rep = dream_mod.dream(str(base), [train], [held], "t1", M=1, backend=FakeCodex([DIVE_FIRST]), sandbox=LocalPolicy)
    assert rep["best_rev"] == 1 and rep["V_train"]["best"] > rep["V_train"]["current"]    # diving wins on training
    assert rep["V_heldout"]["best"] < rep["V_heldout"]["current"] and not rep["accepted"]  # but loses held-out → rejected
    with pytest.raises(ValueError, match="overlap"):
        dream_mod.dream(str(base), [train], [train], "t2", M=1, backend=FakeCodex([DIVE_FIRST]), sandbox=LocalPolicy)
    deep = [recorded()[0], node("z", "root", 0.0, order=1), node("z1", "z", 2.0, order=2, depth=2), node("w", "root", 0.0, order=3)]
    held2 = _save_tree(tmp_path / "held2.json", deep)     # a different tree where diving pays off
    rep2 = dream_mod.dream(str(base), [train], [held2], "t3", M=1, backend=FakeCodex([DIVE_FIRST]), sandbox=LocalPolicy)
    assert rep2["accepted"] and Path(rep2["policy_out"]).read_text() == DIVE_FIRST


def test_dream_ignores_symlinked_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ARLAB_RUNS", str(tmp_path))
    train = _save_tree(tmp_path / "train.json", recorded())
    base = tmp_path / "stop_at_one.py"
    base.write_text(STOP_AT_ONE)

    class LinkCodex(FakeCodex):
        def _call(self, view, out, name, model, effort):
            (view / "policy.py").unlink()
            (view / "policy.py").symlink_to("/etc/passwd")
            return Proposal("edit", "link", "x", None, "fake", {})
    rep = dream_mod.dream(str(base), [train], [], "t4", M=1, backend=LinkCodex([]), sandbox=LocalPolicy)
    assert (tmp_path / "_tree/dream/t4/policy-1.py").read_bytes() == b"" and not rep["accepted"]


# ------------------------------------------------------------------ docker: sandbox + a real fixture tree run
@pytest.mark.docker
def test_sandbox_isolation_and_timeout(tmp_path):
    spy = tmp_path / "spy.py"
    spy.write_text("import os, socket\n"
                   "def select(nodes, budget_left, W):\n"
                   "    print('noise on stdout')\n"
                   "    try:\n        socket.create_connection(('1.1.1.1', 53), timeout=2); net = 'yes'\n    except OSError:\n        net = 'no'\n"
                   "    try:\n        open('/p/x', 'w'); w = 'yes'\n    except OSError:\n        w = 'no'\n"
                   "    return ['root', net, w, str(os.path.exists(os.path.expanduser('~/arlab-runs')))]\n")
    with SandboxPolicy(spy) as p:
        assert p.select(public_view(recorded()), 5, 4) == ["root", "no", "no", "False"]
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ndef select(nodes, budget_left, W):\n    time.sleep(120)\n")
    with SandboxPolicy(slow, timeout_s=2) as p:
        p.warm = True
        with pytest.raises(PolicyError, match="timed out"):
            p.select([], 1, 1)
    junk = tmp_path / "junk.py"
    junk.write_text("import os\ndef select(nodes, budget_left, W):\n    os.write(1, b'garbage\\n')\n    return ['root']\n")
    with SandboxPolicy(junk) as p, pytest.raises(PolicyError, match="not JSON"):
        p.select([], 1, 1)
    notlist = tmp_path / "notlist.py"
    notlist.write_text("def select(nodes, budget_left, W):\n    return {'root': 1}\n")
    with SandboxPolicy(notlist) as p, pytest.raises(PolicyError, match="list of node ids"):
        p.select([], 1, 1)
    bad = tmp_path / "bad.py"
    bad.write_text("import nonexistent_module\n")
    with SandboxPolicy(bad) as p, pytest.raises(PolicyError, match="import failed"):
        p.select([], 1, 1)


def tree_state(runs, tag):
    d = runs / "_fixture" / tag
    return json.loads((d / "state.json").read_text()), Tree.load(d / "tree.json"), d


@pytest.mark.docker
def test_fixture_tree_run_kill_resume_finalize(runs_root, tmp_path):
    """accept-T1: W=2 × depth 3 with parallel_refine, kill -9 mid-batch, resume, top-k confirm, exactly one holdout."""
    pack = make_variant(tmp_path, "t1")
    args = ["tree", "run", pack, "--tag", "t1", "--policy", "parallel_refine", "--budget", "6", "--workers", "2",
            "--script", SCRIPTS / "tree.yaml"]
    r = run_arlab(runs_root, *args, env_extra={"ARLAB_DEBUG_KILL": "after_pending:2"})
    assert r.returncode != 0                                     # killed
    r = run_arlab(runs_root, *args)
    assert r.returncode == 0, r.stderr
    st, t, d = tree_state(runs_root, "t1")
    by = {n["id"]: n for n in t.nodes}
    assert t.used() == 6 and not [n for n in t.nodes if n["status"] == "pending"]
    assert t.meta["selections"][0] == ["root", "root"] and by["n0002"]["status"] == "crash"
    assert by["n0006"]["parent_commit"] == by["n0004"]["commit"]   # a child of a no_op node inherits its code
    assert by["n0006"]["status"] == "no_op" and by["n0003"]["parent"] == "n0001" and by["n0003"]["description"] == "lr 0.1"
    tags = subprocess.run(["git", "tag", "-l", "node/*"], cwd=d / "work", capture_output=True, text=True).stdout.split()
    assert {f"node/{n['id']}" for n in t.nodes if n.get("commit")} <= set(tags)
    ch = st["tree_choice"]
    assert len(ch["candidates"]) >= 2 and ch["node"] in ("n0003", "n0005")
    assert sorted(p.name for p in (d / "holdout").iterdir()) == sorted(
        [f"incumbent-s{s}" for s in (101, 102, 103)] + [f"baseline-s{s}" for s in (101, 102, 103)])
    assert st["incumbent"] == ch["node"] and st["verdict"] in ("supported", "inconclusive")
    assert st["verdict"] == "supported", st["verdict_reason"]
    assert (d / "report.md").read_text().count("Pre-registered choice") == 1
    # a second tree rooted at the chosen node: its baseline IS that node's code; scores are relative to it
    r = run_arlab(runs_root, "tree", "run", pack, "--tag", "t1b", "--budget", "2", "--workers", "2",
                  "--root-from", f"_fixture/t1:{ch['node']}", "--script", SCRIPTS / "tree_root.yaml")
    assert r.returncode == 0, r.stderr
    st2, t2, d2 = tree_state(runs_root, "t1b")
    base_code = subprocess.run(["git", "show", "baseline:model.py"], cwd=d2 / "work", capture_output=True, text=True).stdout
    assert '"lr": 0.1' in base_code or '"lr": 0.2' in base_code
    assert st2["root_from"] == f"_fixture/t1:{ch['node']}" and t2.used() == 2
    # DREAM-PROTOCOL block: two arms grown from t1's first batch (shared opening), then paired block-eval
    for arm, pol in (("bD", "stop_first_win"), ("bP", "parallel_refine")):
        r = run_arlab(runs_root, "tree", "run", pack, "--tag", arm, "--policy", pol, "--budget", "4", "--workers", "2",
                      "--opening-from", "_fixture/t1", "--calib-from", "_fixture/t1", "--script", SCRIPTS / "tree.yaml")
        assert r.returncode == 0, r.stderr
        sa, ta, da = tree_state(runs_root, arm)
        assert [n["id"] for n in ta.nodes[1:3]] == ["n0001", "n0002"] and all(n.get("opening") == "_fixture/t1" for n in ta.nodes[1:3])
        assert ta.nodes[1]["commit"] == by["n0001"]["commit"] and ta.used() <= 4
        assert subprocess.run(["git", "cat-file", "-e", by["n0001"]["commit"]], cwd=da / "work").returncode == 0
    _, tP, _ = tree_state(runs_root, "bP")
    assert tP.used() == 4 and {n["parent"] for n in tP.nodes[3:]} == {"n0001", "n0002"}
    out = tmp_path / "block1"
    r = run_arlab(runs_root, "tree", "block-eval", pack, "--arm", "D=bD", "P=bP", "--seeds", "201", "202", "--out", out)
    assert r.returncode == 0, r.stderr
    blk = json.loads((out / "block.json").read_text())
    assert set(blk["root"]) == {"201", "202"} and set(blk["arms"]) == {"D", "P"}
    assert all(isinstance(a["q"], float) and a["nodes"] <= 4 for a in blk["arms"].values())


def test_root_from_allows_code_neutral_pack_changes(tmp_path):
    from types import SimpleNamespace
    from arlab.tree.online import _same_code
    def mk(name, seeds, frozen="x = 1\n", data="d1", extra=""):
        s = tmp_path / name
        (s / "frozen").mkdir(parents=True)
        (s / "frozen" / "e.py").write_text(frozen)
        (s / "pack.yaml").write_text(f"name: p\nseeds: {seeds}\n{extra}")
        return SimpleNamespace(sealed=s, state={"data_hash": data})
    base = mk("a", "{screen: 1, holdout: [101, 102, 103]}")
    assert _same_code(base, mk("b", "{screen: 1, holdout: [101, 102, 103, 104, 105]}"))
    assert not _same_code(base, mk("c", "{screen: 2, holdout: [101, 102, 103]}"))
    assert not _same_code(base, mk("d", "{screen: 1, holdout: [101, 102, 103]}", frozen="x = 2\n"))
    assert not _same_code(base, mk("e", "{screen: 1, holdout: [101, 102, 103]}", data="d2"))
    assert _same_code(base, mk("f", "{screen: 1, holdout: [101]}", extra="guards: [{name: t, max: 1}]\nagent: {model: m}\n"))
    assert not _same_code(base, mk("g", "{screen: 1, holdout: [101]}", extra="run: {command: other}\n"))


def test_protocol_decision_rule():
    from arlab.tree.blocks import analyze
    import random
    def blocks(tmp, dq, dcalls, n=24, sd=0.3):
        r, files = random.Random(0), []
        for b in range(n):
            base = r.gauss(0.5, 0.5)
            arms = {"P": {"q": base + r.gauss(0, sd), "nodes": 12, "calls": 12},
                    "G": {"q": base + r.gauss(0, sd), "nodes": 12, "calls": 12},
                    "D": {"q": base + dq + r.gauss(0, sd), "nodes": 12 + dcalls, "calls": 12 + dcalls}}
            f = tmp / f"b{b}.json"; f.write_text(json.dumps({"arms": arms})); files.append(f)
        return files
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "a").mkdir(); (t / "b").mkdir(); (t / "c").mkdir(); (t / "d").mkdir()
        assert analyze(blocks(t / "a", 1.0, -6))["verdict"] == "proven"
        assert analyze(blocks(t / "b", 1.0, +2))["verdict"] != "proven"     # better but costlier: not proven
        assert analyze(blocks(t / "c", -0.3, -6))["verdict"] == "denied"
        assert analyze(blocks(t / "d", 0.25, -6, n=4))["verdict"] == "inconclusive"
