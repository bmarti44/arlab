"""CPU checks of agentic-coding-small (TESTS runs as root with /pack, /frozen, /eval, /data): equal disjoint splits,
reference solutions pass and starters fail in the clean room, tools stay inside the working directory."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/frozen")
from arlab.lib.cleanroom import run_hidden_tests  # noqa: E402
from codetools import StepLimit, Tools  # noqa: E402


def load(split, kind):
    return [json.loads(line) for line in open(f"/data/{split}/{kind}/tasks.jsonl")]


def test_splits_equal_disjoint():
    v, h = load("validation", "private"), load("holdout", "private")
    sp = json.load(open("/data/splits.json"))
    assert len(v) == len(h) == sp["validation"] == sp["holdout"] >= 5
    assert not {t["id"] for t in v} & {t["id"] for t in h}
    pub = load("validation", "public")
    assert [t["id"] for t in pub] == [t["id"] for t in v] and "expected" not in pub[0] and "sources" not in pub[0]


def test_reference_passes_starter_fails():
    for t in load("validation", "private")[:3] + load("holdout", "private")[:3]:
        td = Path("/pack/frozen/prepare/tasks") / t["id"]
        tests = Path("/data/validation/private/tests") / t["id"]
        if not tests.exists():
            tests = Path("/data/holdout/private/tests") / t["id"]
        assert run_hidden_tests(td / "solution", t["sources"], tests, t["expected"])["score"] == 1.0, t["id"]
        assert run_hidden_tests(td / "starter", t["sources"], tests, t["expected"])["score"] == 0.0, t["id"]


def test_tools_confined_and_capped(tmp_path):
    tl = Tools(tmp_path, max_steps=5, deadline=time.monotonic() + 60)
    assert tl.write("a/b.py", "x = 1\n").startswith("wrote")
    assert tl.read("../../etc/passwd").startswith("error")
    assert tl.edit("a/b.py", "x = 1", "x = 2") == "edited a/b.py" and (tmp_path / "a/b.py").read_text() == "x = 2\n"
    assert tl.run("echo hi").startswith("exit=0\nhi")
    assert "timeout" in tl.run("sleep 5", timeout=1)
    try:
        tl.run("true")
        raise AssertionError("step cap not enforced")
    except StepLimit:
        pass
