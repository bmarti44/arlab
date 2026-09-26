"""CPU fixture campaigns driven by the ScriptedBackend (accept-M1). Needs docker + the pytorch image."""
import json
import subprocess
from pathlib import Path

import pytest

from arlab.record import load_records
from conftest import ROOT, make_variant, run_arlab

pytestmark = pytest.mark.docker
SCRIPTS = ROOT / "tests" / "scripts"


def campaign(runs_root, tag):
    return runs_root / "_fixture" / tag


def state(runs_root, tag):
    return json.loads((campaign(runs_root, tag) / "state.json").read_text())


def seq(runs_root, tag):
    return [(r["status"], r.get("primary"), r.get("hypothesis_tag")) for r in load_records(campaign(runs_root, tag))
            if r["status"] != "interrupted"]


def go(runs_root, tmp_path, tag, script, env=None, **variant):
    pack = make_variant(tmp_path, tag, **variant)
    return pack, run_arlab(runs_root, "run", pack, "--tag", tag, "--script", SCRIPTS / script, env_extra=env)


def test_statuses_anticheat_and_supported(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "a-statuses", "statuses.yaml", pack={"campaign": {"max_experiments": 13}})
    assert r.returncode == 0, r.stderr
    recs = load_records(campaign(runs_root, "a-statuses"))
    got = [x["status"] for x in recs]
    assert got == ["keep", "discard", "discard", "crash", "timeout", "invalid", "guard_fail", "no_op", "skip",
                   "infra_error", "discard", "discard", "crash", "crash"], got
    by_tag = {x["hypothesis_tag"]: x for x in recs}
    assert by_tag["crashfix"].get("fixed")                       # crash-fix path re-ran the screen
    assert by_tag["anticheat-a"]["primary"] < 0.9                # (a) fake metric ignored
    assert "FileNotFoundError" in (campaign(runs_root, "a-statuses") / "runs" / by_tag["anticheat-b"]["id"] / "s1" / "run.log").read_text()  # (b)
    assert "train_s" in by_tag["anticheat-e"]["reason"]          # (e)
    st = state(runs_root, "a-statuses")
    assert st["verdict"] == "supported" and st["stop_reason"] == "max_experiments"
    assert (campaign(runs_root, "a-statuses") / "report.md").exists()
    assert not list(campaign(runs_root, "a-statuses").glob("runs/*/s*/out")) or all(
        p.parent.parent.name == by_tag["lr"]["id"] for p in campaign(runs_root, "a-statuses").glob("runs/*/s*/out"))


def test_not_found_items(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "b-notfound", "notfound.yaml", pack={"campaign": {"max_experiments": 10}}, eval_cfg={"items": True})
    assert r.returncode == 0, r.stderr
    st = state(runs_root, "b-notfound")
    assert st["n_validation_items"] == 2000 and st["verdict"] == "not_found_at_this_scale", st.get("verdict_reason")
    assert st["holdout"]["skipped"]


def test_underpowered(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "c-underpowered", "notfound.yaml", pack={"metric": {"mes": 0.005}})
    assert r.returncode == 0, r.stderr
    st = state(runs_root, "c-underpowered")
    assert (st["verdict"], st["verdict_reason"]) == ("inconclusive", "underpowered")
    assert not (campaign(runs_root, "c-underpowered") / "holdout").exists() and not load_records(campaign(runs_root, "c-underpowered"))


def test_holdout_uncertain(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "d-uncertain", "uncertain.yaml", pack={"campaign": {"max_experiments": 10}, "metric": {"mes": 0.038}},
              eval_cfg={"items": True})
    assert r.returncode == 0, r.stderr
    st = state(runs_root, "d-uncertain")
    assert load_records(campaign(runs_root, "d-uncertain"))[0]["status"] == "keep"
    assert (st["verdict"], st["verdict_reason"]) == ("inconclusive", "holdout_uncertain"), st["holdout"]


def test_deterministic_pack(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "e-det", "det.yaml", run_cfg={"seed_noise": False}, eval_cfg={"items": True},
              pack={"seeds": {"calibration": [1, 2], "screen": 1, "confirm": [], "holdout": [101]}, "campaign": {"max_experiments": 2}})
    assert r.returncode == 0, r.stderr
    st = state(runs_root, "e-det")
    assert st["sigma"] == 0 and [x["status"] for x in load_records(campaign(runs_root, "e-det"))] == ["keep", "discard"]
    assert st["verdict"] == "supported"


KILL = {"campaign": {"max_experiments": 4}}


def test_kill_minus_9_resume_matches_uninterrupted(runs_root, tmp_path):
    _, r = go(runs_root, tmp_path, "f-ref", "kill.yaml", pack=KILL)
    assert r.returncode == 0, r.stderr
    ref = seq(runs_root, "f-ref")
    assert [s for s, _, _ in ref] == ["keep", "discard", "crash", "discard"]
    for tag, point in (("f-kill-run", "during_run:0002"), ("f-kill-rec", "after_record:0002")):
        pack, r = go(runs_root, tmp_path, tag, "kill.yaml", env={"ARLAB_DEBUG_KILL": point}, pack=KILL)
        assert r.returncode == -9, (tag, r.returncode, r.stderr)
        r2 = run_arlab(runs_root, "run", pack, "--tag", tag, "--script", SCRIPTS / "kill.yaml")
        assert r2.returncode == 0, r2.stderr
        assert seq(runs_root, tag) == ref, tag
        statuses = [x["status"] for x in load_records(campaign(runs_root, tag))]
        assert ("interrupted" in statuses) == point.startswith("during_run"), statuses
        assert state(runs_root, tag)["verdict"] == state(runs_root, "f-ref")["verdict"]
    left = subprocess.run(["docker", "ps", "-q", "--filter", "name=arlab-_fixture-f-kill"], capture_output=True, text=True).stdout
    assert not left.strip()


def _tamper(runs_root, tmp_path, tag, variant, mutate):
    pack, r = go(runs_root, tmp_path, tag, "kill.yaml", env={"ARLAB_DEBUG_KILL": "after_record:0001"}, pack=KILL, **variant)
    assert r.returncode == -9, r.stderr
    mutate(campaign(runs_root, tag))
    r2 = run_arlab(runs_root, "run", pack, "--tag", tag, "--script", SCRIPTS / "kill.yaml")
    assert r2.returncode != 0 and "TamperError" in r2.stderr, r2.stderr[-2000:]


def test_tampered_sealed_refuses(runs_root, tmp_path):
    def mutate(c):
        f = c / "sealed" / "surface" / "model.py"
        f.chmod(0o644)
        f.write_text(f.read_text() + "\n# tampered\n")
    _tamper(runs_root, tmp_path, "g-tamper-seal", {}, mutate)


def test_tampered_data_refuses(runs_root, tmp_path):
    ddirs = []

    def mutate(c):
        ddir = Path(json.loads((c / "state.json").read_text())["data_dir"])
        ddirs.append(ddir)
        f = ddir / "train" / "y.npy"
        subprocess.run(["chmod", "u+w", str(ddir / "train"), str(f)], check=True)
        f.write_bytes(f.read_bytes()[:-1] + b"\x01")
    try:
        _tamper(runs_root, tmp_path, "g-tamper-data", {"run_cfg": {"seed_noise": True, "tamper_test": 1}}, mutate)
    finally:
        for d in ddirs:  # arlab-created, private to this test
            subprocess.run(["chmod", "-R", "u+w", str(d)], check=True)
            subprocess.run(["rm", "-rf", str(d)], check=True)
