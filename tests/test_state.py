"""State reconstruction from records, the contended rule (simulated), pack validation, constraints."""
import shutil
import subprocess

import pytest
import yaml

from arlab import experiment
from arlab.campaign import Campaign, check_metrics
from arlab.pack import Pack, load_pack, validate_dir
from arlab.record import add_constraints, write_json
from conftest import FIXTURE, make_variant


def res(p):
    return {"status": "ok", "primary": p, "metrics": {"train_s": 1.0}, "items": None, "run_s": 1, "eval_s": 1, "wait_s": 0}


@pytest.fixture
def camp(tmp_path):
    c = Campaign(FIXTURE, "unit", runs_root=tmp_path)
    c.pack = load_pack(FIXTURE)
    c.dir.mkdir(parents=True)
    shutil.copytree(FIXTURE / "surface", c.work)
    c.git("init", "-q", "-b", "main")
    c.git("add", "-A")
    c.git("commit", "-q", "-m", "baseline")
    base = c.git("rev-parse", "HEAD")
    c.state = {"baseline_commit": base, "sigma": 0.01}
    for s in c.pack.seeds.calibration:
        (c.dir / "calib" / f"baseline-s{s}").mkdir(parents=True)
        write_json(c.dir / "calib" / f"baseline-s{s}" / "result.json", res(0.35))
    return c


def rec(c, i, status, commit=None, results=None, calls=1):
    d = c.dir / "runs" / f"{i:04d}"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "record.json", {"id": f"{i:04d}", "status": status, "commit": commit, "results": results or {},
                                   "agent_calls": calls, "timings": {"propose_s": 1, "run_s": 1, "eval_s": 1, "wait_s": 100}})


def test_incumbent_and_counters_rebuilt_from_records(camp):
    from arlab.record import load_records
    (camp.work / "model.py").write_text("# changed\n")
    camp.git("commit", "-qam", "k")
    k = camp.git("rev-parse", "HEAD")
    rec(camp, 1, "discard")
    rec(camp, 2, "keep", k, {"1": res(0.5), "2": res(0.5), "3": res(0.5)})
    rec(camp, 3, "interrupted", calls=0)
    rec(camp, 4, "infra_error", calls=0)
    rec(camp, 5, "discard")
    records = load_records(camp.dir)
    commit, vals, iid = camp.incumbent(records)
    assert (commit, iid) == (k, "0002") and vals[2]["primary"] == 0.5
    assert camp.stop_reason(records) is None
    camp.pack.campaign.stop_after_no_keep = 1
    assert camp.stop_reason(records) == "no_keep"
    camp.pack.campaign.stop_after_no_keep = 40
    camp.pack.campaign.max_experiments = 3  # interrupted + infra_error are not counted
    assert camp.stop_reason(records) == "max_experiments"


def test_run_dir_without_record_becomes_interrupted_and_work_resets(camp):
    from arlab.record import load_records
    (camp.dir / "runs" / "0001").mkdir(parents=True)
    (camp.work / "model.py").write_text("dirty\n")
    camp.resume_records()
    r = load_records(camp.dir)
    assert [x["status"] for x in r] == ["interrupted"]
    assert "dirty" not in (camp.work / "model.py").read_text()


def test_infra_streak_stops(camp):
    from arlab.record import load_records
    for i in range(1, 7):
        rec(camp, i, "infra_error", calls=0)
    assert camp.stop_reason(load_records(camp.dir)) == "infra"


def test_contended_rule_loop_reruns_once(camp, monkeypatch):
    seq = iter([{"status": "contended", "reason": "x"}, {**res(0.4)}])
    monkeypatch.setattr(camp, "trial", lambda *a, **k: next(seq))
    monkeypatch.setattr(camp, "export", lambda commit, dest: dest)
    assert experiment.candidate_trial(camp, "c", 1, camp.dir / "r", "0001")["status"] == "ok"
    seq2 = iter([{"status": "contended", "reason": "x"}] * 2)
    monkeypatch.setattr(camp, "trial", lambda *a, **k: next(seq2))
    assert experiment.candidate_trial(camp, "c", 1, camp.dir / "r", "0001")["status"] == "contended"


def test_contended_rule_calibration_waits_until_clean(camp):
    seq = iter([{"status": "contended", "reason": "x"}] * 3 + [res(0.3)])
    assert camp.settled(lambda: next(seq))["status"] == "ok"


def test_metrics_contract():
    assert check_metrics({"valid": True, "primary": 1.0, "metrics": {}, "items": None, "message": ""}) is None
    assert check_metrics({"valid": True, "primary": float("nan"), "metrics": {}, "items": None, "message": ""})
    assert check_metrics({"valid": False, "primary": None, "metrics": {}, "items": None, "message": "bad"})
    assert check_metrics({"valid": True, "primary": 1.0, "metrics": {}, "items": {"a": float("inf")}, "message": ""})
    assert check_metrics(None)


def test_pack_schema_rejects_unknown_fields_and_bad_seeds(tmp_path):
    y = yaml.safe_load((FIXTURE / "pack.yaml").read_text())
    Pack.model_validate(y)
    with pytest.raises(Exception):
        Pack.model_validate({**y, "surprise": 1})
    with pytest.raises(Exception):
        Pack.model_validate({**y, "seeds": {**y["seeds"], "confirm": [9]}})
    with pytest.raises(Exception):
        Pack.model_validate({**y, "guards": [{"name": "x", "max": 1, "min": 0}]})
    assert validate_dir(FIXTURE) == []
    v = make_variant(tmp_path, "v", pack={"agent": {"visible": ["frozen/eval/evaluate.py"]}})
    assert any("agent.visible" in e for e in validate_dir(v))


def test_constraints_dedup_and_cap(tmp_path):
    add_constraints(tmp_path, "0001", "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB", None)
    add_constraints(tmp_path, "0002", "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB", "lr > 0.1 diverges")
    lines = (tmp_path / "constraints.md").read_text().splitlines()
    assert len(lines) == 2 and lines[0].startswith("- [0001] CUDA OOM") and "diverges" in lines[1]
    for i in range(60):
        add_constraints(tmp_path, f"{i:04d}", "", f"fact {i}")
    assert len((tmp_path / "constraints.md").read_text().splitlines()) == 50
