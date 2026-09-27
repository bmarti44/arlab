#!/usr/bin/env python
"""accept-M3 / accept-M4 / accept-M5-<pack> / accept-M6 (PLAN §5). Exit code is the verdict.

Campaign tags: M3 = nanochat-lite/<docs/M3.json tag, default m3>; M4 = the Appendix-B pack (docs/M4.json names it); M5 = <pack>/<docs/M5.json tag, default m5>.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arlab.campaign import RUNS  # noqa: E402
from arlab.record import counted, load_records, read_json  # noqa: E402
from arlab.report import speed  # noqa: E402

fails = []
OWN_RULES = ("max_experiments", "max_hours", "max_agent_calls", "no_keep", "underpowered")
AGENT_EDIT = ("keep", "discard", "crash", "oom", "timeout", "invalid", "guard_fail", "contended")


def check(ok, msg):
    print(("ok: " if ok else "FAIL: ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def campaign_checks(name, tag, min_exp=1, min_active_h=0.0, own_rules=True):
    d = RUNS / name / tag
    st = read_json(d / "state.json", {}) or {}
    check(d.exists(), f"{name}/{tag} exists")
    recs = load_records(d)
    cnt = counted(recs)
    log = (d / "runner.log").read_text(errors="replace") if (d / "runner.log").exists() else ""
    check("Traceback" not in log, f"{name}/{tag}: zero tracebacks in runner.log")
    check(st.get("phase") == "finalized" and st.get("verdict") in ("supported", "not_found_at_this_scale", "inconclusive"),
          f"{name}/{tag}: finalized with verdict {st.get('verdict')} ({st.get('verdict_reason')})")
    if own_rules:
        check(st.get("stop_reason") in OWN_RULES,
              f"{name}/{tag}: ended by its own rules (stop_reason={st.get('stop_reason')})")
    check(len(cnt) >= min_exp, f"{name}/{tag}: {len(cnt)} experiments >= {min_exp}")
    sp = speed(recs)
    check(sp["active_h"] >= min_active_h, f"{name}/{tag}: active hours {sp['active_h']:.2f} >= {min_active_h}")
    edits = [r for r in cnt if r["status"] in AGENT_EDIT]
    missing = [r["id"] for r in edits if not r.get("description") or not (d / "runs" / r["id"] / "diff.patch").exists()
               or not (d / "runs" / r["id"] / "diff.patch").read_text().strip()]
    check(not missing, f"{name}/{tag}: every edit record has a diff and a description (missing: {missing[:5]})")
    rep = (d / "report.md").read_text() if (d / "report.md").exists() else ""
    check("experiments/hour" in rep and "propose" in rep, f"{name}/{tag}: report has the time breakdown and experiments/hour")
    if sp["n"] and sp["per_hour"] < 8:
        check("speed target missed" in rep, f"{name}/{tag}: report labels 'speed target missed' ({sp['per_hour']:.2f}/h)")
    print(f"   {name}/{tag}: {sp['n']} experiments, {sp['per_hour']:.2f}/h active, median {sp['median_min']:.1f} min, "
          f"agent share {sp['agent_share']:.0%}, verdict {st.get('verdict')} ({st.get('verdict_reason')})")
    return d, st, recs


M3_TAG = (read_json(ROOT / "docs" / "M3.json", {}) or {}).get("tag", "m3")  # a re-run under a new tag is recorded there
M5_TAGS = (read_json(ROOT / "docs" / "M5.json", {}) or {}).get("tags", {})    # {pack: tag} for re-runs; default m5
mode = sys.argv[1]
if mode == "m3":
    d, st, recs = campaign_checks("nanochat-lite", M3_TAG, min_exp=12, min_active_h=2.0)
    kills = read_json(d / "kill-tests.json", {}) or {}
    for k in ("during_agent", "during_training"):
        check(kills.get(k, {}).get("resumed") is True, f"kill -9 {k.replace('_', ' ')} then resume: {kills.get(k)}")
    check(any(r["status"] == "interrupted" for r in recs), "interrupted rows recorded after kill -9")
    check(all(r.get("model", "").startswith("gpt-6") for r in counted(recs) if r["status"] in AGENT_EDIT),
          "all edits proposed by gpt-6 models")
elif mode == "m4":
    spec = read_json(ROOT / "docs" / "M4.json")
    check(spec is not None, "docs/M4.json names the Appendix-B pack, tag and arlab/ commit at the start of M4")
    if spec:
        name, tag = spec["pack"], spec["tag"]
        pack = ROOT / "ideas" / name
        check((pack / "REVIEW.md").exists() and (pack / "REVIEW.md").read_text().strip(), f"{name}: astra review saved in REVIEW.md")
        r = subprocess.run([str(ROOT / ".venv/bin/arlab"), "check", str(pack)], capture_output=True, text=True)
        check(r.returncode == 0 and "PASS" in r.stdout, f"arlab check ideas/{name}")
        d, st, recs = campaign_checks(name, tag, min_exp=3, own_rules=True)
        agent = [x for x in counted(recs) if x.get("model", "").startswith("gpt-6")]
        check(len(agent) >= 3, f"{name}/{tag}: {len(agent)} agent experiments >= 3")
        check(speed(recs)["active_h"] <= 0.5 + 1e-9, f"{name}/{tag}: <= 30-minute campaign")
        end = spec.get("end_commit", "HEAD")  # recorded when the M4 campaign finished
        gd = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", spec["arlab_commit"], end, "--", "arlab/"],
                            capture_output=True, text=True)
        diff = gd.stdout.strip() if gd.returncode == 0 else f"git diff failed: {gd.stderr.strip()}"
        check(not diff, f"arlab/ unchanged during M4 (git diff --stat {spec['arlab_commit'][:8]} {end[:8]} -- arlab/): {diff[:200]}")
elif mode == "m5":
    name = sys.argv[2]
    tag = M5_TAGS.get(name, "m5")
    d, st, recs = campaign_checks(name, tag, min_exp=1)
    check(st.get("underpowered") is False, f"{name}/{tag} calibration not underpowered (SE {st.get('expected_holdout_se')})")
    evaluated = [x for x in counted(recs) if x.get("model", "").startswith("gpt-6") and x.get("primary") is not None]
    check(len(evaluated) >= 1, f"{name}/{tag}: {len(evaluated)} agent candidates fully evaluated")
    ck = d / "check.log"
    check(ck.exists() and "PASS" in ck.read_text(), f"{name}: full arlab check passed (saved in {ck})")
    import yaml
    cfg = yaml.safe_load((d / "sealed" / "pack.yaml").read_text())["campaign"]
    check(cfg["max_hours"] == 6 and cfg["stop_after_no_keep"] == 25, f"{name}/{tag} limits max_hours 6, stop_after_no_keep 25")
    if name == "memory-longmemeval":
        tele = [json.loads(line) for p in sorted(d.glob("**/telemetry.jsonl")) for line in p.read_text().splitlines()]
        evals = {str(p.parent) for p in d.glob("**/eval.log")}
        check(len(evals) >= 2 and not any(r["status"] == "contended" for r in recs) and all(not t.get("foreign") for t in tele),
              f"resident vLLM never treated as foreign across {len(evals)} evaluations")
elif mode == "m6":
    total = 0.0
    for name, tag in [("nanochat-lite", M3_TAG)] + [(p, M5_TAGS.get(p, "m5")) for p in ("memory-longmemeval", "agentic-coding-small")]:
        d = RUNS / name / tag
        st = read_json(d / "state.json", {}) or {}
        log = (d / "runner.log").read_text(errors="replace") if (d / "runner.log").exists() else ""
        check(d.exists() and "Traceback" not in log, f"{name}/{tag}: runner.log has zero tracebacks")
        check(st.get("phase") == "finalized" and st.get("verdict") is not None, f"{name}/{tag}: finalized, verdict {st.get('verdict')}")
        check(st.get("stop_reason") in OWN_RULES, f"{name}/{tag}: ended by its own rules (stop_reason={st.get('stop_reason')})")
        total += speed(load_records(d))["active_h"]
    print(f"   total active hours (M3 + M5): {total:.2f}")
else:
    sys.exit(f"unknown mode {mode}")

print(f"accept-{mode.upper()}{'-' + sys.argv[2] if len(sys.argv) > 2 else ''}:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
