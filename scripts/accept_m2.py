#!/usr/bin/env python
"""accept-M2: nanochat-lite on the GPU (PLAN §5 M2). Exit code is the verdict."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arlab import guards  # noqa: E402
from arlab.campaign import Campaign  # noqa: E402
from arlab.pack import load_pack  # noqa: E402
from arlab.record import load_records  # noqa: E402

PACK = ROOT / "ideas" / "nanochat-lite"
EXPECTED = ["keep", "crash", "oom", "invalid", "crash", "crash", "guard_fail", "skip"]  # tests/scripts/nanochat_m2.yaml + exhausted
fails = []


def check(ok, msg):
    print(("ok: " if ok else "FAIL: ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def camp(tag):
    c = Campaign(PACK, tag)
    c.pack = load_pack(c.sealed) if c.sealed.exists() else None
    return c


# 1. arlab check
r = subprocess.run([str(ROOT / ".venv/bin/arlab"), "check", str(PACK)], capture_output=True, text=True)
check(r.returncode == 0 and "PASS" in r.stdout, "arlab check ideas/nanochat-lite")

# 2. pilot set budget/MES; the fresh tag calibrates as not underpowered
pilot, m2 = camp("pilot"), camp("m2")
check("sigma" in pilot.state, f"pilot calibration exists (sigma={pilot.state.get('sigma')})")
check("pilot" in (ROOT / "DECISIONS.md").read_text(), "pilot recorded in DECISIONS.md")
check(m2.state.get("underpowered") is False, f"tag m2 not underpowered (expected SE {m2.state.get('expected_holdout_se')}, mes {m2.pack and m2.pack.metric.mes})")

# 3. scripted campaign statuses (b), (c), (e), (f) included
recs = [x for x in load_records(m2.dir) if x["status"] not in ("interrupted", "infra_error")]
got = [x["status"] for x in recs]
check(got == EXPECTED, f"scripted statuses {got}")
check(m2.state.get("phase") == "finalized" and m2.state.get("verdict") is not None, f"m2 finalized with verdict {m2.state.get('verdict')}")
by_tag = {x["hypothesis_tag"]: x for x in recs}
for t, st in (("anticheat-b", "crash"), ("anticheat-c", "crash"), ("anticheat-e", "guard_fail"), ("anticheat-f", "invalid")):
    check(by_tag.get(t, {}).get("status") == st, f"({t[-1]}) {t} ran and ended {st}: {by_tag.get(t, {}).get('status')}")
f = by_tag.get("anticheat-f", {})
check("non-causal" in f.get("reason", ""), f"(f) non-causal → invalid: {f.get('reason', '')[:80]}")
e = by_tag.get("anticheat-e", {})
check("train_s" in e.get("reason", ""), f"(e) 2x compute per step caught by the train_s guard: {e.get('reason', '')[:80]}")


def run_log(t):
    p = m2.dir / "runs" / by_tag[t]["id"] / "s1" / "run.log" if t in by_tag else None
    return p.read_text() if p and p.exists() else ""


check("FileNotFoundError" in run_log("anticheat-b") and "LEAK" not in run_log("anticheat-b"), "(b) private/holdout/eval not mounted in RUN")
check(any(m in run_log("anticheat-c") for m in ("Read-only file system", "Permission denied")), "(c) /frozen is read-only")

# 4. (g): same checkpoint, surface loss scaled by 0.5 → bit-identical val_bpb
keeps = [x for x in recs if x["status"] == "keep"]
if keeps:
    k = keeps[0]
    out = m2.dir / "runs" / k["id"] / "s1" / "out"
    gdir = m2.dir / "anticheat-g"
    shutil.rmtree(gdir, ignore_errors=True)
    a = m2.export(k["commit"], gdir / "surface-a")
    b = gdir / "surface-b"
    shutil.copytree(a, b)
    src = (b / "train.py").read_text()
    check("return F.cross_entropy(" in src, "(g) surface has a loss function to alter")
    (b / "train.py").write_text(src.replace("return F.cross_entropy(", "return 0.5 * F.cross_entropy("))
    lock = guards.gpu_lock()
    lock.acquire()
    try:
        vals = []
        for name, surf in (("a", a), ("b", b)):
            guards.wait_for_free(m2.need_gb(), set(), True)
            res = m2.evaluate_step(surf, out, "validation", gdir / name)
            vals.append(json.loads((gdir / name / "result" / "metrics.json").read_text()) if res.rc == 0 else {})
    finally:
        lock.release()
    check(all(v.get("valid") for v in vals) and vals[0]["primary"] == vals[1]["primary"],
          f"(g) val_bpb bit-identical with the surface loss halved: {[v.get('primary') for v in vals]}")
else:
    check(False, "(g) needs a kept checkpoint")

print("accept-M2:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
