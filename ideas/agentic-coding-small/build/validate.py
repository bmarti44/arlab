"""Validate generated tasks in the pack image (run as root in a container, like EVALUATE):
starter must score 0; solution must score 1 via the clean-room runner.
Usage: python validate.py /tasks  -> writes /tasks/_validation.json"""
import json
import sys
import time
from pathlib import Path

from arlab.lib.cleanroom import run_hidden_tests

root = Path(sys.argv[1])
report = {}
for td in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
    why = []
    try:
        t = json.loads((td / "task.json").read_text())
        assert t["id"] == td.name, "id != dir name"
        tests = td / "tests"
        files = [p.name for p in tests.iterdir()]
        assert len(files) == 1 and files[0].startswith("test_") and files[0].endswith(".py"), f"tests/ must hold one test_*.py: {files}"
        assert all(e.startswith(files[0] + "::") for e in t["expected"]) and len(set(t["expected"])) == len(t["expected"]) >= 3, "bad expected ids"
        assert all((td / "solution" / s).is_file() for s in t["sources"]), "solution missing a source file"
        assert any((td / "starter").rglob("*")), "empty starter"
        t0 = time.monotonic()
        sol = run_hidden_tests(td / "solution", t["sources"], tests, t["expected"], timeout_s=60)
        secs = time.monotonic() - t0
        if sol["score"] != 1.0:
            why.append("solution fails: " + sol["reason"][:300])
        st = run_hidden_tests(td / "starter", t["sources"], tests, t["expected"], timeout_s=60)
        if st["score"] != 0.0:
            why.append("starter passes")
        if secs > 20:
            why.append(f"tests too slow ({secs:.0f}s)")
    except Exception as e:  # noqa: BLE001
        why.append(f"{type(e).__name__}: {e}")
    report[td.name] = {"ok": not why, "why": why}
    print(td.name, "OK" if not why else "REJECT " + "; ".join(why)[:300], flush=True)
(root / "_validation.json").write_text(json.dumps(report, indent=1))
print(f"{sum(v['ok'] for v in report.values())}/{len(report)} ok")
