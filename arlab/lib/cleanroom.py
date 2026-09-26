"""Clean-room hidden-test runner for coding packs (used by EVALUATE, which runs as root; PLAN §6.3).

Only the task's declared source files and its hidden tests are copied into a fresh temp dir; pytest runs there as
uid 65534 with `-I -p no:cacheprovider --noconftest`; the junit report is written to a separate fresh dir owned by
65534 and read by the parent only after the child exits. An item scores 1 iff the exit code is 0 AND the report
lists exactly the expected test IDs, all passed. Planted conftest.py / sitecustomize.py / .pth files or code that
calls os._exit(0) at import therefore cannot produce a pass.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

NOBODY = 65534


def junit_results(path: Path) -> dict[str, str]:
    """{test_id: outcome} with test_id = '<file>::<name>' (class names folded in)."""
    out = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        module, *klass = tc.get("classname", "").split(".")  # tests sit at the clean-room root: "module[.Class]"
        tid = f"{module}.py::" + "::".join(klass + [tc.get("name", "")])
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            if tc.find(tag) is not None:
                outcome = tag
        out[tid] = outcome
    return out


def run_hidden_tests(workspace: Path, sources: list[str], tests_dir: Path, expected: list[str], timeout_s: int = 120) -> dict:
    """Score one task. `expected` ids look like 'test_x.py::test_name' (or 'test_x.py::Class::test_name')."""
    tmp = Path(tempfile.mkdtemp(prefix="cleanroom-"))
    rep = Path(tempfile.mkdtemp(prefix="cleanroom-report-"))
    try:
        for rel in sources:
            src = workspace / rel
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file() and not src.is_symlink():
                shutil.copyfile(src, dst)
        for t in tests_dir.iterdir():
            if t.is_file() and t.name.startswith("test_") and t.suffix == ".py":
                shutil.copyfile(t, tmp / t.name)
        for p in [tmp, *tmp.rglob("*")]:
            os.chown(p, NOBODY, NOBODY)
        os.chown(rep, NOBODY, NOBODY)
        os.chmod(rep, 0o700)
        cmd = ["python", "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
               f"--junitxml={rep}/junit.xml", "-o", "junit_family=xunit2", "."]
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            r = subprocess.run(cmd, cwd=tmp, env=env, user=NOBODY, group=NOBODY, extra_groups=[], capture_output=True,
                               text=True, timeout=timeout_s)
            rc, log = r.returncode, (r.stdout + r.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "reason": "timeout", "log": ""}
        junit = rep / "junit.xml"
        if not junit.exists():
            return {"score": 0.0, "reason": f"no junit report (rc={rc})", "log": log}
        res = junit_results(junit)
        ok = rc == 0 and sorted(res) == sorted(expected) and all(v == "passed" for v in res.values())
        reason = "ok" if ok else f"rc={rc}; got {sorted(res.items())[:10]} expected {sorted(expected)[:10]}"
        return {"score": 1.0 if ok else 0.0, "reason": reason, "log": log}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(rep, ignore_errors=True)
