"""arlab.lib.cleanroom in a real container as root (anti-cheat (i)): planted conftest.py / sitecustomize.py / .pth
files and os._exit(0) at import must not produce a pass."""
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from conftest import ROOT

pytestmark = pytest.mark.docker
IMAGE = "nvcr.io/nvidia/pytorch:25.10-py3"
TESTS = {"test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_neg():\n    assert add(-1, 1) == 0\n"}
EXPECTED = ["test_calc.py::test_add", "test_calc.py::test_neg"]
CASES = {
    "good": {"calc.py": "def add(a, b):\n    return a + b\n"},
    "wrong": {"calc.py": "def add(a, b):\n    return a - b\n"},
    "exit0_at_import": {"calc.py": "import os\nos._exit(0)\n"},
    "planted_conftest": {"calc.py": "def add(a, b):\n    return 0\n",
                         "conftest.py": "import pytest\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                                        "    out = yield\n    out.get_result().outcome = 'passed'\n"},
    "sitecustomize": {"calc.py": "def add(a, b):\n    return 0\n", "sitecustomize.py": "import os\nos._exit(0)\n"},
    "pth": {"calc.py": "def add(a, b):\n    return 0\n", "evil.pth": "import os; os._exit(0)\n"},
    "forged_report": {"calc.py": "import os, sys\nfor a in sys.argv:\n    if a.startswith('--junitxml='):\n"
                                 "        open(a.split('=', 1)[1], 'w').write('<testsuites><testsuite><testcase classname=\"test_calc\" name=\"test_add\"/>'\n"
                                 "            '<testcase classname=\"test_calc\" name=\"test_neg\"/></testsuite></testsuites>')\n"
                                 "        os._exit(0)\n"},
}


def test_cleanroom_scores(tmp_path):
    for name, files in CASES.items():
        ws = tmp_path / "ws" / name
        ws.mkdir(parents=True)
        for f, c in files.items():
            (ws / f).write_text(c)
    tdir = tmp_path / "hidden"
    tdir.mkdir()
    for f, c in TESTS.items():
        (tdir / f).write_text(c)
    script = textwrap.dedent(f"""
        import json
        from pathlib import Path
        from arlab.lib.cleanroom import run_hidden_tests
        out = {{}}
        for ws in sorted(Path('/ws').iterdir()):
            srcs = [p.name for p in ws.iterdir()]   # even if the task declares every file, planting must not help
            out[ws.name] = run_hidden_tests(ws, srcs, Path('/hidden'), {EXPECTED!r})['score']
        print('RESULT' + json.dumps(out))
    """)
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "-e", "PYTHONPATH=/lib",
                        "-v", f"{ROOT / 'arlab'}:/lib/arlab:ro", "-v", f"{tmp_path / 'ws'}:/ws:ro", "-v", f"{tdir}:/hidden:ro",
                        IMAGE, "python", "-c", script], capture_output=True, text=True, timeout=600)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    scores = json.loads(line[6:])
    assert scores == {"good": 1.0, "wrong": 0.0, "exit0_at_import": 0.0, "planted_conftest": 0.0, "sitecustomize": 0.0, "pth": 0.0, "forged_report": 0.0}, scores
