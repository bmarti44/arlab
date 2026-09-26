import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "ideas" / "_fixture"
ARLAB = ROOT / ".venv" / "bin" / "arlab"


def make_variant(base: Path, name: str, pack: dict | None = None, run_cfg: dict | None = None, eval_cfg: dict | None = None) -> Path:
    """Copy ideas/_fixture to base/<name>/_fixture with pack.yaml / frozen config overrides."""
    dest = base / name / "_fixture"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    if pack:
        y = yaml.safe_load((dest / "pack.yaml").read_text())
        for k, v in pack.items():
            y[k] = {**y[k], **v} if isinstance(v, dict) and isinstance(y.get(k), dict) else v
        (dest / "pack.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    if run_cfg:
        (dest / "frozen/run/config.json").write_text(__import__("json").dumps(run_cfg))
    if eval_cfg:
        (dest / "frozen/eval/config.json").write_text(__import__("json").dumps(eval_cfg))
    return dest


def run_arlab(runs: Path, *args, env_extra=None, check=False):
    env = {**os.environ, "ARLAB_RUNS": str(runs), "ARLAB_BACKOFF_S": "1", **(env_extra or {})}
    env.pop("ARLAB_DEBUG_KILL", None) if not (env_extra or {}).get("ARLAB_DEBUG_KILL") else None
    return subprocess.run([str(ARLAB), *map(str, args)], env=env, capture_output=True, text=True, check=check, timeout=3600)


@pytest.fixture(scope="session")
def runs_root(tmp_path_factory):
    d = tmp_path_factory.mktemp("arlab-runs")
    yield d
    subprocess.run(["chmod", "-R", "u+w", str(d)], check=False)
