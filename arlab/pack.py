"""pack.yaml schema (PLAN §3.2), validation, hashing and sealing."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ARLAB_ROOT = Path(__file__).resolve().parent.parent
LIB_DIR = ARLAB_ROOT / "arlab" / "lib"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Image(_M):
    base: str


class Prepare(_M):
    command: str
    timeout_s: int = 3600


class Run(_M):
    command: str
    timeout_s: int = 600
    gpu: bool = False
    mem_gb: float | Literal["auto"] = "auto"


class Evaluate(_M):
    command: str
    timeout_s: int = 300


class Budget(_M):
    unit: str
    limit: int


class Metric(_M):
    name: str
    direction: Literal["minimize", "maximize"]
    mes: float = Field(gt=0)


class Guard(_M):
    name: str
    max: float | None = None
    min: float | None = None
    max_ratio_vs_baseline: float | None = None
    min_ratio_vs_baseline: float | None = None

    @model_validator(mode="after")
    def _one(self):
        n = sum(v is not None for v in (self.max, self.min, self.max_ratio_vs_baseline, self.min_ratio_vs_baseline))
        if n != 1:
            raise ValueError(f"guard {self.name}: exactly one of max/min/max_ratio_vs_baseline/min_ratio_vs_baseline")
        return self


class Seeds(_M):
    calibration: list[int]
    screen: int
    confirm: list[int]
    holdout: list[int]

    @model_validator(mode="after")
    def _check(self):
        if len(self.calibration) < 2:
            raise ValueError("seeds.calibration needs >= 2 seeds")
        missing = ({self.screen} | set(self.confirm)) - set(self.calibration)
        if missing:
            raise ValueError(f"screen/confirm seeds must be calibration seeds (baseline values): {sorted(missing)}")
        if self.calibration[0] != self.screen:
            raise ValueError("seeds.calibration[0] must be the screen seed (it is the PROBE run)")
        if not self.holdout or set(self.holdout) & set(self.calibration):
            raise ValueError("seeds.holdout must be non-empty and disjoint from calibration seeds")
        return self


class Service(_M):
    name: str
    image: str
    command: str
    port: int
    health_url: str = "/health"
    mem_gb: float
    max_model_len: int | None = None
    gpu: bool = True
    env: dict[str, str] = {}
    volumes: list[str] = []


class Reference(_M):
    name: str
    path: str


class Verdict(_M):
    compare_to: str = "baseline"


class Agent(_M):
    model: str = "gpt-6-sol"
    effort: str = "high"
    visible: list[str] = ["program.md", "IDEA.md"]
    timeout_s: int = 900


class CampaignCfg(_M):
    max_experiments: int = 200
    max_hours: float = 24
    max_agent_calls: int = 400
    stop_after_no_keep: int = 40
    rescue_after: int = 8


class Acceptance(_M):
    allow_underpowered: bool = False


class Pack(_M):
    name: str = Field(pattern=r"^[a-z0-9_][a-z0-9_-]*$")
    image: Image
    prepare: Prepare
    run: Run
    evaluate: Evaluate
    budget: Budget
    metric: Metric
    guards: list[Guard] = []
    seeds: Seeds
    services: list[Service] = []
    references: list[Reference] = []
    verdict: Verdict = Verdict()
    agent: Agent = Agent()
    campaign: CampaignCfg = CampaignCfg()
    acceptance: Acceptance = Acceptance()

    @model_validator(mode="after")
    def _check(self):
        names = {r.name for r in self.references}
        if self.verdict.compare_to != "baseline" and self.verdict.compare_to not in names:
            raise ValueError(f"verdict.compare_to {self.verdict.compare_to!r} is neither baseline nor a reference")
        for r in self.references:
            if not r.path.startswith("frozen/run/"):
                raise ValueError(f"reference {r.name}: path must be under frozen/run/")
        for v in self.agent.visible:
            if v not in ("program.md", "IDEA.md") and (not v.startswith("frozen/run/") or ".." in v.split("/")):
                raise ValueError(f"agent.visible {v!r}: only program.md, IDEA.md or files under frozen/run/")
        return self

    @property
    def needs_gpu(self) -> bool:
        return self.run.gpu or any(s.gpu for s in self.services)


REQUIRED = ["IDEA.md", "pack.yaml", "program.md", "surface", "frozen/run", "frozen/eval", "tests"]


def load_pack(pack_dir: Path) -> Pack:
    return Pack.model_validate(yaml.safe_load((Path(pack_dir) / "pack.yaml").read_text()))


def validate_dir(pack_dir: Path) -> list[str]:
    """Static structure checks. Returns a list of problems (empty = ok)."""
    pack_dir = Path(pack_dir)
    errs = [f"missing {r}" for r in REQUIRED if not (pack_dir / r).exists()]
    if errs:
        return errs
    try:
        pack = load_pack(pack_dir)
    except Exception as e:  # pydantic / yaml error text is the useful message
        return [f"pack.yaml: {e}"]
    if pack.name != pack_dir.name:
        errs.append(f"pack.yaml name {pack.name!r} != folder name {pack_dir.name!r}")
    if not surface_files(pack_dir / "surface"):
        errs.append("surface/ is empty")
    for r in pack.references:
        if not (pack_dir / r.path).is_dir():
            errs.append(f"reference {r.name}: {r.path} is not a directory")
    for v in pack.agent.visible:
        if not (pack_dir / v).is_file():
            errs.append(f"agent.visible {v} does not exist")
    return errs


def surface_files(surface: Path) -> list[str]:
    """Relative paths of the editable files (no caches, no hidden files)."""
    out = []
    for p in sorted(Path(surface).rglob("*")):
        rel = p.relative_to(surface).as_posix()
        if p.is_file() and "__pycache__" not in rel and not any(part.startswith(".") for part in rel.split("/")):
            out.append(rel)
    return out


def compile_error(path: Path) -> str | None:
    """Syntax check without writing bytecode."""
    try:
        compile(Path(path).read_bytes(), str(path), "exec")
    except (SyntaxError, ValueError) as e:
        return f"{Path(path).name}: {e}"
    return None


def hash_paths(paths: list[Path], extra: str = "") -> str:
    """Content hash over files/dirs (relative names + bytes), stable across machines."""
    h = hashlib.sha256(extra.encode())
    for root in paths:
        root = Path(root)
        if not root.exists():
            h.update(f"<missing {root.name}>".encode())
            continue
        items = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for p in items:
            h.update(p.relative_to(root.parent).as_posix().encode() + b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


SEAL_ITEMS = ["frozen", "pack.yaml", "program.md", "IDEA.md", "requirements.txt", "surface"]


def seal(pack_dir: Path, dest: Path) -> str:
    """Copy the pack (+ arlab/lib) into dest; return the seal hash. dest must not exist."""
    pack_dir, dest = Path(pack_dir), Path(dest)
    dest.mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in SEAL_ITEMS:
        src = pack_dir / item
        if src.is_dir():
            shutil.copytree(src, dest / item, ignore=ign)
        elif src.is_file():
            shutil.copy2(src, dest / item)
    shutil.copytree(LIB_DIR, dest / "arlab_lib" / "arlab" / "lib", ignore=ign)
    (dest / "arlab_lib" / "arlab" / "__init__.py").write_text("")
    for p in dest.rglob("*"):  # read-only snapshot
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    return seal_hash(dest)


def seal_hash(sealed: Path) -> str:
    sealed = Path(sealed)
    return hash_paths([sealed / i for i in SEAL_ITEMS] + [sealed / "arlab_lib"])
