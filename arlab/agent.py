"""Research-agent backends (PLAN §3.6). CodexBackend is the only real one; ScriptedBackend drives tests."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from .execute import kill_cid

AGENT_IMAGE = "arlab-agent:0.157.1"
CODEX_HOME = Path.home() / ".cache" / "arlab" / "codex-home"
SCHEMA = Path(__file__).resolve().parent / "proposal.schema.json"
AUTH_MARKERS = ("401", "unauthorized", "not logged in", "login required", "please log in", "refresh token")


class InfraError(Exception):
    """Agent CLI / docker / rate limit / network failure: not an experiment."""


class AuthError(Exception):
    """Codex login is broken: pause the campaign."""


@dataclass
class Proposal:
    action: str
    description: str
    hypothesis_tag: str
    constraint_learned: str | None
    model: str = ""
    tokens: dict = field(default_factory=dict)
    seconds: float = 0.0


class Backend(Protocol):
    def propose(self, view: Path, run_dir: Path, name: str) -> Proposal: ...
    def fix(self, view: Path, run_dir: Path, name: str, log_tail: str) -> Proposal: ...
    def rescue(self, view: Path, run_dir: Path, name: str) -> Proposal: ...


def parse_proposal(obj: dict) -> Proposal:
    if set(obj) != {"action", "description", "hypothesis_tag", "constraint_learned"} or obj["action"] not in ("edit", "skip"):
        raise InfraError(f"proposal does not match schema: {obj}")
    return Proposal(obj["action"], str(obj["description"])[:120], str(obj["hypothesis_tag"]), obj["constraint_learned"])


def usage_from_events(events: Path) -> dict:
    tot = {"input": 0, "cached_input": 0, "output": 0}
    if not events.exists():
        return tot
    for line in events.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        u = ev.get("usage") if isinstance(ev, dict) else None
        if ev.get("type") == "turn.completed" and isinstance(u, dict):
            tot["input"] += int(u.get("input_tokens", 0))
            tot["cached_input"] += int(u.get("cached_input_tokens", 0))
            tot["output"] += int(u.get("output_tokens", 0))
    return tot


class CodexBackend:
    """Containerized `codex exec`, one call per experiment. No host fallback."""

    def __init__(self, model: str, effort: str, timeout_s: int, rescue_model: str = "gpt-6-astra"):
        self.model, self.effort, self.timeout_s, self.rescue_model = model, effort, timeout_s, rescue_model

    def _call(self, view: Path, out: Path, name: str, model: str, effort: str) -> Proposal:
        out.mkdir(parents=True, exist_ok=True)
        (out / "proposal.json").unlink(missing_ok=True)
        cidfile = out / "agent.cid"
        cidfile.unlink(missing_ok=True)
        cmd = ["docker", "run", "--rm", "-i", "--cidfile", str(cidfile), "--name", name, "--user", "1000:1000", "-e", "HOME=/tmp",
               "-v", f"{CODEX_HOME}:/codex", "-e", "CODEX_HOME=/codex",
               "-v", f"{view.resolve()}:/work", "-v", f"{out.resolve()}:/out", "-v", f"{SCHEMA}:/schema.json:ro",
               AGENT_IMAGE, "codex", "exec", "--ephemeral", "-C", "/work", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "-m", model, "-c", f'model_reasoning_effort="{effort}"',
               "--output-schema", "/schema.json", "-o", "/out/proposal.json", "--json", "-"]
        t0 = time.monotonic()
        with open(view / "prompt.md", "rb") as fin, open(out / "events.jsonl", "wb") as fev, open(out / "stderr.log", "wb") as ferr:
            p = subprocess.Popen(cmd, stdin=fin, stdout=fev, stderr=ferr)
            try:
                p.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                kill_cid(cidfile.read_text().strip() if cidfile.exists() else None)
                p.wait()
                raise InfraError(f"agent timeout after {self.timeout_s}s")
        secs = time.monotonic() - t0
        text = ((out / "stderr.log").read_text(errors="replace") + (out / "events.jsonl").read_text(errors="replace")).lower()
        if p.returncode != 0:
            if any(m in text for m in AUTH_MARKERS):
                raise AuthError(text[-500:])
            raise InfraError(f"codex exit {p.returncode}: {text[-500:]}")
        try:
            prop = parse_proposal(json.loads((out / "proposal.json").read_text()))
        except (OSError, ValueError) as e:
            raise InfraError(f"no valid proposal.json: {e}")
        prop.model, prop.tokens, prop.seconds = model, usage_from_events(out / "events.jsonl"), secs
        return prop

    def propose(self, view, run_dir, name):
        return self._call(view, run_dir / "agent", name, self.model, self.effort)

    def fix(self, view, run_dir, name, log_tail):
        (view / "crash.log").write_text(log_tail)
        with open(view / "prompt.md", "a") as f:
            f.write("\n\n# Your previous edit crashed\nThe last 80 lines of the run log are in crash.log. Fix the crash "
                    "(keep the idea if possible) with one edit, then emit the JSON object.\n")
        return self._call(view, run_dir / "agent-fix", name + "-fix", self.model, self.effort)

    def rescue(self, view, run_dir, name):
        return self._call(view, run_dir / "agent-rescue", name + "-rescue", self.rescue_model, "high")


def codex_login_ok() -> bool:
    r = subprocess.run(["docker", "run", "--rm", "--user", "1000:1000", "-e", "HOME=/tmp", "-v", f"{CODEX_HOME}:/codex",
                        "-e", "CODEX_HOME=/codex", AGENT_IMAGE, "codex", "login", "status"],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and "logged in" in (r.stdout + r.stderr).lower()


class ScriptedBackend:
    """Applies prepared edits from a YAML list, indexed by the number of counted experiments (tests only).

    Entry: {action, description, hypothesis_tag, constraint_learned, replace: {file: [[old, new], ...]},
            write: {file: content}, infra_errors: N, fix: {<same edit keys>}}
    """

    def __init__(self, script: Path, index_fn):
        self.entries = yaml.safe_load(Path(script).read_text())
        self.index_fn = index_fn
        self.failures: dict[int, int] = {}

    def _apply(self, view: Path, e: dict) -> Proposal:
        for f, pairs in (e.get("replace") or {}).items():
            text = (view / f).read_text()
            for old, new in pairs:
                if old not in text:
                    raise RuntimeError(f"scripted replace: {old!r} not in {f}")
                text = text.replace(old, new)
            (view / f).write_text(text)
        for f, content in (e.get("write") or {}).items():
            (view / f).write_text(content)
        return Proposal(e.get("action", "edit"), e.get("description", "scripted"), e.get("hypothesis_tag", "scripted"),
                        e.get("constraint_learned"), model="scripted")

    def propose(self, view, run_dir, name):
        i = self.index_fn()
        if i >= len(self.entries):
            return Proposal("skip", "script exhausted", "exhausted", None, model="scripted")
        e = self.entries[i]
        if self.failures.get(i, 0) < e.get("infra_errors", 0):
            self.failures[i] = self.failures.get(i, 0) + 1
            raise InfraError("scripted infra error")
        return self._apply(view, e)

    def fix(self, view, run_dir, name, log_tail):
        e = self.entries[self.index_fn()]
        return self._apply(view, e["fix"]) if e.get("fix") else Proposal("skip", "no fix", e.get("hypothesis_tag", "x"), None, "scripted")

    def rescue(self, view, run_dir, name):
        (view / "notes.md").write_text("# scripted rescue note\n")
        return Proposal("skip", "rescue", "rescue", None, model="scripted")
