"""Exploration policies run as untrusted code: one docker container per replay/online run, JSON lines over stdin/stdout.

A policy file defines  select(nodes: list[dict], budget_left: int, W: int) -> list[str]
  nodes: the revealed tree in order; each {id, parent, depth, order, status, score, cost_s, children, leaf}
         (score = improvement over the pack baseline in MES units, None if the attempt failed; root score is 0)
  return: up to W ids from A(T) = {"root"} ∪ leaves (repeats allowed); [] = stop searching.
It must be a pure function of its inputs (it can be restarted between any two calls).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import select as _select
import subprocess
import time
import uuid
from pathlib import Path

from ..agent import AGENT_IMAGE

HERE = Path(__file__).resolve().parent
BUILTIN = HERE / "policies"
DRIVER = HERE / "driver.py"
TIMEOUT_S = 20
STARTUP_S = 60
MAX_REPLY = 1 << 20


class PolicyError(Exception):
    pass


def resolve(p: str | Path) -> Path:
    q = Path(p)
    if not q.suffix and (BUILTIN / f"{q}.py").exists():
        return BUILTIN / f"{q}.py"
    if not q.exists():
        raise FileNotFoundError(f"policy {p}: not a file or built-in ({', '.join(sorted(x.stem for x in BUILTIN.glob('*.py')))})")
    return q.resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class SandboxPolicy:
    """Context manager: `with SandboxPolicy(path) as pol: pol.select(...)`.

    One monotonic deadline per decision covers the request write and the reply; replies are bounded and validated;
    containers are named <name>-<hex> so a campaign's kill_leftovers reaps them after a runner crash.
    """

    def __init__(self, path: Path, timeout_s: float = TIMEOUT_S, name: str = "arlab-policy"):
        self.path, self.timeout_s, self.name = Path(path).resolve(), timeout_s, f"{name}-{uuid.uuid4().hex[:12]}"
        self.p = None
        self.buf = b""
        self.warm = False

    def __enter__(self):
        self.p = subprocess.Popen(
            ["docker", "run", "--rm", "-i", "--name", self.name, "--log-driver", "none",
             "--network", "none", "--read-only", "--memory", "2g", "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges", "--user", "1000:1000", "--tmpfs", "/tmp:size=64m",
             "-v", f"{self.path}:/p/policy.py:ro", "-v", f"{DRIVER}:/p/driver.py:ro", AGENT_IMAGE, "python3", "-I", "/p/driver.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for f in (self.p.stdin, self.p.stdout):
            os.set_blocking(f.fileno(), False)
        return self

    def select(self, nodes, budget_left, W):
        deadline = time.monotonic() + self.timeout_s + (0 if self.warm else STARTUP_S)
        req = (json.dumps({"nodes": nodes, "budget_left": budget_left, "W": W}) + "\n").encode()
        wfd, rfd = self.p.stdin.fileno(), self.p.stdout.fileno()
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close(kill=True)
                raise PolicyError(f"policy timed out ({self.timeout_s:.0f}s per decision)")
            r, w, _ = _select.select([rfd], [wfd] if req else [], [], left)
            try:
                if w:
                    req = req[os.write(wfd, req):]
                if r:
                    chunk = os.read(rfd, 1 << 16)
                    if not chunk:
                        raise PolicyError("policy process exited (import error or crash)")
                    self.buf += chunk
            except (BrokenPipeError, OSError) as e:
                raise PolicyError(f"policy process gone: {e}")
            if len(self.buf) > MAX_REPLY:
                self.close(kill=True)
                raise PolicyError(f"policy reply larger than {MAX_REPLY} bytes")
        line, self.buf = self.buf.split(b"\n", 1)
        self.warm = True
        try:
            out = json.loads(line)
        except ValueError:
            raise PolicyError(f"policy reply is not JSON: {line[:200]!r}")
        if isinstance(out, dict) and "error" in out:
            raise PolicyError(str(out["error"])[-1500:])
        sel = out.get("select") if isinstance(out, dict) else None
        if not isinstance(sel, list) or len(sel) > 4096 or not all(isinstance(x, str) and len(x) < 64 for x in sel):
            raise PolicyError(f"policy must return a list of node ids, got {str(out)[:200]}")
        return sel

    def close(self, kill: bool = False):
        if self.p and self.p.poll() is None:
            try:
                if kill:
                    raise OSError("kill")
                self.p.stdin.close()
                self.p.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                subprocess.run(["docker", "kill", self.name], capture_output=True, timeout=60)
                try:
                    self.p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()

    def __exit__(self, *exc):
        self.close()


class LocalPolicy:
    """In-process, for the trusted built-ins in unit tests only. Never used for LLM-written policy code."""

    def __init__(self, path: Path):
        path = Path(path).resolve()
        if path.parent != BUILTIN and "PYTEST_CURRENT_TEST" not in os.environ:
            raise PolicyError("LocalPolicy runs only built-in policies (or under pytest)")
        spec = importlib.util.spec_from_file_location(f"policy_{path.stem}", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def select(self, nodes, budget_left, W):
        try:
            return self.mod.select(json.loads(json.dumps(nodes)), budget_left, W)
        except Exception as e:  # same contract as the sandbox: any policy failure is a PolicyError
            raise PolicyError(repr(e))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
