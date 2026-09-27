"""Frozen tools for agentic-coding-small: read / write / edit / run / finish inside one task's working directory.

Every call counts as one step. Past the step cap StepLimit is raised; past the task's wall-clock deadline TimeLimit.
Paths for read/write/edit must stay inside the working directory; run() executes a shell command there
(timeout per command, output truncated).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

MAX_OUTPUT = 10_000


class StepLimit(Exception):
    pass


class TimeLimit(Exception):
    pass


class Tools:
    def __init__(self, root: Path, max_steps: int, deadline: float):
        self.root, self.max_steps, self.deadline = Path(root).resolve(), max_steps, deadline
        self.steps, self.done = 0, False
        self._pgids: list[int] = []

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise StepLimit(f"step cap {self.max_steps} reached")
        if time.monotonic() > self.deadline:
            raise TimeLimit("task wall-clock limit reached")

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path outside the working directory: {rel}")
        return p

    def read(self, path: str) -> str:
        self._tick()
        try:
            return self._path(path).read_text(errors="replace")[:MAX_OUTPUT * 2]
        except (OSError, ValueError) as e:
            return f"error: {e}"

    def write(self, path: str, content: str) -> str:
        self._tick()
        try:
            p = self._path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"wrote {len(content)} chars to {path}"
        except (OSError, ValueError) as e:
            return f"error: {e}"

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace exactly one occurrence of `old`."""
        self._tick()
        try:
            p = self._path(path)
            s = p.read_text()
        except (OSError, ValueError) as e:
            return f"error: {e}"
        n = s.count(old)
        if n != 1:
            return f"error: old text found {n} times (must be exactly 1)"
        p.write_text(s.replace(old, new))
        return f"edited {path}"

    def run(self, cmd: str, timeout: int = 60) -> str:
        """Run a shell command in the working directory. Returns 'exit=<rc>' plus combined output (truncated)."""
        self._tick()
        timeout = max(1, min(timeout, 120, int(self.deadline - time.monotonic()) + 1))
        p = subprocess.Popen(["bash", "-c", cmd], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, text=True, errors="replace",
                             env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.root), "PYTHONDONTWRITEBYTECODE": "1"})
        self._pgids.append(p.pid)
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            rc = f"timeout after {timeout}s"
        out = out or ""
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT // 2] + f"\n... [{len(out) - MAX_OUTPUT} chars cut] ...\n" + out[-MAX_OUTPUT // 2:]
        return f"exit={rc}\n{out}"

    def finish(self) -> None:
        self.done = True

    def kill_all(self) -> None:
        """Called by the harness when the task ends: stop anything its commands left running in the background."""
        for pg in self._pgids:
            try:
                os.killpg(pg, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
