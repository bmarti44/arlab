"""Trusted side of latent-arch EVALUATE (importable by the pack tests): the client for the sandboxed model worker
(/frozen/worker.py) and the scoring math. Labels, scores and the result file never leave this process; the worker
only ever receives token ids (as bytes) and returns logits through a memfd.
"""
from __future__ import annotations

import json
import mmap
import os
import select
import shutil
import signal
import site
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import common
from common import VOCAB

WORKER = os.path.join(os.path.dirname(os.path.abspath(common.__file__)), "worker.py")
STARTUP_S, BUILD_S, FWD_S, MAX_LINE = 180.0, 120.0, 120.0, 1 << 20


class WorkerFailure(Exception):
    pass


class Worker:
    """One sandboxed model process: build(config) once, then forward(x) -> float32 logits (copied out of the memfd)."""

    def __init__(self, work: str, ckpt: str, device: str, deny: list, max_floats: int, worker: str = WORKER):
        self.scratch = tempfile.mkdtemp(prefix="la-worker-")
        os.chmod(self.scratch, 0o777)
        self.size = max(4 * max_floats, mmap.PAGESIZE)
        self.fd = os.memfd_create("la-logits")
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "HF_HOME")}
        env.update(HOME=self.scratch, TMPDIR=self.scratch, PYTHONDONTWRITEBYTECODE="1", PYTHONUSERBASE=site.USER_BASE)
        cmd = [sys.executable, "-B", worker, "--work", work, "--ckpt", ckpt, "--device", device,
               "--memfd", str(self.fd), "--memfd-size", str(self.size), "--scratch", self.scratch]
        cmd += [x for d in deny for x in ("--deny", d)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(self.fd,),
                                  start_new_session=True, cwd=self.scratch, env=env)
        self.buf = b""
        self.hello = self._recv(time.monotonic() + STARTUP_S, "startup")

    def _recv(self, deadline: float, what: str) -> dict:
        fd = self.p.stdout.fileno()
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise WorkerFailure(f"timeout: {what}")
            if select.select([fd], [], [], left)[0]:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    raise WorkerFailure(f"model worker exited during {what} (rc={self.p.wait()})")
                self.buf += chunk
                if len(self.buf) > MAX_LINE:
                    raise WorkerFailure("message too long")
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            msg = json.loads(line)
        except ValueError:
            raise WorkerFailure(f"malformed message during {what}")
        if msg.get("op") == "error":
            raise WorkerFailure(f"surface raised during {what}:\n{msg.get('msg')}")
        return msg

    def _send(self, data: bytes):
        try:
            self.p.stdin.write(data)
            self.p.stdin.flush()
        except BrokenPipeError:
            raise WorkerFailure(f"model worker exited (rc={self.p.poll()})")

    def build(self, config: dict, want_hash: str) -> dict:
        self._send((json.dumps({"op": "build", "config": config}) + "\n").encode())
        msg = self._recv(time.monotonic() + BUILD_S, "make_model(config) + checkpoint load")
        if msg.get("op") != "ready":
            raise WorkerFailure(f"unexpected message {str(msg)[:200]}")
        self._check_state(msg, want_hash, "after make_model() + checkpoint load + eval()")
        return msg

    def check(self, want_hash: str, when: str):
        """The model's tensors must still be exactly the checkpoint's, and no unregistered tensor state may exist."""
        self._send((json.dumps({"op": "check"}) + "\n").encode())
        self._check_state(self._recv(time.monotonic() + BUILD_S, "state check"), want_hash, when)

    @staticmethod
    def _check_state(msg: dict, want_hash: str, when: str):
        if msg.get("tensors_hash") != want_hash:
            raise WorkerFailure(f"model parameters/buffers differ from the checkpoint {when}")
        if msg.get("hidden") != []:
            raise WorkerFailure(f"unregistered tensor state {when}: {str(msg.get('hidden'))[:500]}")

    def forward(self, x: np.ndarray, last: bool = False) -> tuple[torch.Tensor, int, list]:
        x = np.ascontiguousarray(x, dtype=np.int32)
        B, T = x.shape
        self._send((json.dumps({"op": "fwd", "shape": [B, T], "last": last}) + "\n").encode() + x.tobytes())
        msg = self._recv(time.monotonic() + FWD_S, f"forward of {B}x{T} tokens")
        want = [B, 1 if last else T, VOCAB]
        if msg.get("op") != "out" or msg.get("shape") != want:
            raise WorkerFailure(f"bad reply {str(msg)[:200]} (expected logits {want})")
        n = B * want[1] * VOCAB
        lg = torch.from_numpy(np.frombuffer(self.mm, dtype=np.float32, count=n).copy()).view(*want)
        flops, bad = msg.get("flops"), msg.get("bad_ops")
        if type(flops) is not int or flops < 0 or not isinstance(bad, list):
            raise WorkerFailure("bad meter report")
        return lg, flops, [str(b) for b in bad]

    def close(self):
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.p.wait()
        self.mm.close()
        os.close(self.fd)
        shutil.rmtree(self.scratch, ignore_errors=True)


def nll_sum(lg: torch.Tensor, y: torch.Tensor, tb: torch.Tensor) -> tuple[float, int, bool]:
    """Frozen cross-entropy from raw logits: (nats over byte-bearing targets, bytes, all finite)."""
    lp = torch.log_softmax(lg.float(), dim=-1)
    nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    b = tb[y]
    return float((nll * (b > 0)).sum()), int(b.sum()), bool(torch.isfinite(nll[b > 0]).all())


def accuracy_report(pred: np.ndarray, d: dict, idx: np.ndarray) -> dict:
    """Scores for the selected items idx of programs.npz d. pred aligned with idx."""
    correct = (pred == d["answer"][idx]).astype(np.float64)
    group, k = d["group"][idx], d["k"][idx]
    pos = {int(j): n for n, j in enumerate(idx)}
    m = {}
    for name, g in (("acc_id", 0), ("acc_depth", 1), ("acc_ext", 2)):
        sel = group == g
        m[name] = float(correct[sel].mean()) if sel.any() else None
    for kk in sorted(set(k[group < 3].tolist())):
        m[f"acc_k{kk}"] = float(correct[(k == kk) & (group < 3)].mean())
    pairs = [(n, pos[int(d["cf_of"][j])]) for n, j in enumerate(idx) if group[n] == 3 and int(d["cf_of"][j]) in pos]
    m["cf_both"] = float(np.mean([correct[a] * correct[b] for a, b in pairs])) if pairs else None
    main = group < 2
    val = d["value"][idx][main]
    for h in ("h_last_const", "h_own_const", "h_root"):
        m["floor_" + h[2:]] = float((d[h][idx][main] == val).mean())
    m["floor_modal"] = float(np.bincount(val, minlength=100).max() / max(len(val), 1))
    m["accuracy"] = 0.5 * m["acc_id"] + 0.5 * m["acc_depth"]
    ids = {0: "id", 1: "depth"}
    items = {f"{ids[int(g)]}{int(j):05d}": float(c) for j, g, c in zip(idx, group, correct) if g < 2}
    return {"metrics": m, "items": items}
