"""The sandboxed controller process (one per replicate and budget level), started by harness.py with `python -s -B`
and a minimal environment (no PYTHONPATH, PYTHONHASHSEED=0 so set iteration is deterministic).

Order: load the train cache for fit() -> lock down with Landlock (read-only: the Python install and --work; no
writes, no TCP, no abstract-socket or signal IPC outside the sandbox; no new processes or threads) -> self-check (a
cache file and /proc/<ppid>/mem must NOT open, /tmp must not be writable) -> report to the harness -> import
/work/controller.py -> Controller(cfg) -> fit(train) -> serve solve() requests. JSON lines on the original stdout;
anything the controller prints goes to stderr (the run log).
"""
import argparse
import ctypes
import json
import os
import resource
import struct
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402,F401  (imported before the lockdown on purpose)
import collections, functools, heapq, itertools, math, random, statistics  # noqa: E402,F401,E401

import replay  # noqa: E402
import ttc_api  # noqa: E402

SYS_CREATE, SYS_ADD, SYS_RESTRICT = 444, 445, 446      # landlock_* (same number on every architecture)
READ = 1 | 4 | 8                                        # EXECUTE | READ_FILE | READ_DIR
READ_DIRS = ["/usr", "/lib", "/lib64", "/etc"]


def lockdown(read_dirs: list) -> int:
    """Apply a Landlock ruleset to this process; returns the ABI version. Raises OSError if Landlock is unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    fs = (1 << 13) - 1 | (1 << 13 if abi >= 2 else 0) | (1 << 14 if abi >= 3 else 0) | (1 << 15 if abi >= 5 else 0)
    attr = struct.pack("=QQQ", fs, 3 if abi >= 4 else 0, 3 if abi >= 6 else 0)[:8 if abi < 4 else 16 if abi < 6 else 24]
    buf = ctypes.create_string_buffer(attr, len(attr))
    rfd = libc.syscall(SYS_CREATE, buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if rfd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    for d in read_dirs:
        if not os.path.isdir(d):
            continue
        fd = os.open(d, os.O_PATH | os.O_CLOEXEC)
        rule = ctypes.create_string_buffer(struct.pack("=Qi", READ, fd), 12)
        r = libc.syscall(SYS_ADD, ctypes.c_int(rfd), ctypes.c_int(1), rule, ctypes.c_uint32(0))
        os.close(fd)
        if r != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule {d}")
    if libc.prctl(38, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:  # NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    if libc.syscall(SYS_RESTRICT, ctypes.c_int(rfd), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(rfd)
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))   # no fork/threads from here on (ignored for root in TESTS)
    return abi


def blocked(path: str, mode: str = "rb") -> bool:
    try:
        open(path, mode).close()
    except OSError:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--deny", action="append", default=[], help="files the self-check must fail to open")
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer

    def send(obj):
        proto.write(json.dumps(obj) + "\n")

    def recv():
        line = stdin.readline()
        if not line:
            sys.exit(0)
        return json.loads(line)

    train = replay.train_problems(replay.load_cache(a.train)) if a.train and os.path.exists(a.train) else []
    try:
        abi = lockdown(READ_DIRS + [a.work])
    except OSError as e:
        send({"op": "hello", "sandbox": {"ok": False, "error": repr(e)}})
        return
    checks = {"deny": all(blocked(p) for p in a.deny) and bool(a.deny), "procmem": blocked(f"/proc/{os.getppid()}/mem"),
              "write": blocked("/tmp/.ttc_sandbox_probe", "wb")}
    send({"op": "hello", "sandbox": {"ok": all(checks.values()), "landlock_abi": abi, **checks}})
    if not all(checks.values()):
        return
    try:
        cfg = recv()["cfg"]
        sys.path.insert(0, a.work)
        import controller
        t0 = time.monotonic()
        ctl = controller.Controller(dict(cfg))
        if hasattr(ctl, "fit"):
            ctl.fit(train)
        del train
        send({"op": "ready", "fit_s": round(time.monotonic() - t0, 3)})

        def rpc(msg):
            send(msg)
            return recv()

        while True:
            msg = recv()
            if msg["op"] != "solve":
                return
            try:
                label = ctl.solve(ttc_api.Problem(rpc, msg))
            except ttc_api.BudgetExhausted:
                label = None
            if label is not None and not isinstance(label, str):
                raise TypeError(f"solve() must return a label string or None, got {type(label).__name__}")
            send({"op": "answer", "label": label})
    except Exception:
        send({"op": "error", "msg": traceback.format_exc()[-3000:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
