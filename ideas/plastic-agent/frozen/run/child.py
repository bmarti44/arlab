"""The sandboxed surface process: ONE per world, started by harness.py with `python -s -B`, a minimal environment (no
PYTHONPATH, CUDA hidden, single-threaded BLAS/OpenMP) and its own session. Frozen code, but it shares its interpreter
with the surface, so nothing in it is trusted: every check that matters is repeated by the supervisor from outside.

Order: Landlock lockdown (readable: the Python install, /usr /lib /etc /opt /proc /sys, this directory, --work and
/dev/{urandom,random}; writable: only /dev/null and --scratch, a fresh empty directory per world that the
supervisor deletes afterwards; no other /dev file, so no GPU; no TCP; no signals or ptrace-style access outside the sandbox) -> self-check (the
--deny files, e.g. other worlds' transcripts, the model snapshot and the replay rows, and /proc/<parent>/mem must not
open; /tmp must not be writable) -> hello -> wait for "go" (this world's transcript and tool names, the supervisor's
CLOCK_MONOTONIC deadline, the seed) -> disable every Python-level way to start a thread or a process -> import
/work/adapt.py -> adapt(transcript, tool_names, gen, train) with the RPC stubs of surface_api.py -> "done" with the
returned adapter id. The supervisor then kills this process group, so nothing (threads, children, atexit callbacks)
runs after adapt() returns. JSON lines on the original stdout; anything the surface prints goes to stderr (the run log).
"""
import argparse
import json
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sandbox import blocked, lockdown  # noqa: E402
import surface_api  # noqa: E402

READ_DIRS = ["/usr", "/lib", "/lib64", "/etc", "/opt", "/proc", "/sys"]
DEV_FILES = (("/dev/urandom", False), ("/dev/random", False), ("/dev/null", True))      # importing torch needs these


def python_dirs() -> list:
    import site
    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    for d in site.getsitepackages():
        dirs.add(d)
    return sorted(d for d in dirs if d and os.path.isdir(d))


def _forbidden(*args, **kwargs):
    raise RuntimeError("threads and child processes are not allowed in adapt(): all model work goes through gen / train")


def forbid_threads_and_processes():
    """Before the surface is imported: every Python-level way to start a thread or a process raises. (A thread or
    process started by other means is seen by the supervisor from outside and invalidates the run.)"""
    import _posixsubprocess
    import _thread
    import subprocess
    import threading
    for mod, names in ((_thread, ("start_new_thread", "start_new", "start_joinable_thread")),
                       (threading, ("_start_new_thread", "_start_joinable_thread")),
                       (_posixsubprocess, ("fork_exec",)), (subprocess, ("_fork_exec",)),
                       (os, ("fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen"))):
        for n in names:
            if hasattr(mod, n):
                setattr(mod, n, _forbidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--scratch", required=True, help="this world's empty scratch dir (the only writable path)")
    ap.add_argument("--deny", action="append", default=[], help="files the self-check must fail to open")
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = sys.stdin

    def send(obj):
        proto.write(json.dumps(obj) + "\n")

    def recv():
        line = stdin.readline()
        if not line:
            os._exit(0)
        return json.loads(line)

    try:
        abi = lockdown(READ_DIRS + python_dirs() + [HERE, a.work], [a.scratch], [], DEV_FILES)
    except OSError as e:
        send({"op": "hello", "sandbox": {"ok": False, "error": repr(e)}})
        return
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"] = a.scratch
    checks = {"deny": bool(a.deny) and all(blocked(p) for p in a.deny), "procmem": blocked(f"/proc/{os.getppid()}/mem"),
              "write": blocked("/tmp/.pa_sandbox_probe", "wb")}
    send({"op": "hello", "sandbox": {"ok": all(checks.values()), "landlock_abi": abi, **checks}})
    if not all(checks.values()):
        return
    try:
        go = recv()
        random.seed(go["seed"])
        client = surface_api._Client(send, recv, go["gen_left"], go["train_left"])
        gen = surface_api.Gen(client, go["tools"], go["transcript"], go["deadline"])
        train = surface_api.Train(client)
        transcript, tools = go["transcript"], list(go["tools"])
        del go
        forbid_threads_and_processes()
        sys.path.insert(0, a.work)
        try:
            import adapt as surface          # the editable surface (the supervisor's clock is already running)
            res = surface.adapt(transcript, tools, gen, train)
        except surface_api.BudgetExceeded as e:
            send({"op": "done", "adapter": "last", "budget_hit": str(e)[:300]})
        else:
            if res is not None and not isinstance(res, surface_api.AdapterRef):
                raise TypeError(f"adapt() must return an adapter produced by train() in this world (or None), "
                                f"got {type(res).__name__}")
            send({"op": "done", "adapter": None if res is None else res.id, "budget_hit": None})
        recv()                               # the supervisor kills this process group; nothing runs after done
    except Exception:
        send({"op": "error", "msg": traceback.format_exc()[-3000:]})
        os._exit(1)


if __name__ == "__main__":
    main()
