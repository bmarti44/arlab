"""Process isolation helpers (frozen; the Landlock and subreaper code is the latent-arch pack's, reused as is).

Landlock (Linux >= 5.13; the GB10 kernel has ABI 7) restricts the calling process and all its descendants, even as
root: only the listed directories stay readable, only `rw` directories writable; TCP bind/connect (ABI >= 4), abstract
unix sockets and signals outside the sandbox (ABI >= 6) are denied, and so are ptrace-style accesses to processes
outside it (/proc/<parent>/mem, environ, fds). The RUN supervisor (harness.py) uses the other half: it is a subreaper,
inspects its sandboxed child's threads and descendants FROM OUTSIDE (/proc), and kills them all before it accepts a
world's adapter.
"""
from __future__ import annotations

import ctypes
import os
import signal
import struct
import time

SYS_CREATE, SYS_ADD, SYS_RESTRICT = 444, 445, 446      # landlock_* syscalls (same number on every architecture)
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1, 2, 4, 8
READ = EXECUTE | READ_FILE | READ_DIR
IOCTL_DEV = 1 << 15


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def lockdown(read_dirs: list, rw_dirs: list, dev_dirs: list, files: tuple = ()) -> int:
    """Apply a Landlock ruleset to this process; returns the ABI version. Raises OSError if unavailable.
    files: (path, writable) single files that stay readable (and writable if True), e.g. /dev/urandom and /dev/null,
    which importing torch needs."""
    libc = _libc()
    abi = libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    fs = (1 << 13) - 1 | (1 << 13 if abi >= 2 else 0) | (1 << 14 if abi >= 3 else 0) | (IOCTL_DEV if abi >= 5 else 0)
    attr = struct.pack("=QQQ", fs, 3 if abi >= 4 else 0, 3 if abi >= 6 else 0)[:8 if abi < 4 else 16 if abi < 6 else 24]
    buf = ctypes.create_string_buffer(attr, len(attr))
    rfd = libc.syscall(SYS_CREATE, buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if rfd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    dev = READ | WRITE_FILE | (IOCTL_DEV if abi >= 5 else 0)
    for dirs, access in ((read_dirs, READ), (rw_dirs, fs), (dev_dirs, dev)):
        for d in dirs:
            if not os.path.isdir(d):
                continue
            fd = os.open(d, os.O_PATH | os.O_CLOEXEC)
            rule = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
            r = libc.syscall(SYS_ADD, ctypes.c_int(rfd), ctypes.c_int(1), rule, ctypes.c_uint32(0))
            os.close(fd)
            if r != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule {d}")
    for f, writable in files:
        if os.path.exists(f) and not os.path.isdir(f):
            fd = os.open(f, os.O_PATH | os.O_CLOEXEC)
            rule = ctypes.create_string_buffer(struct.pack("=Qi", READ_FILE | (WRITE_FILE if writable else 0), fd), 12)
            r = libc.syscall(SYS_ADD, ctypes.c_int(rfd), ctypes.c_int(1), rule, ctypes.c_uint32(0))
            os.close(fd)
            if r != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule {f}")
    if libc.prctl(38, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:  # NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    if libc.syscall(SYS_RESTRICT, ctypes.c_int(rfd), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(rfd)
    return abi


def blocked(path: str, mode: str = "rb") -> bool:
    try:
        open(path, mode).close()
    except OSError:
        return True
    return False


def become_subreaper():
    """Orphaned descendants (double-forked daemons included) are re-parented to this process, not to init."""
    if _libc().prctl(36, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")


def _ppids() -> dict:
    """{pid: parent pid} of every process visible in /proc."""
    out = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                stat = open(f"/proc/{p}/stat").read()
            except OSError:
                continue
            out[int(p)] = int(stat.rsplit(")", 1)[1].split()[1])
    return out


def _children(pid: int) -> list[int]:
    return [p for p, pp in _ppids().items() if pp == pid]


def descendants(root: int) -> list[int]:
    """Every live process below `root` (children, grandchildren, ...)."""
    pp = _ppids()
    out, frontier = [], [root]
    while frontier:
        kids = [p for p, q in pp.items() if q in frontier]
        out += kids
        frontier = kids
    return sorted(out)


def threads(pid: int) -> set:
    """The kernel's thread ids of process `pid` (empty if it is gone)."""
    try:
        return set(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return set()


def cpu_s(pid: int) -> float | None:
    """CPU seconds (user + system, every thread including exited ones, plus reaped children) of process `pid`."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return sum(int(x) for x in fields[11:15]) / os.sysconf("SC_CLK_TCK")


def kill_descendants() -> int:
    """SIGKILL and reap every descendant of this (subreaper) process; returns how many were still there."""
    me, seen = os.getpid(), set()
    for _ in range(100):
        kids = _children(me)
        if not kids:
            break
        for k in kids:
            seen.add(k)
            try:
                os.kill(k, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.02)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
    return len(seen)
