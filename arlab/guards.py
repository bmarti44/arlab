"""GPU/memory/disk gates, the campaign GPU lock and the reservations-file reader (PLAN §3.8).

arlab only READS ~/.gb10-gpu.reservations; it never writes it.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "arlab"
RESERVATIONS = Path.home() / ".gb10-gpu.reservations"
POLL_S = 60
OOM_FLOOR_GB = 6
MARGIN_GB = 8


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("no MemAvailable")


def disk_free_gb(path: Path = Path.home()) -> float:
    return shutil.disk_usage(path).free / 1e9


def fast_cpus() -> str | None:
    """The fastest cores (GB10: 10× Cortex-X925 @3.9 GHz of 20) as a cpuset, so wall-clock guards like train_s don't
    depend on whether a run landed on an X925 or an A725 core. None if all cores are equal or unknown."""
    freqs = {}
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        freqs[int(f.parent.parent.name[3:])] = int(f.read_text())
    if not freqs or len(set(freqs.values())) == 1:
        return None
    top = max(freqs.values())
    return ",".join(str(c) for c in sorted(freqs) if freqs[c] == top)


def gpu_procs() -> list[dict]:
    """[{pid, name, used_gb}] from nvidia-smi; [] if nvidia-smi fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            used = float(parts[2]) / 1024 if parts[2].replace(".", "").isdigit() else 0.0
            procs.append({"pid": int(parts[0]), "name": parts[1], "used_gb": used})
    return procs


def gpu_temp() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def container_of(pid: int) -> str | None:
    try:
        m = re.search(r"docker-([0-9a-f]{64})\.scope", Path(f"/proc/{pid}/cgroup").read_text())
    except OSError:
        return None
    return m.group(1) if m else None


def foreign_gpu_procs(own_cids: set[str]) -> list[dict]:
    return [p for p in gpu_procs() if container_of(p["pid"]) not in own_cids]


def reserved_unmaterialized_gb(procs: list[dict] | None = None) -> float:
    """Sum over live reservations of max(peak_gb - current usage, 0)."""
    if not RESERVATIONS.exists():
        return 0.0
    used = {p["pid"]: p["used_gb"] for p in (gpu_procs() if procs is None else procs)}
    total = 0.0
    for line in RESERVATIONS.read_text().splitlines():
        try:
            r = json.loads(line)
            pid, peak = int(r["pid"]), float(r["peak_gb"])
        except (ValueError, KeyError, TypeError):
            continue
        if Path(f"/proc/{pid}").exists():
            total += max(peak - used.get(pid, 0.0), 0.0)
    return total


def blockers(need_gb: float, own_cids: set[str], gpu: bool) -> list[str]:
    """Why we may not start now ([] = free). CPU packs only check memory."""
    why = []
    procs = gpu_procs() if gpu else []
    if gpu:
        why += [f"foreign GPU process {p['pid']} {p['name']}" for p in procs if container_of(p["pid"]) not in own_cids]
    reserved = reserved_unmaterialized_gb(procs) if gpu else 0.0
    avail = mem_available_gb()
    if avail < need_gb + reserved + MARGIN_GB:
        why.append(f"MemAvailable {avail:.0f} GB < need {need_gb:.0f} + reserved {reserved:.0f} + {MARGIN_GB} GB")
    return why


def wait_for_free(need_gb: float, own_cids: set[str], gpu: bool, on_wait=None) -> float:
    """Poll every POLL_S with no deadline. Returns seconds waited."""
    t0 = time.monotonic()
    while True:
        why = blockers(need_gb, own_cids, gpu)
        if not why:
            return time.monotonic() - t0
        if on_wait:
            on_wait(why, t0)
        time.sleep(POLL_S)


class FileLock:
    """Exclusive flock; blocking or not."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fd = None

    def acquire(self, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            return False
        os.ftruncate(self.fd, 0)
        os.write(self.fd, str(os.getpid()).encode())
        return True

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def gpu_lock() -> FileLock:
    return FileLock(CACHE / "gpu.lock")
