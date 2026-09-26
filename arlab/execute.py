"""docker run wrappers (PLAN §3.2 mounts, §3.8 telemetry/watchdog): images, steps, services, kill."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import guards

OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|MemoryError|Killed process|std::bad_alloc")
TOTAL_MEM_GB = 119


def sh(*args, check=True, timeout=None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=check, text=True, capture_output=True, timeout=timeout, **kw)


# ---------------------------------------------------------------- images
def ensure_image(name: str, base: str, requirements: Path | None) -> tuple[str, str]:
    """Build arlab-<name>:<hash> = base + requirements.txt (base torch pinned). Returns (tag, image id)."""
    req = requirements.read_text() if requirements and requirements.exists() else ""
    tag = f"arlab-{name.strip('_').replace('_', '-')}:{hashlib.sha256((base + '\n' + req).encode()).hexdigest()[:12]}"
    if sh("docker", "image", "inspect", tag, check=False).returncode != 0:
        with tempfile.TemporaryDirectory() as td:
            df = f"FROM {base}\n"
            if req.strip():
                (Path(td) / "requirements.txt").write_text(req)
                df += ("COPY requirements.txt /tmp/req.txt\n"
                       "RUN pip list --format=freeze > /tmp/constraints.txt && "
                       "pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/req.txt\n")
            (Path(td) / "Dockerfile").write_text(df)
            r = sh("docker", "build", "-t", tag, td, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"image build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return tag, image_id(tag)


def image_id(tag: str) -> str:
    return sh("docker", "image", "inspect", "-f", "{{.Id}}", tag).stdout.strip()


# ---------------------------------------------------------------- steps
@dataclass
class Step:
    name: str                      # container name
    image: str
    command: str
    cidfile: Path
    log: Path
    timeout_s: int
    mounts: list[tuple[Path, str, bool]] = field(default_factory=list)  # (host, container, rw)
    user: str = "1000:1000"
    network: str = "none"
    gpu: bool = False
    env: dict[str, str] = field(default_factory=dict)
    telemetry: Path | None = None


@dataclass
class StepResult:
    rc: int
    wall_s: float
    timed_out: bool = False
    oom: bool = False
    contended: bool = False
    peak_mem_gb: float = 0.0
    gpu_temp_max: float | None = None
    foreign: list[str] = field(default_factory=list)


BASE_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/frozen:/arlab_lib",
            "HOME": "/tmp", "HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor", "TRITON_CACHE_DIR": "/cache/triton"}


def docker_cmd(s: Step) -> list[str]:
    cmd = ["docker", "run", "--rm", "--cidfile", str(s.cidfile), "--name", s.name, "--user", s.user,
           "--network", s.network, "--shm-size", "8g", "-w", "/tmp"]
    if s.gpu:
        cmd += ["--gpus", "all"]
    for host, cont, rw in s.mounts:
        cmd += ["-v", f"{host}:{cont}:{'rw' if rw else 'ro'}"]
    for k, v in {**BASE_ENV, **s.env}.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [s.image, "sh", "-c", s.command]


def _cid(cidfile: Path) -> str | None:
    try:
        c = cidfile.read_text().strip()
        return c or None
    except OSError:
        return None


def _memcg_gb(cid: str) -> float:
    try:
        return int(Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current").read_text()) / 1e9
    except (OSError, ValueError):
        return 0.0


def run_step(s: Step, own_cids: set[str], tick=None) -> StepResult:
    """Run one container to completion with timeout, memory watchdog, contention and telemetry sampling."""
    s.cidfile.parent.mkdir(parents=True, exist_ok=True)
    s.cidfile.unlink(missing_ok=True)
    t0 = time.monotonic()
    res = StepResult(rc=-1, wall_s=0.0)
    temps: list[float] = []
    with open(s.log, "ab") as logf:
        p = subprocess.Popen(docker_cmd(s), stdout=logf, stderr=subprocess.STDOUT)
        next_sample, killed = t0 + 1.0, False
        while p.poll() is None:
            time.sleep(0.25)
            now = time.monotonic()
            if tick:
                tick(now - t0, _cid(s.cidfile))
            if now - t0 > s.timeout_s and not killed:
                res.timed_out, killed = True, True
                kill_cid(_cid(s.cidfile))
            if now >= next_sample:
                next_sample = now + 5.0
                cid = _cid(s.cidfile)
                avail = guards.mem_available_gb()
                sample = {"t": round(now - t0, 1), "mem_available_gb": round(avail, 2)}
                used = _memcg_gb(cid) if cid else 0.0
                if s.gpu:
                    procs = guards.gpu_procs()
                    mine = own_cids | ({cid} if cid else set())
                    used += sum(pr["used_gb"] for pr in procs if guards.container_of(pr["pid"]) == cid)
                    foreign = [f"{pr['pid']} {pr['name']}" for pr in procs if guards.container_of(pr["pid"]) not in mine]
                    if foreign:
                        res.contended = True
                        res.foreign = sorted(set(res.foreign) | set(foreign))
                    t = guards.gpu_temp()
                    if t is not None:
                        temps.append(t)
                    sample.update(gpu_procs=len(procs), foreign=foreign, gpu_temp=t)
                sample["container_gb"] = round(used, 2)
                res.peak_mem_gb = max(res.peak_mem_gb, used)
                if s.telemetry:
                    with open(s.telemetry, "a") as tf:
                        tf.write(json.dumps(sample) + "\n")
                if avail < guards.OOM_FLOOR_GB and not killed:
                    res.oom, killed = True, True
                    kill_cid(cid)
        res.rc = p.returncode
    res.wall_s = time.monotonic() - t0
    res.gpu_temp_max = max(temps) if temps else None
    if res.rc != 0 and not res.timed_out:
        tail = Path(s.log).read_bytes()[-20000:].decode(errors="replace")
        if OOM_RE.search(tail) or (res.rc == 137 and not killed):
            res.oom = True
    return res


def kill_cid(cid: str | None):
    """docker kill a container arlab started (by the ID in its cidfile). Never used on foreign containers."""
    if cid:
        sh("docker", "kill", cid, check=False, timeout=60)


def kill_leftovers(campaign: Path):
    for cf in list(campaign.glob("**/*.cid")):
        cid = _cid(cf)
        if cid and sh("docker", "inspect", cid, check=False).returncode == 0:
            kill_cid(cid)
            sh("docker", "rm", "-f", cid, check=False, timeout=60)


# ---------------------------------------------------------------- services
def service_name(prefix: str, svc) -> str:
    return f"{prefix}-svc-{svc.name}"


def start_service(svc, prefix: str, network: str, cidfile: Path, log: Path, hf: Path) -> str:
    """Start a pack service on the campaign network; returns container id once healthy."""
    if sh("docker", "network", "inspect", network, check=False).returncode != 0:
        sh("docker", "network", "create", network)
    name = service_name(prefix, svc)
    sh("docker", "rm", "-f", name, check=False)
    command = svc.command
    if "vllm" in (svc.image + command):
        command += f" --gpu-memory-utilization {svc.mem_gb / TOTAL_MEM_GB:.2f}"
        if svc.max_model_len:
            command += f" --max-model-len {svc.max_model_len}"
    cmd = ["docker", "run", "-d", "--cidfile", str(cidfile), "--name", name, "--network", network,
           "--network-alias", svc.name, "--ipc=host", "-v", f"{hf}:/hf:ro", "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1"]
    cmd += ["--gpus", "all"] if svc.gpu else []
    for k, v in svc.env.items():
        cmd += ["-e", f"{k}={v}"]
    for v in svc.volumes:
        cmd += ["-v", v]
    cidfile.unlink(missing_ok=True)
    cid = sh(*cmd, svc.image, "sh", "-c", command).stdout.strip()
    ip = service_ip(cid, network)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{ip}:{svc.port}{svc.health_url}", timeout=5) as r:
                if r.status == 200:
                    return cid
        except OSError:
            pass
        if sh("docker", "inspect", "-f", "{{.State.Running}}", cid, check=False).stdout.strip() != "true":
            break
        time.sleep(5)
    logs = sh("docker", "logs", "--tail", "60", cid, check=False)
    log.write_text(logs.stdout + logs.stderr)
    kill_cid(cid)
    raise RuntimeError(f"service {svc.name} did not become healthy; see {log}")


def service_ip(cid: str, network: str) -> str:
    fmt = "{{(index .NetworkSettings.Networks \"%s\").IPAddress}}" % network
    return sh("docker", "inspect", "-f", fmt, cid).stdout.strip()


def service_tokens(cid: str, network: str, port: int) -> float:
    """vLLM prompt + generation token counters from /metrics."""
    with urllib.request.urlopen(f"http://{service_ip(cid, network)}:{port}/metrics", timeout=10) as r:
        text = r.read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith(("vllm:prompt_tokens_total", "vllm:generation_tokens_total")):
            total += float(line.rsplit(" ", 1)[1])
    return total


def stop_service(cid: str | None, network: str | None = None):
    if cid:
        sh("docker", "rm", "-f", cid, check=False, timeout=120)
    if network:
        sh("docker", "network", "rm", network, check=False)
