"""Campaign lifecycle (PLAN §3.3): CHECK → PREPARE → TESTS → SEAL → PROBE → CALIBRATE → LOOP → FINALIZE → REPORT.

Idempotent: re-running resumes. runs/<id>/record.json, calib/*/result.json and holdout/*/result.json are the truth.
"""
from __future__ import annotations

import hashlib
import json
import re
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import execute, guards, stats
from .agent import AuthError, CodexBackend, ScriptedBackend, codex_login_ok, AGENT_IMAGE
from .pack import ARLAB_ROOT, compile_error, load_pack, seal, seal_hash, surface_files, validate_dir
from .record import counted, load_records, read_json, write_json, write_results_tsv

RUNS = Path(os.environ.get("ARLAB_RUNS", Path.home() / "arlab-runs"))
DATA = Path(os.environ.get("ARLAB_DATA", Path.home() / "arlab-data"))
HF = Path.home() / ".cache" / "huggingface"
DISK_FLOOR_GB = 60
GIT_ENV = {"GIT_AUTHOR_NAME": "arlab", "GIT_AUTHOR_EMAIL": "arlab@localhost", "GIT_COMMITTER_NAME": "arlab",
           "GIT_COMMITTER_EMAIL": "arlab@localhost", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


class ConfigError(Exception):
    """A planned stop: the pack cannot be run as configured."""


class TamperError(Exception):
    """sealed/, data or image differ from the stored values: resume refuses."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def debug_kill(point: str):
    """Test hook: ARLAB_DEBUG_KILL=<point> makes the runner SIGKILL itself there (≡ kill -9)."""
    if os.environ.get("ARLAB_DEBUG_KILL") == point:
        os.kill(os.getpid(), signal.SIGKILL)


def arlab_commit() -> str:
    """Read ~/arlab's HEAD without running git there."""
    g = ARLAB_ROOT / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (g / ref).exists():
            return (g / ref).read_text().strip()
        for line in (g / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


class Campaign:
    def __init__(self, pack_dir: Path, tag: str, script: Path | None = None, runs_root: Path | None = None):
        self.pack_dir = Path(pack_dir).resolve()
        self.name, self.tag = self.pack_dir.name, tag
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", tag):
            raise ValueError(f"tag {tag!r}: use letters, digits, '-' or '_' (one path component)")
        self.dir = (runs_root or RUNS) / self.name / tag
        self.sealed, self.work = self.dir / "sealed", self.dir / "work"
        self.prefix = f"arlab-{self.name}-{tag}"
        self.script = script
        self.state: dict = read_json(self.dir / "state.json", {}) or {}
        self.pack = None
        self.backend = None
        self.services: dict[str, str] = {}
        self.network: str | None = None
        self.lock = None
        self.gpu_lock = None

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str):
        line = f"{now()} {msg}"
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "runner.log", "a") as f:
            f.write(line + "\n")
        if sys.stderr.isatty():
            print(line, file=sys.stderr, flush=True)

    def save(self, **kw):
        self.state.update(kw)
        write_json(self.dir / "state.json", self.state)

    def git(self, *args, cwd=None) -> str:
        r = subprocess.run(["git", *args], cwd=cwd or self.work, env={**os.environ, **GIT_ENV}, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr}")
        return r.stdout.strip()

    def export(self, commit: str, dest: Path) -> Path:
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", commit], cwd=self.work, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
        return dest

    def own_cids(self) -> set[str]:
        return {c for c in (p.read_text().strip() for p in self.dir.glob("**/*.cid")
                            if p.is_file() and not execute.UNTRUSTED_DIRS & set(p.relative_to(self.dir).parts[:-1])) if c}

    @property
    def data(self) -> Path:
        return Path(self.state["data_dir"])

    def need_gb(self) -> float:
        m = self.pack.run.mem_gb
        return float(m) if m != "auto" else float(self.state.get("mem_gb") or 40)

    def _on_wait(self, why, t0):
        self.save(waiting=f"waiting for GPU/memory since {datetime.fromtimestamp(time.time() - (time.monotonic() - t0), timezone.utc).isoformat(timespec='seconds')}; blocked by: {'; '.join(why)}")

    # ------------------------------------------------------------ setup: CHECK / PREPARE / TESTS / SEAL
    def data_hash(self, root: Path, image_id: str) -> str:
        from .pack import hash_paths
        extra_src = [root / "frozen" / "prepare"] if (root / "frozen" / "prepare").exists() else []  # keeps older hashes stable
        return hash_paths([root / "frozen" / "run", root / "requirements.txt"] + extra_src, extra=self.pack.prepare.command + "\n" + image_id)

    def prepare_data(self, root: Path, image: str, image_id: str) -> tuple[str, Path]:
        dh = self.data_hash(root, image_id)
        ddir = DATA / self.name / dh[:16]
        if (ddir / "MANIFEST.json").exists():
            return dh, ddir
        for stale in ddir.parent.glob(ddir.name + ".tmp*"):  # left by a killed PREPARE whose runner is gone
            if not Path(f"/proc/{stale.name.rsplit('.tmp', 1)[1]}").exists():
                subprocess.run(["chmod", "-R", "u+w", str(stale)], check=False)
                shutil.rmtree(stale, ignore_errors=True)
        tmp = ddir.with_name(ddir.name + f".tmp{os.getpid()}")
        tmp.mkdir(parents=True)
        self.log(f"PREPARE into {ddir}")
        step = execute.Step(name=f"{self.prefix}-prepare", image=image, command=self.pack.prepare.command,
                            cidfile=self.dir / "prepare.cid", log=self.dir / "prepare.log", timeout_s=self.pack.prepare.timeout_s,
                            mounts=[(root / "frozen" / "run", "/frozen", False), (tmp, "/data", True), (HF, "/hf", True)]
                            + ([(root / "frozen" / "prepare", "/prepare", False)] if (root / "frozen" / "prepare").exists() else []),
                            network="bridge", env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
        r = execute.run_step(step, set())
        if r.rc != 0:
            raise ConfigError(f"PREPARE failed (rc={r.rc}, timeout={r.timed_out}); see {self.dir / 'prepare.log'}")
        files = {p.relative_to(tmp).as_posix(): file_hash(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        write_json(tmp / "MANIFEST.json", {"data_hash": dh, "files": files, "created": now()})
        for p in sorted(tmp.rglob("*"), reverse=True):
            os.chmod(p, (0o500 if "private" in p.relative_to(tmp).parts else 0o555) if p.is_dir() else 0o444)
        os.replace(tmp, ddir)
        return dh, ddir

    def verify_data(self, ddir: Path) -> None:
        man = read_json(ddir / "MANIFEST.json")
        if not man or file_hash(ddir / "MANIFEST.json") != self.state["manifest_hash"]:
            raise TamperError(f"data MANIFEST changed: {ddir}")
        actual = {p.relative_to(ddir).as_posix(): file_hash(p) for p in sorted(ddir.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
        if actual != man["files"]:
            bad = sorted(k for k in set(actual) | set(man["files"]) if actual.get(k) != man["files"].get(k))
            raise TamperError(f"data files changed: {bad[:5]}")

    def run_tests(self, root: Path, image: str, ddir: Path) -> None:
        if not list((root / "tests").glob("test_*.py")):
            return
        step = execute.Step(name=f"{self.prefix}-tests", image=image, cidfile=self.dir / "tests.cid", log=self.dir / "tests.log",
                            command="python -m pytest -q -p no:cacheprovider /pack/tests", timeout_s=1800, user="0:0",
                            mounts=[(root, "/pack", False), (root / "frozen" / "run", "/frozen", False), (root / "frozen" / "eval", "/eval", False),
                                    (root / "arlab_lib", "/arlab_lib", False), (root / "surface", "/work", False), (ddir, "/data", False), (HF, "/hf", False)])
        r = execute.run_step(step, set())
        if r.rc != 0:
            raise ConfigError(f"pack tests failed; see {self.dir / 'tests.log'}")

    def setup(self, static_only: bool = False) -> None:
        """First run: CHECK, PREPARE, TESTS, SEAL. Resume: verify sealed/, data and image."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.sealed.exists():
            self.pack = load_pack(self.sealed)
            if seal_hash(self.sealed) != self.state.get("seal_hash"):
                raise TamperError("sealed/ changed since SEAL")
            self.verify_data(self.data)
            if execute.image_id(self.state["image"]) != self.state["image_id"]:
                raise TamperError("pack image changed")
            return
        errs = validate_dir(self.pack_dir)
        if errs:
            raise ConfigError("CHECK: " + "; ".join(errs))
        tmp = self.dir / "sealed.tmp"
        if tmp.exists():
            subprocess.run(["chmod", "-R", "u+w", str(tmp)], check=True)
            shutil.rmtree(tmp)
        shash = seal(self.pack_dir, tmp)
        self.pack = load_pack(tmp)
        for f in surface_files(tmp / "surface"):
            if f.endswith(".py") and compile_error(tmp / "surface" / f):
                raise ConfigError(f"CHECK: surface does not compile: {compile_error(tmp / 'surface' / f)}")
        self.log("CHECK ok; building image")
        image, iid = execute.ensure_image(self.name, self.pack.image.base, tmp / "requirements.txt")
        dh, ddir = self.prepare_data(tmp, image, iid)
        self.log(f"TESTS (data {ddir.name})")
        self.run_tests(tmp, image, ddir)
        os.replace(tmp, self.sealed)
        codex_v = subprocess.run(["docker", "run", "--rm", AGENT_IMAGE, "codex", "--version"], capture_output=True, text=True)
        self.save(phase="sealed", created=now(), seal_hash=shash, data_hash=dh, data_dir=str(ddir), image=image, image_id=iid,
                  manifest_hash=file_hash(ddir / "MANIFEST.json"), uv_lock_hash=file_hash(ARLAB_ROOT / "uv.lock"),
                  codex_version=codex_v.stdout.strip() or "unavailable", arlab_commit=arlab_commit(), pack_dir=str(self.pack_dir))
        self.log(f"SEAL {shash[:12]}")

    def init_work(self):
        if (self.work / ".git").exists():
            return
        shutil.copytree(self.sealed / "surface", self.work)
        subprocess.run(["chmod", "-R", "u+w", str(self.work)], check=True)
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "baseline")
        self.git("tag", "baseline")
        self.save(baseline_commit=self.git("rev-parse", "HEAD"))

    # ------------------------------------------------------------ services
    def start_services(self):
        if not self.pack.services:
            return
        self.network = f"{self.prefix}-net"
        (self.dir / "services").mkdir(exist_ok=True)
        for svc in self.pack.services:
            self.save(waiting=None)
            w = guards.wait_for_free(svc.mem_gb, self.own_cids(), svc.gpu, self._on_wait)
            self.log(f"starting service {svc.name} (waited {w:.0f}s)")
            cid = execute.start_service(svc, self.prefix, self.network, self.dir / "services" / f"{svc.name}.cid",
                                        self.dir / "services" / f"{svc.name}.log", HF)
            self.services[svc.name] = cid
            used = sum(pr["used_gb"] for pr in guards.gpu_procs() if guards.container_of(pr["pid"]) == cid)
            self.save(service_mem_gb={**self.state.get("service_mem_gb", {}), svc.name: used})
        self.save(waiting=None)

    def stop_services(self):
        for cid in self.services.values():
            execute.stop_service(cid)
        if self.network:
            execute.stop_service(None, self.network)
        self.services = {}

    def svc_tokens(self) -> float | None:
        if not self.services:
            return None
        port = {s.name: s.port for s in self.pack.services}
        return sum(execute.service_tokens(cid, self.network, port[n]) for n, cid in self.services.items())

    # ------------------------------------------------------------ one trial = RUN + EVALUATE of a surface on (seed, split)
    def trial(self, surface: Path, seed: int, split: str, tdir: Path, tick=None) -> dict:
        tdir.mkdir(parents=True, exist_ok=True)
        label = tdir.relative_to(self.dir).as_posix().replace("/", "-")
        out, result = tdir / "out", tdir / "result"
        for d in (out, result):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
        cache = guards.CACHE / self.name
        (cache / "run-cache").mkdir(parents=True, exist_ok=True)
        (cache / "eval-cache").mkdir(parents=True, exist_ok=True)
        sp = self.data / split
        common = [(self.sealed / "frozen" / "run", "/frozen", False), (self.sealed / "arlab_lib", "/arlab_lib", False),
                  (surface, "/work", False), (HF, "/hf", False)]
        if (sp / "public").exists():
            common.append((sp / "public", "/data/public", False))
        run_m = common + [(out, "/out", True), (cache / "run-cache", "/cache", True)]
        if (self.data / "train").exists():
            run_m.append((self.data / "train", "/data/train", False))
        net = self.network or "none"
        gpu = self.pack.run.gpu
        res = {"seed": seed, "split": split, "wait_s": 0.0, "run_s": 0.0, "eval_s": 0.0}

        res["wait_s"] += guards.wait_for_free(self.need_gb(), self.own_cids(), self.pack.needs_gpu, self._on_wait)
        self.save(waiting=None)
        tok0 = self.svc_tokens()
        r = execute.run_step(execute.Step(
            name=f"{self.prefix}-run-{label}", image=self.state["image"], command=self.pack.run.command.format(seed=seed, split=split),
            cidfile=tdir / "run.cid", log=tdir / "run.log", timeout_s=self.pack.run.timeout_s, mounts=run_m, network=net, gpu=gpu,
            telemetry=tdir / "telemetry.jsonl", cpus=guards.fast_cpus()), self.own_cids(), tick)
        tokens = None if tok0 is None else self.svc_tokens() - tok0
        res.update(run_s=r.wall_s, peak_mem_gb=r.peak_mem_gb, gpu_temp_max=r.gpu_temp_max, foreign=r.foreign)
        if r.launch_failed:
            return {**res, "status": "infra_error", "reason": f"docker failed to start RUN (rc={r.rc}): {tail(tdir / 'run.log')[-300:]}"}
        if r.contended:
            return {**res, "status": "contended", "reason": f"foreign GPU process during RUN: {r.foreign}"}
        bad = "timeout" if r.timed_out else "oom" if r.oom else "crash" if r.rc != 0 else None
        if bad:
            return {**res, "status": bad, "reason": f"RUN rc={r.rc}", "log_tail": tail(tdir / "run.log")}

        try:
            escaping = [p for p in out.rglob("*") if p.is_symlink() and not p.resolve().is_relative_to(out.resolve())]
        except (OSError, RuntimeError) as e:  # symlink loop
            escaping = [Path(f"<{e}>")]
        if escaping:  # EVALUATE would follow it into its own mounts (e.g. /data/private)
            return {**res, "status": "invalid", "reason": f"RUN output links outside itself: {escaping[0].name}"}
        res["wait_s"] += guards.wait_for_free(self.need_gb(), self.own_cids(), self.pack.needs_gpu, self._on_wait)
        self.save(waiting=None)
        e = self.evaluate_step(surface, out, split, tdir)
        res.update(eval_s=e.wall_s, peak_mem_gb=max(r.peak_mem_gb, e.peak_mem_gb),
                   gpu_temp_max=max([t for t in (r.gpu_temp_max, e.gpu_temp_max) if t is not None], default=None))
        if e.launch_failed:
            return {**res, "status": "infra_error", "reason": f"docker failed to start EVALUATE (rc={e.rc}): {tail(tdir / 'eval.log')[-300:]}"}
        if e.contended:
            return {**res, "status": "contended", "reason": f"foreign GPU process during EVALUATE: {e.foreign}"}
        if e.timed_out or e.oom:
            return {**res, "status": "timeout" if e.timed_out else "oom", "reason": "EVALUATE", "log_tail": tail(tdir / "eval.log")}
        m = read_json(result / "metrics.json")
        why = check_metrics(m) if e.rc == 0 else f"EVALUATE rc={e.rc}"
        if why:
            return {**res, "status": "invalid", "reason": why, "log_tail": tail(tdir / "eval.log")}
        budget = read_json(out / "budget.json", {}) or {}
        used = tokens if self.pack.budget.unit == "service_tokens" else budget.get(self.pack.budget.unit)
        if used is None or used > self.pack.budget.limit:
            return {**res, "status": "invalid", "reason": f"budget {self.pack.budget.unit}={used} (limit {self.pack.budget.limit})"}
        metrics = {**m["metrics"], "train_s": r.wall_s, "eval_s": e.wall_s, "peak_mem_gb": res["peak_mem_gb"],
                   "gpu_temp_max": res["gpu_temp_max"], "service_tokens": tokens, "budget_used": used}
        return {**res, "status": "ok", "primary": float(m["primary"]), "metrics": metrics, "items": m.get("items"),
                "out_gb": dir_gb(out)}

    def evaluate_step(self, surface: Path, out: Path, split: str, tdir: Path) -> execute.StepResult:
        """EVALUATE a RUN output dir; writes tdir/result/metrics.json."""
        label = tdir.relative_to(self.dir).as_posix().replace("/", "-")
        sp, cache = self.data / split, guards.CACHE / self.name
        (tdir / "result").mkdir(parents=True, exist_ok=True)
        m = [(self.sealed / "frozen" / "run", "/frozen", False), (self.sealed / "arlab_lib", "/arlab_lib", False),
             (surface, "/work", False), (HF, "/hf", False), (self.sealed / "frozen" / "eval", "/eval", False), (out, "/run_out", False),
             (tdir / "result", "/result", True), (cache / "eval-cache", "/cache", True)]
        m += [(sp / d, f"/data/{d}", False) for d in ("public", "private") if (sp / d).exists()]
        return execute.run_step(execute.Step(
            name=f"{self.prefix}-eval-{label}", image=self.state["image"], user="0:0",
            command=f"{self.pack.evaluate.command}; rc=$?; chown -R 1000:1000 /result; exit $rc",
            cidfile=tdir / "eval.cid", log=tdir / "eval.log", timeout_s=self.pack.evaluate.timeout_s, mounts=m,
            network=self.network or "none", gpu=self.pack.run.gpu, telemetry=tdir / "telemetry.jsonl", cpus=guards.fast_cpus()),
            self.own_cids())

    def clean_trial(self, res: dict, tdir: Path, keep_out: bool = False):
        if not keep_out:
            shutil.rmtree(tdir / "out", ignore_errors=True)
        shutil.rmtree(tdir / "surface", ignore_errors=True)

    def settled(self, fn) -> dict:
        """CALIBRATE/FINALIZE: never use a contended run; wait and retry until clean."""
        infra = 0
        while True:
            res = fn()
            if res["status"] == "infra_error" and infra < 3:
                infra += 1
                wait = float(os.environ.get("ARLAB_BACKOFF_S", "60"))
                self.log(f"infra_error ({res['reason']}); retry {infra}/3 in {wait:.0f}s")
                time.sleep(wait)
                continue
            if res["status"] != "contended":
                return res
            self.log(f"contended ({res['reason']}); waiting and retrying")

    # ------------------------------------------------------------ CALIBRATE (PROBE = first calibration seed)
    def calibrate(self):
        p, cal = self.pack, {}
        base = self.state["baseline_commit"]
        for i, seed in enumerate(p.seeds.calibration):
            tdir = self.dir / "calib" / f"baseline-s{seed}"
            res = read_json(tdir / "result.json")
            if res is None:
                self.save(phase="probe" if i == 0 else "calibrate")
                self.log(f"{'PROBE' if i == 0 else 'CALIBRATE'} baseline seed {seed}")
                res = self.settled(lambda: self.trial(self.export(base, tdir / "surface"), seed, "validation", tdir))
                if res["status"] != "ok":
                    raise ConfigError(f"baseline failed on seed {seed}: {res['status']} {res.get('reason')}")
                self.clean_trial(res, tdir)
                write_json(tdir / "result.json", res)
            if i == 0 and "probe" not in self.state:
                mem = max(16.0, 1.5 * res["peak_mem_gb"]) if p.run.mem_gb == "auto" else float(p.run.mem_gb)
                self.save(probe={"peak_mem_gb": res["peak_mem_gb"], "out_gb": res["out_gb"], "run_s": res["run_s"]}, mem_gb=mem)
            cal[seed] = res
        refs = {}
        for ref in p.references:
            tdir = self.dir / "calib" / f"ref-{ref.name}-s{p.seeds.screen}"
            res = read_json(tdir / "result.json")
            if res is None:
                self.log(f"CALIBRATE reference {ref.name}")
                res = self.settled(lambda: self.trial(self.sealed / ref.path, p.seeds.screen, "validation", tdir))
                if res["status"] != "ok":
                    raise ConfigError(f"reference {ref.name} failed: {res['status']} {res.get('reason')}")
                self.clean_trial(res, tdir)
                write_json(tdir / "result.json", res)
            refs[ref.name] = res
        if "sigma" in self.state:
            return
        sig = stats.sigma([cal[s]["primary"] for s in p.seeds.calibration])
        items = cal[p.seeds.screen].get("items")
        n_val = len(items) if items is not None else None
        if not p.seeds.confirm and not stats.deterministic_enough(sig, n_val):
            raise ConfigError(f"seeds.confirm is empty but the baseline is not deterministic enough (sigma={sig:.4g})")
        n_hold, item_sd = None, None
        if items is not None:
            n_hold = (read_json(self.data / "splits.json", {}) or {}).get("holdout")
            if not n_hold:
                raise ConfigError("item pack: PREPARE must write /data/splits.json with the holdout item count")
            if refs:
                s = p.seeds.screen
                item_sd = max(stats.compare({s: r}, {s: cal[s]}, [s], p.metric.direction, 0.0)["item_sd"] for r in refs.values())
        exp_se = stats.expected_holdout_se(sig, len(p.seeds.holdout), n_hold, item_sd)
        up = stats.underpowered(exp_se, p.metric.mes)
        self.save(sigma=sig, n_validation_items=n_val, n_holdout_items=n_hold, expected_holdout_se=exp_se, underpowered=up,
                  calibration={str(s): cal[s]["primary"] for s in p.seeds.calibration},
                  references={k: v["primary"] for k, v in refs.items()})
        self.log(f"CALIBRATE sigma={sig:.4g} expected holdout SE={exp_se:.4g} mes={p.metric.mes} underpowered={up}")

    def baseline_vals(self) -> dict:
        return {s: read_json(self.dir / "calib" / f"baseline-s{s}" / "result.json") for s in self.pack.seeds.calibration}

    def baseline_metric(self, name: str) -> float | None:
        vals = [v["metrics"].get(name) for v in self.baseline_vals().values()]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    # ------------------------------------------------------------ LOOP
    def incumbent(self, records: list[dict]) -> tuple[str, dict, str]:
        keeps = [r for r in records if r["status"] == "keep"]
        if keeps:
            k = keeps[-1]
            return k["commit"], {int(s): v for s, v in k["results"].items()}, k["id"]
        return self.state["baseline_commit"], self.baseline_vals(), "baseline"

    def stop_reason(self, records: list[dict]) -> str | None:
        c, cfg = counted(records), self.pack.campaign
        streak = 0
        for r in reversed(c):
            if r["status"] == "keep":
                break
            streak += 1
        infra = 0
        for r in reversed(records):
            if r["status"] != "infra_error":
                break
            infra += 1
        active_h = (sum(sum(r.get("timings", {}).get(k, 0) for k in ("propose_s", "run_s", "eval_s")) for r in c)
                    + sum(v["run_s"] + v["eval_s"] for v in self.baseline_vals().values() if v)) / 3600
        need_disk = DISK_FLOOR_GB + 2 * self.state.get("probe", {}).get("out_gb", 0)
        checks = [(len(c) >= cfg.max_experiments, "max_experiments"), (active_h >= cfg.max_hours, "max_hours"),
                  (sum(r.get("agent_calls", 0) for r in records) >= cfg.max_agent_calls, "max_agent_calls"),
                  (streak >= cfg.stop_after_no_keep, "no_keep"), ((self.dir / "STOP").exists(), "stop"),
                  (guards.disk_free_gb(self.dir) < need_disk, "disk"), (infra >= 6, "infra")]
        return next((why for hit, why in checks if hit), None)

    def make_backend(self):
        if self.script:
            return ScriptedBackend(self.script, lambda: len(counted(load_records(self.dir))))
        a = self.pack.agent
        return CodexBackend(a.model, a.effort, a.timeout_s)

    def loop(self):
        from .experiment import run_experiment
        self.backend = self.backend or self.make_backend()
        backoff = float(os.environ.get("ARLAB_BACKOFF_S", "60"))
        while True:
            records = load_records(self.dir)
            write_results_tsv(self.dir, records)
            why = self.stop_reason(records)
            if why:
                self.save(stop_reason=why)
                self.log(f"STOP: {why}")
                return
            eid = f"{int(records[-1]['id']) + 1:04d}" if records else "0001"
            self.save(phase="loop", current=eid)
            try:
                rec = run_experiment(self, eid, records)
            except AuthError as e:
                shutil.rmtree(self.dir / "runs" / eid, ignore_errors=True)
                self.save(paused="codex auth")
                self.log(f"PAUSED: codex auth ({str(e)[:200]})")
                while True:
                    time.sleep(600)
                    if codex_login_ok():
                        break
                self.save(paused=None)
                continue
            write_json(self.dir / "runs" / eid / "record.json", rec)
            self.log(f"{eid} {rec['status']} primary={rec.get('primary')} d={rec.get('delta')} {rec.get('description', '')[:80]}")
            debug_kill(f"after_record:{eid}")
            if rec["status"] == "infra_error":
                streak = sum(1 for _ in _tail_while(load_records(self.dir), "infra_error"))
                time.sleep(min(backoff * 2 ** (streak - 1), 30 * backoff))

    # ------------------------------------------------------------ FINALIZE
    def records(self) -> list[dict]:
        return load_records(self.dir)

    def report(self):
        from .report import write_report
        return write_report(self)

    def finalize(self):
        p = self.pack
        records = self.records()
        write_results_tsv(self.dir, records)
        n_exp = len(counted(records))
        mes, direction = p.metric.mes, p.metric.direction
        if self.state.get("underpowered") and not p.acceptance.allow_underpowered:
            self.save(verdict="inconclusive", verdict_reason="underpowered", holdout=None, phase="finalized")
            self.log("VERDICT inconclusive: underpowered")
            return
        inc_commit, _, inc_id = self.incumbent(records)
        cmp_name = p.verdict.compare_to
        if cmp_name == "baseline" and inc_id == "baseline":
            c = {"d": 0.0, "se": self.state["expected_holdout_se"], "per_seed": [], "n_items": self.state.get("n_holdout_items"),
                 "skipped": "no keeps: holdout not run"}
        else:
            self.save(phase="finalize")
            vals = {"incumbent": {}, cmp_name: {}}
            for seed in p.seeds.holdout:
                for which in vals:
                    tdir = self.dir / "holdout" / f"{which}-s{seed}"
                    res = read_json(tdir / "result.json")
                    if res is None:
                        self.log(f"FINALIZE holdout {which} seed {seed}")
                        if which == "incumbent":
                            surf = self.export(inc_commit, tdir / "surface")
                        elif which == "baseline":
                            surf = self.export(self.state["baseline_commit"], tdir / "surface")
                        else:
                            surf = self.sealed / next(r.path for r in p.references if r.name == which)
                        res = self.settled(lambda: self.trial(surf, seed, "holdout", tdir))
                        self.clean_trial(res, tdir)
                        write_json(tdir / "result.json", res)
                    if res["status"] != "ok":
                        self.save(verdict="inconclusive", verdict_reason=f"holdout_failed:{which}-s{seed}:{res['status']}", phase="finalized")
                        return
                    vals[which][seed] = res
            c = stats.compare(vals["incumbent"], vals[cmp_name], p.seeds.holdout, direction, self.state["sigma"])
        v, why = stats.verdict(c, mes, is_underpowered=bool(self.state.get("underpowered")), n_experiments=n_exp,
                               stop_reason=self.state.get("stop_reason") or "")
        self.save(verdict=v, verdict_reason=why, holdout=c, incumbent=inc_id, phase="finalized")
        self.log(f"VERDICT {v}: {why}")

    # ------------------------------------------------------------ entry point
    def run(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = guards.FileLock(self.dir / ".lock")
        if not self.lock.acquire(blocking=False):
            print(f"another arlab runner holds {self.dir}/.lock", file=sys.stderr)
            return 3
        try:
            if self.state.get("phase") == "finalized":
                self.pack = load_pack(self.sealed)
                self.report()
                return 0
            execute.kill_leftovers(self.dir, self.prefix)
            self.setup()
            self.init_work()
            self.resume_records()
            try:
                if self.pack.needs_gpu:
                    self.save(waiting="waiting for the arlab GPU lock")
                    self.gpu_lock = guards.gpu_lock()
                    self.gpu_lock.acquire()
                    self.save(waiting=None)
                self.start_services()
                self.calibrate()
                if self.state.get("underpowered") and not self.pack.acceptance.allow_underpowered:
                    self.save(stop_reason="underpowered")
                    self.log("STOP: underpowered (2 x expected holdout SE > mes); no experiments run")
                elif not self.state.get("stop_reason"):  # once the LOOP has stopped it never restarts (FINALIZE resumes)
                    self.loop()
                self.finalize()
            finally:
                self.stop_services()
                if self.gpu_lock:
                    self.gpu_lock.release()
        except ConfigError as e:
            self.log(f"CONFIG ERROR: {e}")
            self.save(phase="finalized", verdict=None, verdict_reason=f"config error: {e}", stop_reason="config")
        self.report()
        self.lock.release()
        return 0

    def resume_records(self):
        """Run dirs without record.json become `interrupted`; work/ is hard-reset to the incumbent."""
        for rd in sorted((self.dir / "runs").glob("*")) if (self.dir / "runs").exists() else []:
            if rd.is_dir() and not (rd / "record.json").exists():
                calls = read_json(rd / "calls.json", {}) or {}
                write_json(rd / "record.json", {"id": rd.name, "status": "interrupted", "reason": "runner died mid-experiment",
                                                "description": "", "hypothesis_tag": "", "agent_calls": calls.get("agent_calls", 0),
                                                "tokens": calls.get("tokens", {}), "timings": calls.get("timings", {})})
                for sub in rd.glob("*/out"):
                    shutil.rmtree(sub, ignore_errors=True)
        if self.state.get("baseline_commit"):
            keeps = [r for r in load_records(self.dir) if r["status"] == "keep"]
            self.git("reset", "-q", "--hard", keeps[-1]["commit"] if keeps else self.state["baseline_commit"])
            self.git("clean", "-qfdx")


def _tail_while(records, status):
    for r in reversed(records):
        if r["status"] != status:
            return
        yield r


def tail(path: Path, n: int = 80) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def dir_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9


def check_metrics(m) -> str | None:
    """metrics.json contract; returns the reason it is invalid, or None."""
    if not isinstance(m, dict) or not {"valid", "primary", "metrics", "items", "message"} <= set(m):
        return "metrics.json missing or malformed"
    if not m["valid"]:
        return f"evaluator: {m.get('message')}"
    ok = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    if not ok(m["primary"]):
        return f"non-finite primary {m['primary']!r}"
    if not isinstance(m["metrics"], dict):
        return "metrics is not a dict"
    if m["items"] is not None and (not isinstance(m["items"], dict) or not m["items"] or not all(ok(v) for v in m["items"].values())):
        return "items malformed or non-finite"
    return None
