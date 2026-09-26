"""arlab CLI: new | check | run | status | stop | report."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .campaign import RUNS, Campaign, ConfigError
from .pack import ARLAB_ROOT, load_pack
from .record import load_records, read_json
from .report import speed, write_report

ARLAB_BIN = ARLAB_ROOT / ".venv" / "bin" / "arlab"


def unit_name(name: str, tag: str) -> str:
    return f"arlab-{name.strip('_')}-{tag}".replace("_", "-")


def cmd_new(a):
    dest = ARLAB_ROOT / "ideas" / a.name
    if dest.exists():
        sys.exit(f"{dest} already exists")
    shutil.copytree(ARLAB_ROOT / "templates" / "pack", dest)
    y = dest / "pack.yaml"
    y.write_text(y.read_text().replace("__NAME__", a.name))
    if a.idea:
        (dest / "IDEA.md").write_text(Path(a.idea).read_text())
    print(f"scaffolded {dest}; now author it following docs/PACK-AUTHORING.md")


def cmd_check(a):
    pack_dir = Path(a.pack).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="arlab-check-"))
    c = Campaign(pack_dir, "check", runs_root=tmp)
    ok = False
    try:
        c.setup()
        print(f"CHECK + PREPARE + TESTS + SEAL: ok (data {c.state['data_dir']}, image {c.state['image']})")
        if a.static:
            ok = True
        else:
            c.init_work()
            if c.pack.needs_gpu:
                from .guards import gpu_lock
                c.gpu_lock = gpu_lock()
                print("waiting for the arlab GPU lock ...", flush=True)
                c.gpu_lock.acquire()
            try:
                c.start_services()
                tdir = c.dir / "probe"
                res = c.trial(c.export(c.state["baseline_commit"], tdir / "surface"), c.pack.seeds.screen, "validation", tdir)
            finally:
                c.stop_services()
                if c.gpu_lock:
                    c.gpu_lock.release()
            ok = res["status"] == "ok"
            print(json.dumps({k: res.get(k) for k in ("status", "reason", "primary", "run_s", "eval_s", "peak_mem_gb", "out_gb")}, default=str))
            if not ok:
                print(res.get("log_tail", ""))
            for svc in c.pack.services:
                used = c.state.get("service_mem_gb", {}).get(svc.name, 0)
                print(f"service {svc.name}: measured {used:.1f} GB (mem_gb {svc.mem_gb})")
                if used > 1.1 * svc.mem_gb:
                    ok = False
                    print(f"FAIL: service {svc.name} uses more than 110% of mem_gb")
    except ConfigError as e:
        print(f"FAIL: {e}")
        for f in ("prepare.log", "tests.log"):
            if (c.dir / f).exists():
                print(f"--- {f} (tail)\n" + "\n".join((c.dir / f).read_text(errors="replace").splitlines()[-40:]))
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(tmp)], check=False)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_run(a):
    pack_dir = Path(a.pack).resolve()
    name = pack_dir.name
    if a.detach:
        camp = RUNS / name / a.tag
        camp.mkdir(parents=True, exist_ok=True)
        log = camp / "runner.log"
        cmd = ["systemd-run", "--user", f"--unit={unit_name(name, a.tag)}", "--collect", "-p", "Restart=on-failure", "-p", "RestartSec=60",
               "-p", "StartLimitIntervalSec=3600", "-p", "StartLimitBurst=5", "-p", f"StandardOutput=append:{log}",
               "-p", f"StandardError=append:{log}", str(ARLAB_BIN), "run", str(pack_dir), "--tag", a.tag]
        if a.script:
            cmd += ["--script", str(Path(a.script).resolve())]
        return subprocess.run(cmd).returncode
    return Campaign(pack_dir, a.tag, script=Path(a.script).resolve() if a.script else None).run()


def _campaign(name: str, tag: str) -> Campaign:
    name = Path(name).name
    c = Campaign(ARLAB_ROOT / "ideas" / name, tag)
    if not c.dir.exists():
        sys.exit(f"no campaign {c.dir}")
    if (c.dir / "sealed" / "pack.yaml").exists():
        c.pack = load_pack(c.dir / "sealed")
    return c


def status_of(c: Campaign) -> dict:
    st = c.state
    recs = load_records(c.dir)
    sp = speed(recs)
    unit = unit_name(c.name, c.tag)
    active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True).stdout.strip()
    keeps = [r for r in recs if r["status"] == "keep"]
    base = (st.get("calibration") or {}).get(str(c.pack.seeds.screen)) if c.pack else None
    cfg = c.pack.campaign if c.pack else None
    n = sp["n"]
    out = {"campaign": f"{c.name}/{c.tag}", "phase": st.get("phase"), "unit": f"{unit} ({active})", "unit_failed": active == "failed",
           "waiting": st.get("waiting"), "paused": st.get("paused"), "baseline": base,
           "incumbent": (keeps[-1]["id"], keeps[-1]["primary"]) if keeps else ("baseline", base),
           "sigma": st.get("sigma"), "expected_holdout_se": st.get("expected_holdout_se"), "underpowered": st.get("underpowered"),
           "experiments": n, "keeps": len(keeps), "accept_rate": len(keeps) / n if n else None,
           "per_hour_active": round(sp["per_hour"], 2), "stop_reason": st.get("stop_reason"),
           "verdict": st.get("verdict"), "verdict_reason": st.get("verdict_reason"),
           "last10": [(r["id"], r["status"], r.get("primary"), r.get("delta"), r.get("description", "")[:60]) for r in recs[-10:]]}
    if cfg:
        out["limits_left"] = {"experiments": cfg.max_experiments - n, "hours": round(cfg.max_hours - sp["active_h"], 2),
                              "agent_calls": cfg.max_agent_calls - sum(r.get("agent_calls", 0) for r in recs)}
    return out


def cmd_status(a):
    if a.name:
        s = status_of(_campaign(a.name, a.tag))
        if a.json:
            print(json.dumps(s, default=str, indent=1))
        else:
            for k, v in s.items():
                if k != "last10":
                    print(f"{k:>20}: {v}")
            print("last 10:")
            for row in s["last10"]:
                print("   ", *row)
        return 0
    rows = []
    for sf in sorted(RUNS.glob("*/*/state.json")):
        st = read_json(sf, {})
        rows.append({"campaign": f"{sf.parent.parent.name}/{sf.parent.name}", "phase": st.get("phase"), "verdict": st.get("verdict"),
                     "waiting": st.get("waiting"), "paused": st.get("paused")})
    print(json.dumps(rows, indent=1) if a.json else "\n".join(f"{r['campaign']:40} {r['phase']!s:10} {r['verdict'] or ''} "
                                                                f"{r['paused'] or r['waiting'] or ''}" for r in rows))
    return 0


def cmd_stop(a):
    c = _campaign(a.name, a.tag)
    (c.dir / "STOP").touch()
    print(f"stop requested: {c.dir}/STOP (takes effect after the current experiment)")


def cmd_report(a):
    c = _campaign(a.name, a.tag)
    print(write_report(c))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new"); s.add_argument("name"); s.add_argument("--idea", help="prose IDEA.md to copy in"); s.set_defaults(f=cmd_new)
    s = sub.add_parser("check"); s.add_argument("pack"); s.add_argument("--static", action="store_true"); s.set_defaults(f=cmd_check)
    s = sub.add_parser("run"); s.add_argument("pack"); s.add_argument("--tag", required=True); s.add_argument("--detach", action="store_true")
    s.add_argument("--script", help=argparse.SUPPRESS); s.set_defaults(f=cmd_run)
    s = sub.add_parser("status"); s.add_argument("name", nargs="?"); s.add_argument("--tag"); s.add_argument("--json", action="store_true")
    s.set_defaults(f=cmd_status)
    s = sub.add_parser("stop"); s.add_argument("name"); s.add_argument("--tag", required=True); s.set_defaults(f=cmd_stop)
    s = sub.add_parser("report"); s.add_argument("name"); s.add_argument("--tag", required=True); s.set_defaults(f=cmd_report)
    a = ap.parse_args(argv)
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
