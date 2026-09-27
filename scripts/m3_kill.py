#!/usr/bin/env python
"""M3 kill tests (PLAN §5 M3): `kill -9` the detached nanochat-lite/m3 runner during an agent call and during
training; systemd restarts it (Restart=on-failure) and it must resume. Results go to <campaign>/kill-tests.json,
which accept-M3 reads. Usage: m3_kill.py during_agent|during_training [--tag m3]"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arlab.campaign import RUNS  # noqa: E402
from arlab.record import read_json, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["during_agent", "during_training"])
ap.add_argument("--tag", default="m3")
a = ap.parse_args()
camp = RUNS / "nanochat-lite" / a.tag
unit = f"arlab-nanochat-lite-{a.tag}"
prefix = f"arlab-nanochat-lite-{a.tag}-" + ("agent-" if a.kind == "during_agent" else "run-runs-")


def sh(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def main_pid():
    return int(sh("systemctl", "--user", "show", "-p", "MainPID", "--value", unit) or 0)


# 1. wait until the runner is inside the wanted phase (an agent / RUN container of this campaign, >20 s old)
seen: dict[str, float] = {}
name = None
give_up = time.monotonic() + 3 * 3600
while name is None:
    if time.monotonic() > give_up:
        sys.exit(f"no {prefix}* container ran for 20 s within 3 h")
    for n in sh("docker", "ps", "--format", "{{.Names}}").splitlines():
        if n.startswith(prefix) and not n.endswith("-fix"):
            seen.setdefault(n, time.time())
            if time.time() - seen[n] >= 20:
                name = n
    time.sleep(5)
eid = name.removeprefix(prefix).split("-")[0]
pid = main_pid()
if pid <= 0:
    sys.exit(f"{unit} has no main PID; not killing anything")
t_kill = time.time()
subprocess.run(["kill", "-9", str(pid)], check=True)
print(f"killed runner pid {pid} while {name} was running (experiment {eid})", flush=True)

# 2. wait for systemd to restart the runner and for it to write the interrupted record and move on
ok, why = False, ""
deadline = time.time() + 1800
while time.time() < deadline:
    time.sleep(10)
    rec = read_json(camp / "runs" / eid / "record.json")
    newpid = main_pid()
    later = sorted(p.name for p in (camp / "runs").iterdir() if p.name > eid)
    if rec and rec.get("status") == "interrupted" and newpid and newpid != pid and later:
        ok, why = True, f"runner restarted (pid {newpid}); {eid} recorded interrupted; next experiment {later[0]} started"
        break
    why = f"record={rec and rec.get('status')} newpid={newpid} later={later[:1]}"
leftover = name in sh("docker", "ps", "--format", "{{.Names}}").splitlines()
res = read_json(camp / "kill-tests.json", {}) or {}
res[a.kind] = {"killed_pid": pid, "container": name, "experiment": eid, "killed_at": t_kill, "resumed": ok and not leftover,
               "leftover_container": leftover, "detail": why}
write_json(camp / "kill-tests.json", res)
print(json.dumps(res[a.kind], indent=1))
sys.exit(0 if ok and not leftover else 1)
