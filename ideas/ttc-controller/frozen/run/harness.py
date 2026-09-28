"""Frozen RUN entry point for ttc-controller: replays the cached traces of one split through the surface controller.

For each budget level (1, 0.5, 2 x --budget-tokens) and replicate r < --replicates it starts a FRESH sandboxed child
(child.py) that imports /work/controller.py, and serves it one Episode (replay.py) at a time: children never overlap,
cannot write files or start processes, and a new SysV/POSIX IPC object left behind marks the run. Limits: 120 s to
start, 60 s for Controller(cfg) + fit(), 5 s per solve(); breaking one crashes the run.

Outputs in --out (the evaluator re-derives every score from log.json and its own private gold):
  log.json     settings, per episode: sandbox self-check, ipc_clean, pool, charged, and per problem (in episode order)
               the pid, the events (["o"] open, ["r", t, n, charged] read) and the returned label
  budget.json  {"budget_fraction": max over episodes of charged / pool}
"""
import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time

from replay import LEVELS, Episode, controller_seed, load_cache

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child.py")
STARTUP_S, FIT_S, SOLVE_S, MAX_LINE = 120.0, 60.0, 5.0, 1 << 20
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}


class ControllerFailure(Exception):
    pass


class Child:
    def __init__(self, work: str, train: str, deny: list):
        cmd = [sys.executable, "-s", "-B", CHILD, "--work", work, "--train", train] + [x for d in deny for x in ("--deny", d)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True,
                                  start_new_session=True, cwd="/", env=ENV)
        self.buf = b""

    def send(self, obj):
        try:
            self.p.stdin.write((json.dumps(obj) + "\n").encode())
            self.p.stdin.flush()
        except BrokenPipeError:
            raise ControllerFailure(f"controller process exited (rc={self.p.poll()})")

    def recv(self, deadline: float, what: str) -> dict:
        fd = self.p.stdout.fileno()
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ControllerFailure(f"timeout: {what}")
            if select.select([fd], [], [], left)[0]:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    raise ControllerFailure(f"controller process exited during {what} (rc={self.p.wait()})")
                self.buf += chunk
                if len(self.buf) > MAX_LINE:
                    raise ControllerFailure("message too long")
        line, self.buf = self.buf.split(b"\n", 1)
        msg = json.loads(line)
        if msg.get("op") == "error":
            raise ControllerFailure(f"controller raised during {what}:\n{msg.get('msg')}")
        return msg

    def close(self):
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.p.wait()


def ipc_objects() -> set:
    out = set()
    for f in ("shm", "msg", "sem"):
        try:
            out |= {f"{f}:{ln.split()[1]}" for ln in open(f"/proc/sysvipc/{f}").read().splitlines()[1:] if ln.strip()}
        except OSError:
            pass
    if os.path.isdir("/dev/mqueue"):
        out |= {f"mq:{x}" for x in os.listdir("/dev/mqueue")}
    return out


def run_episode(c, a, r, level, public, train) -> dict:
    ep = Episode(c, a.seed, r, round(level * a.budget_tokens))
    ipc0, t_start = ipc_objects(), time.monotonic()
    rec = {"level": level, "replicate": r, "budget": ep.budget, "pool": ep.pool}
    ch = Child(a.work, train, [p for p in (public, train) if os.path.exists(p)])
    try:
        hello = ch.recv(time.monotonic() + STARTUP_S, "startup")
        rec["sandbox"] = hello.get("sandbox", {"ok": False})
        if rec["sandbox"].get("ok") is not True:
            return rec
        cfg = {"budget": ep.budget, "n_problems": c["n"], "pool": ep.pool, "max_traces": c["T"],
               "seed": controller_seed(a.seed, r, level)}
        t0 = time.monotonic()
        ch.send({"op": "init", "cfg": cfg})
        rec["fit_s"] = ch.recv(t0 + FIT_S, "Controller(cfg) + fit()").get("fit_s")
        for i in range(c["n"]):
            info = ep.begin(i)
            t0 = time.monotonic()
            ch.send({"op": "solve", **info})
            while True:
                m = ch.recv(t0 + SOLVE_S, f"solve() of problem {i} (limit {SOLVE_S:.0f} s)")
                if m.get("op") in ("open", "read"):
                    ch.send(ep.handle(m))
                elif m.get("op") == "answer" and (m.get("label") is None or (isinstance(m["label"], str) and len(m["label"]) < 32)):
                    ep.finish(m["label"], time.monotonic() - t0)
                    break
                else:
                    raise ControllerFailure(f"unexpected message {str(m)[:200]}")
        ch.send({"op": "exit"})
    finally:
        ch.close()
    rec.update(charged=ep.charged, ipc_clean=ipc_objects() <= ipc0, wall_s=round(time.monotonic() - t_start, 2),
               problems=ep.problems)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--budget-tokens", type=int, required=True, help="B: mean generated tokens per problem (level 1)")
    ap.add_argument("--replicates", type=int, required=True)
    ap.add_argument("--data", default="/data", help="reads <data>/public/traces.npz and <data>/train/traces.npz")
    ap.add_argument("--work", default="/work")
    ap.add_argument("--limit", type=int, default=0, help="tests/pilot only (never in pack.yaml): first N problems")
    a = ap.parse_args()
    public, train = f"{a.data}/public/traces.npz", f"{a.data}/train/traces.npz"
    c = load_cache(public, a.limit)
    log = {"seed": a.seed, "split": a.split, "budget_tokens": a.budget_tokens, "replicates": a.replicates,
           "levels": list(LEVELS), "limit": a.limit, "n_problems": c["n"], "n_traces": c["T"], "episodes": []}
    print(f"{a.split}: {c['n']} problems x {c['T']} traces, seed {a.seed}, B {a.budget_tokens}, R {a.replicates}", flush=True)
    t0 = time.monotonic()
    try:
        for level in LEVELS:
            for r in range(a.replicates):
                rec = run_episode(c, a, r, level, public, train)
                log["episodes"].append(rec)
                if rec["sandbox"].get("ok") is not True or not rec["ipc_clean"]:
                    print(f"SANDBOX FAILURE: {rec['sandbox']} ipc_clean={rec.get('ipc_clean')}; the run is invalid", flush=True)
                    raise StopIteration
                ab = sum(p["answer"] is None for p in rec["problems"]) / c["n"]
                print(f"level {level} r{r}: used {rec['charged']}/{rec['pool']} ({rec['charged'] / rec['pool']:.3f}), "
                      f"abstain {ab:.3f}, fit {rec['fit_s']}s, wall {rec['wall_s']}s", flush=True)
    except StopIteration:
        pass
    except ControllerFailure as e:
        sys.exit(f"CONTROLLER FAILURE: {e}")
    log["wall_s"] = round(time.monotonic() - t0, 2)
    frac = max((e.get("charged", 0) / e["pool"] for e in log["episodes"]), default=0.0)
    for name, obj in (("log.json", log), ("budget.json", {"budget_fraction": frac})):
        json.dump(obj, open(f"{a.out}/{name}.tmp", "w"))
        os.replace(f"{a.out}/{name}.tmp", f"{a.out}/{name}")
    print(f"done in {log['wall_s']} s; budget_fraction {frac:.4f}", flush=True)


if __name__ == "__main__":
    main()
