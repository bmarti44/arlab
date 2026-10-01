"""Frozen RUN entry point for latent-arch: a trusted SUPERVISOR that never imports the surface.

It starts trainer.py (the only process that runs surface code) in its own session, waits until the trainer has
loaded torch and the data, then starts the clock (CLOCK_MONOTONIC) and sends the deadline. Surface import, build,
torch.compile, every train_step and the checkpoint write all count. The trainer stops cooperatively at the deadline;
the supervisor kills its whole process group at deadline + GRACE_S no matter what the surface did to its own clock.
train_seconds is measured here, from go to the trainer's done message. Then every remaining descendant is killed
(this process is a subreaper, so double-forked helpers cannot escape), and only then are the checkpoint validated
(flat {name: tensor} dict, weights-only) and budget.json / stats.json written.

Outputs in --out:
  model.pt       data-only checkpoint {name: tensor} of every parameter and buffer of the surface's model
  budget.json    {"train_seconds": t}   (runner: invalid if > budget.limit = 690)
  stats.json     supervisor facts (trusted: train_seconds, killed, checkpoint sha256 / tensor hash / element count,
                 stray processes) + the trainer's own report under "trainer" (informational only)
"""
import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time

from common import GRACE_S, SEQ_LEN, count_elements, file_sha256, load_checkpoint, tensors_hash
from sandbox import become_subreaper, kill_descendants

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.py")
STARTUP_S, MAX_LINE = 300.0, 1 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", required=True)          # unused: RUN never sees eval data
    ap.add_argument("--train-seconds", type=float, required=True)
    ap.add_argument("--data", default="/data/train")
    ap.add_argument("--work", default="/work")
    ap.add_argument("--grace", type=float, default=GRACE_S, help="tests only (never in pack.yaml)")
    ap.add_argument("--smoke", action="store_true", help="tests only: CPU, 3 text + 1 program rows of 128 tokens")
    a = ap.parse_args()
    become_subreaper()
    ckpt = f"{a.out}/model.pt"
    for f in ("model.pt", "budget.json", "stats.json"):
        if os.path.lexists(f"{a.out}/{f}"):
            os.remove(f"{a.out}/{f}")
    cmd = [sys.executable, "-B", TRAINER, "--seed", str(a.seed), "--train-seconds", str(a.train_seconds),
           "--data", a.data, "--work", a.work, "--ckpt", ckpt] + (["--smoke"] if a.smoke else [])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True, env=env)
    buf = b""

    def recv(deadline):
        nonlocal buf
        fd = p.stdout.fileno()
        while b"\n" not in buf:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                return {"op": "timeout"}
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return {"op": "exited"}
            buf += chunk
            if len(buf) > MAX_LINE:
                return {"op": "error", "msg": "message too long"}
        line, buf = buf.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError:
            return {"op": "error", "msg": "malformed message"}

    status, report = "ok", {}
    msg = recv(time.monotonic() + STARTUP_S)
    if msg.get("op") != "ready":
        status, report = "startup_failed", msg
        t0 = t_end = time.monotonic()
    else:
        t0 = time.monotonic()
        deadline = t0 + a.train_seconds
        p.stdin.write((json.dumps({"op": "go", "t0": t0, "deadline": deadline}) + "\n").encode())
        p.stdin.flush()
        msg = recv(deadline + a.grace)
        t_end = time.monotonic()
        if msg.get("op") == "done":
            report = msg
        elif msg.get("op") == "timeout":
            status = "killed"
            print(f"SUPERVISOR: no checkpoint by budget + {a.grace:.0f} s; killing the trainer", flush=True)
        else:
            status, report = "trainer_failed", msg
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    rc = p.wait()
    strays = kill_descendants()
    train_s = t_end - t0
    stats = {"status": status, "train_seconds": train_s, "killed": status == "killed", "trainer_rc": rc,
             "stray_processes_killed": strays, "seed": a.seed, "seq_len": 128 if a.smoke else SEQ_LEN, "trainer": report}
    if status == "ok":
        if os.path.islink(ckpt) or not os.path.isfile(ckpt):
            stats["status"] = "no_checkpoint"
        else:
            try:
                ck = load_checkpoint(ckpt)
                stats.update(model_sha256=file_sha256(ckpt), tensors_hash=tensors_hash(ck), n_elements=count_elements(ck))
            except ValueError as e:
                stats["status"] = f"bad_checkpoint: {e}"
    for name, obj in (("budget.json", {"train_seconds": train_s}), ("stats.json", stats)):
        if os.path.lexists(f"{a.out}/{name}.tmp"):   # the trainer could have planted a symlink there
            os.remove(f"{a.out}/{name}.tmp")
        with open(f"{a.out}/{name}.tmp", "w") as f:
            json.dump(obj, f)
        os.replace(f"{a.out}/{name}.tmp", f"{a.out}/{name}")
    print(json.dumps(stats), flush=True)
    if status in ("startup_failed", "trainer_failed"):   # a surface crash is a crash (the runner's status "crash")
        print(report.get("msg", report), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
