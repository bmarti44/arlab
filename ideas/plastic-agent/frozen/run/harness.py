"""Frozen RUN entry point for plastic-agent: a trusted SUPERVISOR that owns the model and never imports the surface.

It loads the base model once, then for every world of the split (fixed order) starts a FRESH sandboxed child process
(child.py: Landlock, no GPU, no network, own session; writable only: a fresh per-world scratch dir that is deleted
afterwards) that imports /work/adapt.py and calls
adapt(transcript, tool_names, gen, train). The child receives only that world's transcript and tool names (over its
stdin; it cannot open any data file, the model snapshot or /hf) and reaches the model only through RPC: every
generate / teacher / train request is validated, executed and metered here (engine.Gen, trainer.Trainer, one Budget per
world). The child cannot pass budget=None, touch a counter, move the deadline or see adapter weights: adapters stay in
this process under immutable integer ids, and adapt() returns an id (or None). Teacher log-probs stay here too.

Clock: CLOCK_MONOTONIC, started when the child is ready and the world is sent ("go"): surface import + adapt() +
every RPC count. The per-world budget refuses work after the deadline; the child's whole process group is SIGKILLed at
deadline + GRACE_S no matter what it does (the last adapter trained in that world is then kept). On "done" (or kill)
the supervisor records adapt_s, inspects the child from outside (/proc: OS threads beyond its startup set, any
descendant process, any orphan re-parented to this subreaper) and records each as a violation (the evaluator
invalidates the run), then kills the process group and every descendant BEFORE it saves the world's adapter. Nothing
the surface started can run after adapt() returns, and the next world starts in a new process: no state carries over.

World-specific control: for a candidate ("adapter" arm) every target world is followed by a second, identical pass on
its TWIN (public/twins/<id>.json: the same tool names and argument vocabulary, other semantics, its own exploration
transcript), whose adapter is saved as adapters/<id>x and scored by the evaluator on the TARGET world. The child cannot
tell the passes apart by anything but the transcript it is given. Every pass has the full per-world budget.

The arm is chosen by frozen code (common.arm_of: the sha256 of /work/adapt.py against the frozen references), never by
the surface. The placebo reference adapts each world to its twin's transcript only (scored on the target).
The base weights are hashed (sha256 over every parameter and buffer) at load and after
every world. Any non-finite loss/gradient in any train() call makes the run invalid (stats.nan).

Outputs in --out (written only by this process; scored only by the frozen evaluator):
  adapters/<world_id>/adapter.{safetensors,json}   one per world whose adapt() returned an adapter (this process's copy)
  adapters/<world_id>x/...                         the candidate's twin (control) adapter of that world
  stats.json     arm, per-world timings / tokens / trainer stats / sandbox self-check / violations (all trusted)
  budget.json    {"adapt_s_max": the longest per-world adapt() wall time}
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import select  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import lora  # noqa: E402
from common import MODEL_DIR, arm_of  # noqa: E402
from engine import Budget, BudgetExceeded, Engine, Gen  # noqa: E402
from sandbox import become_subreaper, descendants, kill_descendants, threads  # noqa: E402
from trainer import Trainer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "child.py")
STARTUP_S, GRACE_S, MAX_MSG = 120.0, 15.0, 256 << 20
CHILD_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}


class Fatal(Exception):
    pass


class Child:
    def __init__(self, work: str, scratch: str, deny: list):
        cmd = [sys.executable, "-s", "-B", CHILD, "--work", work, "--scratch", scratch] + \
            [x for d in deny for x in ("--deny", d)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True,
                                  start_new_session=True, cwd="/", env=CHILD_ENV)
        self.buf = bytearray()

    def send(self, obj) -> bool:
        try:
            self.p.stdin.write((json.dumps(obj) + "\n").encode())
            self.p.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def recv(self, deadline: float) -> dict:
        fd = self.p.stdout.fileno()
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[:i + 1]
                try:
                    msg = json.loads(line)
                except ValueError:
                    return {"op": "malformed"}
                return msg if isinstance(msg, dict) else {"op": "malformed"}
            if not select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
                return {"op": "timeout"}
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return {"op": "exited"}
            self.buf += chunk
            if len(self.buf) > MAX_MSG:
                return {"op": "malformed", "msg": "message too long"}

    def kill(self):
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--adapt-seconds", type=float, required=True, help="per-world wall-clock budget of adapt()")
    ap.add_argument("--gen-tokens", type=int, required=True, help="per-world gen budget (positions processed)")
    ap.add_argument("--train-tokens", type=int, required=True, help="per-world train budget (positions processed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=MODEL_DIR, help="tests only: a tiny random model")
    ap.add_argument("--data", default="/data")
    ap.add_argument("--work", default="/work")
    ap.add_argument("--limit-worlds", type=int, default=0, help="tests only")
    ap.add_argument("--grace", type=float, default=GRACE_S, help="tests only")
    a = ap.parse_args()
    become_subreaper()
    dev = a.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    arm = arm_of(a.work, HERE)
    order = json.load(open(f"{a.data}/public/order.json"))
    if a.limit_worlds:
        order = order[:a.limit_worlds]
    replay = np.load(f"{a.data}/train/replay.npy")
    # files the child's self-check must fail to open: every data file RUN can see, the model snapshot
    deny = [f"{a.data}/public/order.json", f"{a.data}/train/replay.npy", f"{MODEL_DIR}/config.json", f"{a.model}/config.json"] + \
        [f"{a.data}/public/{d}/{x}.json" for x in order for d in ("worlds", "twins")]
    deny = [p for p in dict.fromkeys(deny) if os.path.isfile(p)]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    engine = Engine(model, tok, dev)
    fp0 = lora.fingerprint(model)
    os.makedirs(f"{a.out}/adapters", exist_ok=True)
    me = os.getpid()

    def check_base(where: str):
        if lora.n_wrapped(model) or lora.fingerprint(model) != fp0:
            raise Fatal(f"base model changed {where}: adapters must not leak between worlds")

    def serve(msg: dict, gen: Gen, trainer: Trainer, teachers: list, budget: Budget) -> dict:
        """One RPC from the child: validated and metered here; the child only ever gets JSON data back."""
        fn, args = msg.get("fn"), msg.get("args")
        try:
            if not isinstance(args, dict):
                raise TypeError("malformed request")
            if fn == "generate":
                if not isinstance(args["prompts"], list):
                    raise TypeError("prompts must be a list of strings")
                out = gen.generate(args["prompts"], args["max_new_tokens"], args["temperature"], bool(args["context"]),
                                   int(args["batch_size"]))
            elif fn == "teacher":
                if not isinstance(args["prompts"], list) or not isinstance(args["completions"], list):
                    raise TypeError("prompts and completions must be lists of strings")
                ts = gen.teacher(args["prompts"], args["completions"], args["k"], bool(args["context"]), int(args["batch_size"]))
                out = list(range(len(teachers), len(teachers) + len(ts)))
                teachers.extend(ts)
            elif fn == "train":
                exs = args["examples"]
                if not isinstance(exs, list):
                    raise TypeError("examples must be a list of dicts")
                conv = []
                for ex in exs:
                    if isinstance(ex, dict) and ex.get("teacher") is not None:
                        t = ex["teacher"]
                        if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(teachers):
                            raise ValueError("example['teacher'] must be a gen.teacher() ref from this world")
                        ex = {**ex, "teacher": teachers[t]}
                    conv.append(ex)
                ad = trainer.train(conv, args.get("config"), args.get("init"))
                out = {"id": ad, "stats": trainer.saved_stats(ad)}
            elif fn == "n_tokens":
                if not isinstance(args["text"], str):
                    raise TypeError("text must be a string")
                out = len(engine.encode(args["text"]))
            else:
                raise ValueError(f"unknown call {fn!r}")
            r = {"ok": True, "result": out}
        except BudgetExceeded as e:
            r = {"ok": False, "kind": "budget", "msg": str(e)[:500]}
        except TypeError as e:
            r = {"ok": False, "kind": "type", "msg": str(e)[:1000]}
        except (ValueError, KeyError, IndexError, OverflowError) as e:
            r = {"ok": False, "kind": "value", "msg": repr(e)[:1000]}
        return {**r, "gen_left": budget.gen_cap - budget.gen_used, "train_left": budget.train_cap - budget.train_used}

    def adapt_world(wid: str, src: str, s: int) -> dict:
        """One adapt() pass in a fresh sandboxed child on the transcript file `src`; the adapter is saved as `wid`."""
        w = json.load(open(src))
        pre = set(descendants(me))
        scratch = tempfile.mkdtemp(prefix=f"pa-{wid}-")    # the child's only writable dir; deleted after this world
        ch = Child(a.work, scratch, deny)
        try:
            hello = ch.recv(time.monotonic() + STARTUP_S)
            sandbox = hello.get("sandbox") if hello.get("op") == "hello" else None
            if not isinstance(sandbox, dict) or sandbox.get("ok") is not True:
                raise Fatal(f"SANDBOX FAILURE in world {wid}: {hello}")
            base_tids = threads(ch.p.pid)

            def strays() -> list:
                return [f"thread {t}" for t in sorted(threads(ch.p.pid) - base_tids)] + \
                    [f"process {p}" for p in descendants(me) if p != ch.p.pid and p not in pre]

            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            t0 = time.monotonic()                          # the surface is imported after this: import time counts
            budget = Budget(a.adapt_seconds, a.gen_tokens, a.train_tokens, start=t0)
            gen = Gen(engine, w["tools"], w["transcript"], budget, s)
            trainer = Trainer(model, engine, w["tools"], replay, budget, s, dev)
            teachers, viol, res, hit, killed = [], [], None, None, False
            ch.send({"op": "go", "tools": w["tools"], "transcript": w["transcript"], "deadline": budget.deadline, "seed": s,
                     "gen_left": budget.gen_cap, "train_left": budget.train_cap})
            while True:
                msg = ch.recv(budget.deadline + a.grace)
                viol += [v for v in strays() if v not in viol]
                if time.monotonic() > budget.deadline + a.grace and msg.get("op") != "done":
                    msg = {"op": "timeout"}                  # stop serving a surface that ignores its deadline
                if msg.get("op") != "call":
                    break
                ch.send(serve(msg, gen, trainer, teachers, budget))
            t_end = time.monotonic()
            viol += [v for v in strays() if v not in viol]
            op = msg.get("op")
            if op == "done":
                if msg.get("adapter") == "last":             # BudgetExceeded escaped adapt(): keep the last adapter
                    res, hit = (trainer.produced[-1] if trainer.produced else None), str(msg.get("budget_hit"))[:300]
                elif msg.get("adapter") is not None:
                    res = msg["adapter"]
                    trainer.saved(res)                       # raises unless it is an id issued in this world
            elif op == "timeout":
                res, hit, killed = (trainer.produced[-1] if trainer.produced else None), f"killed at deadline + {a.grace:.0f} s", True
            elif op == "error":
                raise Fatal(f"adapt() raised in world {wid}:\n{msg.get('msg')}")
            else:
                raise Fatal(f"the surface process of world {wid} failed ({op}, rc={ch.p.poll()})")
        finally:
            ch.kill()
            n_strays = kill_descendants()
            shutil.rmtree(scratch, ignore_errors=True)
        lora.detach(model)
        model.eval()
        if res is not None:
            cfg, tens = trainer.saved(res)
            lora.save(f"{a.out}/adapters/{wid}", cfg, tens)
        return {"id": wid, "donor": w["id"], "adapter": res is not None, "adapt_s": t_end - t0, "killed": killed,
                "gen_tokens": budget.gen_used, "train_tokens": budget.train_used, "budget_hit": hit,
                "adapter_params_m": lora.n_params(tens) / 1e6 if res is not None else 0.0, "train_calls": trainer.calls,
                "nan_seen": trainer.nan_seen, "train": trainer.saved_stats(res) if res is not None else None,
                "teacher_calls": len(teachers), "sandbox": sandbox, "violations": viol, "stray_processes_killed": n_strays}

    per_world, per_twin, t_all = [], [], time.monotonic()
    try:
        for wi, wid in enumerate(order):
            world, twin = f"{a.data}/public/worlds/{wid}.json", f"{a.data}/public/twins/{wid}.json"
            # the placebo reference adapts to the twin (same names, other semantics) and is scored on the target
            rec = adapt_world(wid, twin if arm == "placebo" else world, a.seed * 1000 + wi)
            print(json.dumps(rec), flush=True)
            per_world.append(rec)
            if arm == "adapter":      # the world-specific control: the same surface adapts to the target's twin
                lora.detach(model)
                if lora.n_wrapped(model):
                    raise Fatal("LoRA modules left attached")
                rec = adapt_world(f"{wid}x", twin, a.seed * 1000 + 500 + wi)
                print(json.dumps(rec), flush=True)
                per_twin.append(rec)
            if dev == "cuda":
                torch.cuda.empty_cache()
            check_base(f"after world {wid}")
    except (Fatal, ValueError) as e:
        print(f"HARNESS: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    passes = per_world + per_twin
    for name, obj in (("budget.json", {"adapt_s_max": max((r["adapt_s"] for r in passes), default=0.0)}),
                      ("stats.json", {"arm": arm, "split": a.split, "seed": a.seed, "worlds": per_world, "twins": per_twin,
                                      "run_s": time.monotonic() - t_all, "base_sha256": fp0,
                                      "nan": any(r["nan_seen"] for r in passes),
                                      "violations": sum(len(r["violations"]) for r in passes)})):
        with open(f"{a.out}/{name}.tmp", "w") as f:
            json.dump(obj, f)
        os.replace(f"{a.out}/{name}.tmp", f"{a.out}/{name}")
    print(json.dumps({k: v for k, v in obj.items() if k not in ("worlds", "twins")}), flush=True)


if __name__ == "__main__":
    main()
