"""EVALUATE model worker: the ONLY process in EVALUATE that runs surface code (started by frozen/eval/evalcore.py).

Order: import torch, initialize CUDA, load the data-only checkpoint (weights_only), map the logits memfd -> Landlock
lockdown (readable: the Python install, /usr, /etc, /proc, /sys, this directory and --work; read/write: /dev for the
GPU and its own scratch dir; nothing else: no eval data, no evaluator code, no result dir, no TCP, no signals or
ptrace-style access to the evaluator) -> self-check (the --deny files and /proc/<parent>/mem must not open, /tmp must
not be writable) -> hello -> import /work/train.py -> make_model(config) -> copy the checkpoint tensors into its
parameters/buffers (names, shapes and dtypes must match exactly) -> serve forward requests.

Request: a JSON line {"op": "fwd", "shape": [B, T], "last": bool} followed by B*T int32 token ids on stdin (the worker
builds its own tensor from those bytes: nothing it receives aliases evaluator memory). Reply: a JSON line
{"op": "out", "shape": [...], "flops": n, "bad_ops": [...]} with the float32 logits (all positions, or only the last
one if "last") written at the start of the memfd. The evaluator reads, checks and scores them itself.
"""
import argparse
import json
import mmap
import os
import site
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.flop_counter  # noqa: E402,F401  (imported before the lockdown)
import torch.utils._python_dispatch  # noqa: E402,F401

from common import VOCAB, load_checkpoint, model_tensors, tensors_hash  # noqa: E402
from meter import metered  # noqa: E402
from sandbox import blocked, lockdown  # noqa: E402

READ_DIRS = ["/usr", "/lib", "/lib64", "/etc", "/opt", "/proc", "/sys"]


def python_dirs() -> list:
    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.dirname(torch.__file__)),
            os.path.dirname(os.path.dirname(np.__file__))}
    for d in site.getsitepackages() + [site.getusersitepackages()]:
        dirs.add(d)
    return sorted(d for d in dirs if d and os.path.isdir(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--memfd", type=int, required=True)
    ap.add_argument("--memfd-size", type=int, required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--deny", action="append", default=[], help="files the self-check must fail to open")
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer

    def send(obj):
        proto.write(json.dumps(obj) + "\n")

    dev = a.device
    if dev == "cuda":
        torch.zeros(1, device=dev)
    try:
        ck = load_checkpoint(a.ckpt)
    except ValueError as e:
        send({"op": "hello", "sandbox": {"ok": False, "error": str(e)}})
        return
    mm = mmap.mmap(a.memfd, a.memfd_size)
    out = torch.frombuffer(mm, dtype=torch.float32)
    try:
        abi = lockdown(READ_DIRS + python_dirs() + [HERE, a.work], [a.scratch], ["/dev"])
    except OSError as e:
        send({"op": "hello", "sandbox": {"ok": False, "error": repr(e)}})
        return
    tempfile.tempdir = a.scratch
    checks = {"deny": bool(a.deny) and all(blocked(p) for p in a.deny),
              "procmem": blocked(f"/proc/{os.getppid()}/mem"), "write": blocked("/tmp/.la_sandbox_probe", "wb")}
    send({"op": "hello", "sandbox": {"ok": all(checks.values()), "landlock_abi": abi, **checks}})
    if not all(checks.values()):
        return
    try:
        cfg = json.loads(stdin.readline())["config"]
        sys.path.insert(0, a.work)
        import train as surface
        model = surface.make_model(dict(cfg))
        if not isinstance(model, torch.nn.Module):
            raise TypeError("make_model(config) must return a torch.nn.Module")
        have = dict(list(model.named_parameters()) + list(model.named_buffers()))
        if set(have) != set(ck):
            raise ValueError(f"checkpoint names differ from make_model()'s parameters/buffers: "
                             f"missing {sorted(set(have) - set(ck))[:5]}, unexpected {sorted(set(ck) - set(have))[:5]}")
        with torch.no_grad():
            for name, t in have.items():
                if t.shape != ck[name].shape or t.dtype != ck[name].dtype:
                    raise ValueError(f"{name}: checkpoint {ck[name].dtype}{tuple(ck[name].shape)} vs model "
                                     f"{t.dtype}{tuple(t.shape)}")
                t.copy_(ck[name])
        del ck
        model.eval()
        loaded_hash = tensors_hash(model_tensors(model))
        if dev == "cpu":
            model.float()  # CPU tests: no bf16 autocast on CPU, so run everything in fp32
        send({"op": "ready", "tensors_hash": loaded_hash})
        while True:
            line = stdin.readline()
            if not line:
                return
            req = json.loads(line)
            if req.get("op") != "fwd":
                return
            B, T = req["shape"]
            raw = stdin.read(B * T * 4)
            x = torch.from_numpy(np.frombuffer(raw, dtype=np.int32).astype(np.int64).reshape(B, T)).to(dev)
            lg, flops, bad = metered(model, x)
            if not isinstance(lg, torch.Tensor) or tuple(lg.shape) != (B, T, VOCAB):
                raise ValueError(f"model(x) must return logits (B, T, {VOCAB}) for x {(B, T)}; "
                                 f"got {getattr(lg, 'shape', type(lg))}")
            lg = lg[:, -1:] if req.get("last") else lg
            n = lg.numel()
            if n > out.numel():
                raise ValueError("reply larger than the logits buffer")
            out[:n].view(lg.shape).copy_(lg.float())
            send({"op": "out", "shape": list(lg.shape), "flops": flops, "bad_ops": bad})
    except Exception:
        send({"op": "error", "msg": traceback.format_exc()[-3000:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
