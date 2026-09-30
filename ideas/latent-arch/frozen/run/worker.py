"""EVALUATE model worker: the ONLY process in EVALUATE that runs surface code (started by frozen/eval/evalcore.py).

Order: import torch, initialize CUDA, load the data-only checkpoint (weights_only), map the logits memfd -> Landlock
lockdown (readable: the Python install, /usr, /etc, /proc, /sys, this directory and --work; read/write: /dev for the
GPU and its own scratch dir; nothing else: no eval data, no evaluator code, no result dir, no TCP, no signals or
ptrace-style access to the evaluator) -> self-check (the --deny files and /proc/<parent>/mem must not open, /tmp must
not be writable) -> hello -> import /work/train.py -> make_model(config) -> copy the checkpoint tensors into its
parameters/buffers (names, shapes and dtypes must match exactly) -> eval() -> report the tensor hash, the
hidden-state inventory (incl. storage bytes not covered by registered tensors) and new OS threads/child processes ->
serve forward requests; a "check" request re-reports all three (after scoring). Before the surface is imported, torch's
own thread pools are warmed up, the OS thread set is recorded, and every Python-level thread/process start is
disabled: the metering dispatch modes are thread-local, so all surface computation must run on the calling thread
(checked again at OS level after every forward).

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
import types

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


def hidden_state(model: torch.nn.Module, work: str) -> list:
    """Tensors / arrays / large byte buffers reachable from the model's module attributes or from the globals of
    the surface's own modules (files under --work) that are NOT its registered parameters/buffers (or views of
    them). All tensor state must be registered (so it is checkpointed, counted and hashed)."""
    reg = list(model.parameters()) + list(model.buffers())
    ids = {id(t) for t in reg}
    ptrs = {t.untyped_storage().data_ptr() for t in reg if t.numel()}
    mods = {n for n, m in list(sys.modules.items())
            if os.path.abspath(getattr(m, "__file__", None) or "/").startswith(os.path.abspath(work) + os.sep)}
    found, seen = [], set()

    def visit(obj, path, depth):
        if id(obj) in seen or depth > 8 or len(found) > 20:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if id(obj) not in ids and obj.numel() > 1 and obj.untyped_storage().data_ptr() not in ptrs:
                found.append(f"{path}: tensor {tuple(obj.shape)}")
        elif isinstance(obj, np.ndarray):
            if obj.size > 1:
                found.append(f"{path}: ndarray {obj.shape}")
        elif isinstance(obj, (bytes, bytearray, memoryview)):
            if len(obj) > 1 << 20:
                found.append(f"{path}: {type(obj).__name__} of {len(obj)} bytes")
        elif isinstance(obj, (str, int, float, complex, bool, type(None), types.ModuleType)):
            return
        elif isinstance(obj, dict):
            for k, v in list(obj.items()):
                visit(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for i, v in enumerate(list(obj)):
                visit(v, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, torch.nn.Module):
            for k, v in vars(obj).items():
                if k not in ("_parameters", "_buffers"):
                    visit(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, (types.FunctionType, types.MethodType)):
            fn = getattr(obj, "__func__", obj)
            if fn.__module__ in mods:
                for i, c in enumerate(fn.__closure__ or ()):
                    try:
                        visit(c.cell_contents, f"{path}.<closure {i}>", depth + 1)
                    except ValueError:
                        pass
                visit(fn.__defaults__, f"{path}.<defaults>", depth + 1)
                visit(fn.__kwdefaults__, f"{path}.<kwdefaults>", depth + 1)
        elif isinstance(obj, type):
            if obj.__module__ in mods:
                visit(dict(vars(obj)), f"{path}.<class>", depth + 1)
        elif type(obj).__module__ in mods and hasattr(obj, "__dict__"):
            visit(vars(obj), f"{path}.<obj>", depth + 1)

    visit(model, "model", 0)
    for n in sorted(mods):
        visit(dict(vars(sys.modules[n])), f"module {n}", 0)
    return found


def uncovered_storage(model: torch.nn.Module) -> list:
    """Every byte of every storage behind a registered parameter/buffer must belong to a registered, contiguous
    tensor (so it is checkpointed, counted and hashed): registering backing[:1] and keeping backing[1:] as a
    hidden cache is rejected."""
    by = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        st = t.untyped_storage()
        if st.nbytes() == 0:
            continue
        e = by.setdefault(st.data_ptr(), [st.nbytes(), [], name])
        if t.is_contiguous():
            o = t.storage_offset() * t.element_size()
            e[1].append((o, o + t.numel() * t.element_size()))
    bad = []
    for n, spans, name in by.values():
        end = 0
        for a, b in sorted(spans):
            if a > end:
                break
            end = max(end, b)
        if end < n:
            bad.append(f"{name}: storage of {n} bytes, only the first {end} covered by registered tensors")
    return bad


def _forbidden(*args, **kwargs):
    raise RuntimeError("threads and child processes are not allowed in the evaluation worker "
                       "(FLOPs and ops are metered on the calling thread only)")


def forbid_threads_and_processes():
    """Before the surface is imported: every Python-level way to start a thread or a process raises."""
    import _posixsubprocess
    import _thread
    import threading
    import subprocess
    for mod, names in ((_thread, ("start_new_thread", "start_new", "start_joinable_thread")),
                       (threading, ("_start_new_thread", "_start_joinable_thread")),
                       (_posixsubprocess, ("fork_exec",)), (subprocess, ("_fork_exec",)),
                       (os, ("fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen", "register_at_fork"))):
        for n in names:
            if hasattr(mod, n):
                setattr(mod, n, _forbidden)


def os_tasks() -> tuple:
    """(thread ids of this process, child process ids) as seen by the kernel."""
    tids = set(os.listdir("/proc/self/task"))
    kids = set()
    for t in tids:
        try:
            kids |= set(open(f"/proc/self/task/{t}/children").read().split())
        except OSError:
            pass
    return tids, kids


def warm_torch(dev: str):
    """Run the common kernels once so torch's own thread pools (OpenMP / CUDA) exist before the baseline is taken."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(4, 256, 512, generator=g).to(dev)
    w = torch.randn(512, 512, generator=g).to(dev)
    with torch.no_grad(), torch.autocast(torch.device(dev).type, dtype=torch.bfloat16):
        for _ in range(2):
            h = torch.nn.functional.layer_norm(x @ w, (512,)).softmax(-1)
            q = h.view(4, 256, 8, 64).transpose(1, 2)
            h = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.nn.functional.embedding(torch.randint(0, 512, (4, 256), generator=g).to(dev), w).float().cumsum(1).sum()
    big = torch.ones(1 << 22)  # a parallel CPU op: torch starts its OpenMP pool lazily (19 threads on 20 cores)
    (big * 2).sum()
    torch.randn(256, 256, generator=g) @ torch.randn(256, 256, generator=g)
    if dev == "cuda":
        torch.cuda.synchronize()


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
        warm_torch(dev)
        forbid_threads_and_processes()
        base_tids, _ = os_tasks()

        def new_tasks() -> list:
            tids, kids = os_tasks()
            return [f"thread {t}" for t in sorted(tids - base_tids)] + [f"child process {k}" for k in sorted(kids)]

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

        def state():
            return {"tensors_hash": tensors_hash(model_tensors(model)),
                    "hidden": hidden_state(model, a.work) + uncovered_storage(model), "threads": new_tasks()}

        send({"op": "ready", **state()})
        while True:
            line = stdin.readline()
            if not line:
                return
            req = json.loads(line)
            if req.get("op") == "check":
                send({"op": "state", **state()})
                continue
            if req.get("op") != "fwd":
                return
            B, T = req["shape"]
            raw = stdin.read(B * T * 4)
            x = torch.from_numpy(np.frombuffer(raw, dtype=np.int32).astype(np.int64).reshape(B, T)).to(dev)
            lg, flops, bad = metered(model, x)
            if new_tasks():
                raise RuntimeError(f"threads/processes started by surface code: {new_tasks()[:5]}")
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
