"""RUN trainer child (frozen code, but it shares its interpreter with the surface, so nothing here is trusted):
started by harness.py. Loads torch and the data, reports ready, waits for the go message carrying the absolute
CLOCK_MONOTONIC deadline, then imports the surface and trains until the deadline (cooperative stop; the supervisor
kills this process group at deadline + GRACE_S regardless). Finally it writes the data-only checkpoint
{name: tensor} of state["model"] to --ckpt and reports done. JSON lines on the original stdout; anything the
surface prints goes to stderr (the run log).

Batches: 64 rows x 1024 tokens = TEXT_ROWS random windows of the relabelled text stream + PROG_ROWS rows of packed
programs (random BOS-aligned program start), in a seeded random row order. Plain (x, y) next-token pairs.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import PROG_ROWS, SEQ_LEN, TEXT_ROWS, VOCAB, model_tensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-seconds", type=float, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj) + "\n")

    dev = "cpu" if a.smoke else "cuda"
    text_rows, prog_rows, seq_len = (3, 1, 128) if a.smoke else (TEXT_ROWS, PROG_ROWS, SEQ_LEN)
    batch = text_rows + prog_rows
    text = np.memmap(f"{a.data}/tokens.bin", dtype=np.uint16, mode="r")
    progs = np.memmap(f"{a.data}/programs.bin", dtype=np.uint16, mode="r")
    starts = np.load(f"{a.data}/program_starts.npy")
    starts = starts[starts <= len(progs) - seq_len - 1]
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    buf = torch.empty((batch, seq_len + 1), dtype=torch.long)
    if dev == "cuda":
        torch.cuda.manual_seed(a.seed)
        torch.zeros(1, device=dev)
        buf = buf.pin_memory()

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def next_batch():
        t0s = rng.integers(0, len(text) - seq_len - 1, text_rows)
        p0s = starts[rng.integers(0, len(starts), prog_rows)]
        rows = [text[s:s + seq_len + 1] for s in t0s] + [progs[s:s + seq_len + 1] for s in p0s]
        order = rng.permutation(batch)
        buf.copy_(torch.from_numpy(np.stack([rows[i] for i in order]).astype(np.int64)))
        xy = buf.to(dev, non_blocking=True)
        return xy[:, :-1], xy[:, 1:]

    send({"op": "ready"})
    go = json.loads(sys.stdin.readline())
    deadline, t0 = go["deadline"], go["t0"]
    try:
        sys.path.insert(0, a.work)
        import train as surface  # the editable surface (the supervisor's clock is already running)

        state = surface.build({"vocab_size": VOCAB, "seq_len": seq_len, "batch": batch, "device": dev,
                               "seed": a.seed, "train_seconds": a.train_seconds})
        if not isinstance(state, dict) or not isinstance(state.get("model"), torch.nn.Module):
            raise TypeError('build() must return a dict whose "model" is the torch.nn.Module to checkpoint')
        sync()
        build_s = time.monotonic() - t0
        step, nan_at, first_step_s, recent = 0, None, None, []
        while True:
            sync()
            now = time.monotonic()
            if now >= deadline:
                break
            x, y = next_batch()
            loss = surface.train_step(state, (x, y), step, (now - t0) / a.train_seconds)
            step += 1
            lv = float(loss)
            if first_step_s is None:
                first_step_s = time.monotonic() - t0
            recent = (recent + [lv])[-50:]
            if not math.isfinite(lv):
                nan_at = step
                print(f"non-finite loss at step {step}; stopping", flush=True)
                break
            if step % 50 == 0:
                print(f"step {step} {now - t0:.0f}s loss {sum(recent) / len(recent):.4f}", flush=True)
        sync()
        loop_end_s = time.monotonic() - t0
        torch.save(model_tensors(state["model"]), a.ckpt)
        send({"op": "done", "train_steps": step, "tokens_seen": step * batch * seq_len,
              "program_tokens_seen": step * prog_rows * seq_len, "build_s": build_s, "first_step_s": first_step_s,
              "loop_end_s": loop_end_s, "nan_at": nan_at, "train_loss_last50": sum(recent) / max(1, len(recent)),
              "batch": [text_rows, prog_rows, seq_len]})
    except Exception:
        send({"op": "error", "msg": traceback.format_exc()[-3000:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
