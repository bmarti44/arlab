"""Frozen, deterministic generator of straight-line Python "program tracing" problems (no LLM).

A problem is a short program over a few integer variables (arithmetic mod 100, swaps, small ifs, `for range(3)`
loops) that ends in `print(v)`; the answer is the printed value (0..99), found by executing the program.
`steps` counts the statements after the initial assignments (a `for` or `if` block is one step).

DIFFICULTY is the knob the owner tunes in the GPU pilot (target: the no-loop baseline B1 at 40-70% on 2-8 steps).
Changing it changes frozen/run/ and therefore the data_hash: a new data dir and a new campaign tag.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import random

DIFFICULTY = {
    "n_vars": 3,          # variables per program
    "names": "abcdxyzmnk",  # pool the variable names are drawn from
    "const_max": 9,       # constants in updates are 1..const_max
    "ops": "+-*",         # binary ops in assignments ("*" only ever multiplies by a constant 2..const_max)
    "p_swap": 0.10,       # per-step probabilities; the rest are plain assignments
    "p_if": 0.15,
    "p_for": 0.10,
    "for_n": 3,           # iterations of each `for _ in range(for_n)` block
    "p_chain": 0.7,       # probability that a step reads the variable written by the previous step
}
TRAIN_STEPS, ID_STEPS, HARD_STEPS = (2, 8), (2, 8), (9, 12)


def _expr(rng: random.Random, d: dict, src: str, others: list[str]) -> str:
    op = rng.choice(d["ops"])
    if op == "*":
        return f"({src} * {rng.randint(2, max(2, d['const_max']))}) % 100"
    rhs = rng.choice(others) if others and rng.random() < 0.5 else str(rng.randint(1, d["const_max"]))
    return f"({src} {op} {rhs}) % 100"


def make_program(rng: random.Random, steps: int, d: dict = DIFFICULTY) -> tuple[str, str]:
    """Return (program text, printed var). Every value stays in 0..99."""
    names = rng.sample(d["names"], d["n_vars"])
    lines = [f"{v} = {rng.randint(0, 99)}" for v in names]
    last = rng.choice(names)
    for _ in range(steps):
        src = last if rng.random() < d["p_chain"] else rng.choice(names)
        tgt = rng.choice(names)
        others = [v for v in names if v != src]
        r = rng.random()
        if r < d["p_swap"]:
            u = rng.choice([v for v in names if v != tgt])
            lines.append(f"{tgt}, {u} = {u}, {tgt}")
            last = tgt
            continue
        r -= d["p_swap"]
        if r < d["p_if"]:
            cmp_ = rng.choice([v for v in names if v != src] or [str(rng.randint(0, 99))])
            c1, c2 = rng.randint(1, d["const_max"]), rng.randint(1, d["const_max"])
            lines += [f"if {src} > {cmp_}:", f"    {tgt} = ({tgt} + {c1}) % 100", "else:", f"    {tgt} = ({tgt} - {c2}) % 100"]
        elif r - d["p_if"] < d["p_for"]:
            lines += [f"for _ in range({d['for_n']}):", f"    {tgt} = {_expr(rng, d, tgt, [v for v in names if v != tgt])}"]
        else:
            lines.append(f"{tgt} = {_expr(rng, d, src, others)}")
        last = tgt
    lines.append(f"print({last})")
    return "\n".join(lines), last


def run_program(code: str) -> str:
    """Ground truth: execute the (generated, trusted) program and capture what it prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<gen>", "exec"), {"__builtins__": {"print": print, "range": range}})
    return buf.getvalue().strip()


def question(code: str) -> str:
    return f"```python\n{code}\n```\nWhat does this program print? Reply with the number only."


def generate(seed: int, n: int, steps_range: tuple[int, int], d: dict = DIFFICULTY) -> list[dict]:
    """n problems with steps uniform in steps_range (inclusive), fully determined by seed."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        steps = rng.randint(*steps_range)
        code, _ = make_program(rng, steps, d)
        out.append({"code": code, "steps": steps, "answer": run_program(code)})
    return out


def code_hash(code: str) -> str:
    return hashlib.sha256(" ".join(code.split()).encode()).hexdigest()[:16]
