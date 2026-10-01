"""Frozen, deterministic generator of ordered permutation-composition records (PREPARE-only; RUN never sees it).

v2 task (2026-10-01; the v1 arithmetic programs were not learnable at pack scale, see BLOCKED.md B2 and FIX-sol.md).
A record is `INIT s0 t1 ... t14 ?`: an initial state s0 in {0..4}, then N_SLOTS operator slots. Exactly k slots hold
an active operator, one of the 44 permutations of 5 states with no fixed point; the rest hold NOP. The answer is s0
pushed through the active operators in order (one state token). Depth k = the number of active operators, i.e. the
number of serial composition steps. The length is fixed (17 pieces + BOS), so neither length nor position reveals
depth. The NOP count does reveal k; that is accepted and documented.

Training is densely supervised through targets, never through inputs: the state after each active operator is the
target at that operator's position, and the final state is the target at `?` (prepare.py writes these targets to a
separate stream). Intermediate states never appear in any input.
"""
from __future__ import annotations

import hashlib
import itertools
import random

N_STATES = 5
DERANGEMENTS = [p for p in itertools.permutations(range(N_STATES)) if all(p[i] != i for i in range(N_STATES))]  # 44
DIFFICULTY = {"n_slots": 14, "n_states": N_STATES, "n_ops": len(DERANGEMENTS)}
TRAIN_K = (1, 6)      # active operators seen in training
ID_K = (1, 6)         # eval: in-distribution split
DEPTH_K = (7, 10)     # eval: deeper than any training record (the depth split)
EXT_K = (11, 12)      # eval: reported only
STATES = [str(i) for i in range(N_STATES)]                      # pieces "0".."4"
OPS = [str(10 + i) for i in range(len(DERANGEMENTS))]          # pieces "10".."53"
NOP, INIT, QUERY = "-", "=", "?"
# every piece is one token (PREPARE asserts this for the trained tokenizer; unmerged numbers get reserved tokens)
PIECES = STATES + OPS + [NOP, INIT, QUERY]
PIECE_ID = {p: i for i, p in enumerate(PIECES)}


def n_pieces(d: dict = DIFFICULTY) -> int:
    """Pieces in a prompt (INIT s0 slots ?); the answer is the target at `?`, not an input piece."""
    return 2 + d["n_slots"] + 1


def make_program(rng: random.Random, k: int, d: dict = DIFFICULTY) -> dict:
    """One record with exactly k active operators at uniformly random slots."""
    n = d["n_slots"]
    if not 1 <= k <= n:
        raise ValueError(f"k={k} impossible")
    slots = [-1] * n
    for pos in rng.sample(range(n), k):
        slots[pos] = rng.randrange(len(DERANGEMENTS))
    return {"s0": rng.randrange(N_STATES), "slots": slots, "k": k}


def run(p: dict) -> list:
    """The state after each slot (None for NOP slots); the last non-None value is the answer."""
    s, out = p["s0"], []
    for o in p["slots"]:
        if o < 0:
            out.append(None)
        else:
            s = DERANGEMENTS[o][s]
            out.append(s)
    return out


def answer(p: dict) -> int:
    s = p["s0"]
    for o in p["slots"]:
        if o >= 0:
            s = DERANGEMENTS[o][s]
    return s


def annotate(p: dict, d: dict = DIFFICULTY) -> dict:
    """Answer + the heuristic-floor predictions the evaluator reports (all states 0..4):
    h_last_const = only the LAST active operator applied to s0; h_own_const = s0 itself (no operators);
    h_root = only the FIRST active operator applied to s0."""
    act = [o for o in p["slots"] if o >= 0]
    return {**p, "answer": answer(p), "h_last_const": DERANGEMENTS[act[-1]][p["s0"]], "h_own_const": p["s0"],
            "h_root": DERANGEMENTS[act[0]][p["s0"]]}


def pieces(p: dict) -> list[str]:
    """Token pieces of the prompt (no BOS, no answer); every piece is exactly one token."""
    return [INIT, STATES[p["s0"]]] + [OPS[o] if o >= 0 else NOP for o in p["slots"]] + [QUERY]


def targets(p: dict) -> list:
    """Supervision aligned with pieces(): the state piece after each active operator (at that operator's position),
    the final state at `?`, None elsewhere (INIT, s0, NOP)."""
    st = run(p)
    return [None, None] + [STATES[s] if s is not None else None for s in st] + [STATES[answer(p)]]


def to_text(p: dict) -> str:
    return " ".join(pieces(p))


def parse(text: str) -> dict:
    """Inverse of to_text."""
    xs = text.split()
    assert xs[0] == INIT and xs[-1] == QUERY
    slots = [-1 if x == NOP else OPS.index(x) for x in xs[2:-1]]
    return {"s0": STATES.index(xs[1]), "slots": slots, "k": sum(o >= 0 for o in slots)}


def norm_hash(p: dict) -> int:
    """64-bit hash of the record (exact; there are no names to normalize), used to keep the splits disjoint."""
    return int.from_bytes(hashlib.sha256(to_text(p).encode()).digest()[:8], "little")


def counterfactual(p: dict, rng: random.Random, d: dict = DIFFICULTY, tries: int = 20) -> dict | None:
    """The same operators with a different s0. The composition is a bijection, so the answer always changes."""
    s = rng.choice([x for x in range(N_STATES) if x != p["s0"]])
    twin = annotate({"s0": s, "slots": list(p["slots"]), "k": p["k"]}, d)
    return twin if twin["answer"] != p["answer"] else None


def generate(seed: int, n: int, k_range: tuple[int, int], balanced: bool = False, seen: set | None = None,
             d: dict = DIFFICULTY) -> list[dict]:
    """n annotated records, fully determined by seed (and `seen`). balanced: k cycles through k_range (equal counts).
    seen: hashes already used (other splits); such records are redrawn, new hashes are added."""
    rng = random.Random(seed)
    seen = set() if seen is None else seen
    lo, hi = k_range
    out = []
    for i in range(n):
        k = lo + i % (hi - lo + 1) if balanced else rng.randint(lo, hi)
        while True:
            p = make_program(rng, k, d)
            h = norm_hash(p)
            if h not in seen:
                break
        seen.add(h)
        out.append({**annotate(p, d), "hash": h})
    return out
