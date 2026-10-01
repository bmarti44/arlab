"""Frozen, deterministic generator of straight-line single-assignment programs (PREPARE-only; RUN never sees it).

A program is N_STMT statements over distinct lowercase letters, arithmetic mod 100, then a query:
    a=37;k=12;b=a+4;c=k-3;d=b+c;...;e?      answer: the value of e (0..99, one token)
Statements are `v=CONST`, `v=u±c` (c in 1..C_MAX) or `v=u±w`. Every program has exactly N_CONST constants and
N_STMT - N_CONST operations, so every program has the same number of pieces (tokens): neither length nor position
can reveal depth.

Depth k of a variable = the longest dependency chain from it down to constants (constants have depth 0;
v=u±c has depth(u)+1; v=u±w has max(depth(u), depth(w))+1). The queried variable is the end of a chain built to
have depth exactly k; the other statements are distractors interleaved at random positions.

DIFFICULTY and the k ranges are the knobs tuned once in the GPU pilot; changing them changes frozen/prepare/ and
therefore the data_hash (new data dir, new campaign tag).
"""
from __future__ import annotations

import hashlib
import random

DIFFICULTY = {
    "n_stmt": 14,     # statements per program (fixed)
    "n_const": 2,     # constant statements per program (fixed; chain root + N_CONST-1 distractor constants)
    "c_max": 9,       # constants in v=u±c are 1..c_max
    "p_bin": 0.3,     # probability that an operation's second operand is a variable (v=u±w)
    "mod": 100,
    "query_min": 11,  # the queried statement is uniform over indices query_min..n_stmt-1 for every k <= 10
    "distractor_max_depth": 6,  # no statement outside the queried chain is deeper than this (in every split)
}
TRAIN_K = (1, 6)      # queried depths seen in training
ID_K = (1, 6)         # eval: in-distribution split
DEPTH_K = (7, 10)     # eval: deeper than any training program (the depth split)
EXT_K = (11, 12)      # eval: reported only
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SYMBOLS = ["=", "+", "-", ";", "?"]
# every piece is one token: letters, numbers 0..99, symbols (PREPARE asserts this for the trained tokenizer)
PIECES = list(LETTERS) + [str(i) for i in range(100)] + SYMBOLS
PIECE_ID = {p: i for i, p in enumerate(PIECES)}


def n_pieces(d: dict = DIFFICULTY) -> int:
    """Pieces in a prompt (statements + `v?`); the answer is one more piece."""
    return 4 * d["n_const"] + 6 * (d["n_stmt"] - d["n_const"]) + 2


def make_program(rng: random.Random, k: int, d: dict = DIFFICULTY) -> dict:
    """One program whose queried variable has depth exactly k. Statements: (v, op, a, b) with op in
    {"c", "+", "-"}; for "c" a is the constant; otherwise a is a variable and b a variable or an int constant.
    Distractors (every statement outside the queried chain) never exceed distractor_max_depth and never read the
    queried variable, in every split, so training programs (k <= 6) contain no statement deeper than 6 and the
    distractor structure of ID and DEPTH programs is the same."""
    cap = d["distractor_max_depth"] - 1   # operands of a distractor operation have depth <= cap
    n = d["n_stmt"]
    if d["n_const"] != 2 or not 1 <= k <= n - 2:
        raise ValueError(f"k={k} impossible (or n_const != 2)")
    names = rng.sample(LETTERS, n)
    # Position plan, identical in distribution for every k <= 10 (no positional depth cue): the two constants are
    # statements 0 and 1 (which one is the chain root is random); the queried statement q is uniform over
    # query_min..n-1; for k >= 2 the query's first operand is statement q-1; the other k-2 chain steps are a uniform
    # random subset of statements 2..q-2; distractors fill the rest. (k 11-12 need q >= k+1: EXT only.)
    q = rng.randint(max(d["query_min"], k + 1), n - 1)
    root = rng.randint(0, 1)
    chain_pos = [root, q] if k == 1 else [root] + sorted(rng.sample(range(2, q - 1), k - 2)) + [q - 1, q]
    slots = ["D"] * n
    slots[1 - root] = "K"
    for pos in chain_pos:
        slots[pos] = "C"
    stmts, depth, chain = [], {}, []
    for slot, v in zip(slots, names):
        if slot == "K" or (slot == "C" and not chain):
            stmt = (v, "c", rng.randint(0, d["mod"] - 1), None)
            depth[v] = 0
        else:
            pool = [x for x in depth if depth[x] <= cap and x != (chain[-1] if len(chain) == k + 1 else None)]
            u = chain[-1] if slot == "C" else rng.choice(pool)
            lim = depth[u] if slot == "C" else cap  # the chain's second operand must not deepen the chain
            cands = [w for w in depth if w != u and depth[w] <= lim and w not in chain[k:]]
            op = rng.choice("+-")
            if cands and rng.random() < d["p_bin"]:
                w = rng.choice(cands)
                stmt, depth[v] = (v, op, u, w), max(depth[u], depth[w]) + 1
            else:
                stmt, depth[v] = (v, op, u, rng.randint(1, d["c_max"])), depth[u] + 1
        if slot == "C":
            chain.append(v)
        stmts.append(stmt)
    assert depth[chain[-1]] == k and stmts[q][0] == chain[-1]
    return {"stmts": stmts, "query": chain[-1], "k": k}


def evaluate(stmts: list, mod: int = 100) -> dict:
    """Execute the program: {var: value}."""
    val = {}
    for v, op, a, b in stmts:
        if op == "c":
            val[v] = a % mod
        else:
            rhs = val[b] if isinstance(b, str) else b
            val[v] = (val[a] + rhs) % mod if op == "+" else (val[a] - rhs) % mod
    return val


def depths(stmts: list) -> dict:
    dep = {}
    for v, op, a, b in stmts:
        dep[v] = 0 if op == "c" else max(dep[a], dep[b] if isinstance(b, str) else 0) + 1
    return dep


def to_text(p: dict) -> str:
    out = []
    for v, op, a, b in p["stmts"]:
        out.append(f"{v}={a}" if op == "c" else f"{v}={a}{op}{b}")
    return ";".join(out) + f";{p['query']}?"


def parse(text: str) -> dict:
    """Inverse of to_text (used by tests on hand-made programs)."""
    body, q = text.rsplit(";", 1)
    assert q.endswith("?")
    stmts = []
    for s in body.split(";"):
        v, rhs = s.split("=")
        if rhs.isdigit():
            stmts.append((v, "c", int(rhs), None))
            continue
        op = "+" if "+" in rhs else "-"
        a, b = rhs.split(op)
        stmts.append((v, op, a, int(b) if b.isdigit() else b))
    return {"stmts": stmts, "query": q[:-1]}


def pieces(p: dict) -> list[str]:
    """Token pieces of the prompt (no BOS, no answer); every piece is exactly one token."""
    out = []
    for v, op, a, b in p["stmts"]:
        out += [v, "=", str(a), ";"] if op == "c" else [v, "=", a, op, str(b), ";"]
    return out + [p["query"], "?"]


def norm_hash(p: dict) -> int:
    """64-bit hash of the program with variables renamed in order of first appearance (alpha-equivalent programs
    collide), used to keep the splits disjoint."""
    ren = {}
    for v, _, a, b in p["stmts"]:
        for x in (v, a, b):
            if isinstance(x, str) and x not in ren:
                ren[x] = LETTERS[len(ren)]
    stmts = [(ren[v], op, ren.get(a, a) if op != "c" else a, ren.get(b, b) if isinstance(b, str) else b)
             for v, op, a, b in p["stmts"]]
    text = to_text({"stmts": stmts, "query": ren[p["query"]]})
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def annotate(p: dict, d: dict = DIFFICULTY) -> dict:
    """Answer + the heuristic-floor predictions the evaluator reports (all values 0..99, -1 = not applicable)."""
    val = evaluate(p["stmts"], d["mod"])
    by_var = {s[0]: s for s in p["stmts"]}
    consts = [s[2] for s in p["stmts"] if s[1] == "c"]
    qs = by_var[p["query"]]
    root = p["query"]
    while by_var[root][1] != "c":  # follow the first operand down to the chain root
        root = by_var[root][2]
    return {**p, "answer": val[p["query"]], "h_last_const": consts[-1],
            "h_own_const": qs[3] if qs[1] != "c" and not isinstance(qs[3], str) else -1,
            "h_root": by_var[root][2]}


def generate(seed: int, n: int, k_range: tuple[int, int], balanced: bool = False, seen: set | None = None,
             d: dict = DIFFICULTY) -> list[dict]:
    """n annotated programs, fully determined by seed (and `seen`). balanced: k cycles through k_range (equal counts).
    seen: normalized hashes already used (other splits); such programs are redrawn, new hashes are added."""
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


def counterfactual(p: dict, rng: random.Random, d: dict = DIFFICULTY, tries: int = 20) -> dict | None:
    """The same program with the chain-root constant shifted by a nonzero delta (mod 100) such that the ANSWER
    changes (roots whose contribution cancels are rejected); None if no tried delta changes it."""
    by_var = {s[0]: i for i, s in enumerate(p["stmts"])}
    root = p["query"]
    while p["stmts"][by_var[root]][1] != "c":
        root = p["stmts"][by_var[root]][2]
    i = by_var[root]
    for _ in range(tries):
        delta = rng.choice([x for x in range(-9, 10) if x])
        stmts = list(p["stmts"])
        v, _, c, _ = stmts[i]
        stmts[i] = (v, "c", (c + delta) % d["mod"], None)
        twin = annotate({"stmts": stmts, "query": p["query"], "k": p["k"]}, d)
        if twin["answer"] != p["answer"]:
            return twin
    return None
