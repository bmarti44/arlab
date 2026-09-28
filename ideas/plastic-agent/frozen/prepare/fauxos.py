"""FauxOS: a procedurally generated fictional tool world (deterministic, pure Python, no LLM inside).

This file is PREPARE/EVALUATE-only. It lives in frozen/prepare/ (mounted only into PREPARE) and PREPARE copies it into
each split's private/ dir for the evaluator. RUN never sees it: the surface gets only the exploration transcript
(calls + observations) and the tool names, so the semantics below can only be learned from what the transcript shows.

A world ("instance") is a spec generated from one integer seed:
  - 4 kinds, 4 places, 6 tags, two weight units (display unit U, base unit B, 1 U = C B, C in 2..9), 3 error codes
    (locked / missing / bad-argument), 30-60 objects with hidden fields;
  - one tool per operation in OPS plus 1-3 decoys (a second tool for a list/count/extreme op with another variant),
    each with a pseudo-word name, a random positional argument order and per-op semantic variants; tool verbs are
    pseudo-words, truthful English verbs, or misleading English verbs ("purge" that archives) -- the "drive on the left".
Programs: one call per line, `name(arg, ...)`, positional int / "string" literals only, <= MAX_CALLS lines.
The output of the last line is the answer; the builtin answer(x) returns x. The first error stops the program.
"""
from __future__ import annotations

import ast
import copy
import json
import random
import re

MAX_CALLS = 12
CONS, VOWS = "bdfgklmnprstvz", "aeiou"

# op -> (canonical args [(name, type)], mutating)
OPS = {
    "list_place": ([("place", "place")], False),
    "list_kind": ([("kind", "kind")], False),
    "inspect": ([("id", "id")], False),
    "count": ([("kind", "kind"), ("place", "place")], False),
    "weigh": ([("kind", "kind")], False),
    "extreme": ([("place", "place")], False),
    "newest": ([("kind", "kind")], False),
    "find_tag": ([("tag", "tag")], False),
    "checksum": ([("place", "place")], False),
    "archived": ([], False),
    "census": ([], False),
    "convert": ([("n", "num")], False),
    "move": ([("id", "id"), ("place", "place")], True),
    "archive": ([("id", "id")], True),
    "delete": ([("id", "id")], True),
    "restore": ([("id", "id")], True),
    "lock": ([("id", "id")], True),
    "unlock": ([("id", "id")], True),
    "retag": ([("id", "id"), ("tag", "tag")], True),
    "archive_kind": ([("kind", "kind"), ("threshold", "num")], True),
    "move_kind": ([("kind", "kind"), ("src", "place"), ("dst", "place")], True),
    "lock_place": ([("place", "place")], True),
    "clone": ([("id", "id")], True),
    "swap": ([("a", "id"), ("b", "id")], True),
    "purge_archive": ([], True),
}
DECOY_OPS = ("list_place", "list_kind", "count", "extreme", "newest", "checksum", "weigh")
# English verbs: truthful ones a pretrained model would guess right, misleading ones it would guess wrong.
TRUE_VERBS = {"list_place": ["list", "show"], "list_kind": ["list", "show"], "inspect": ["inspect", "info"],
              "count": ["count", "tally"], "weigh": ["weigh", "mass"], "extreme": ["top", "pick"],
              "newest": ["latest", "recent"], "find_tag": ["find", "search"], "checksum": ["checksum", "digest"],
              "archived": ["archived", "vault"], "census": ["census", "stats"], "convert": ["convert", "units"],
              "move": ["move", "send"], "archive": ["archive", "shelve"], "delete": ["delete", "remove"],
              "restore": ["restore", "revive"], "lock": ["lock", "seal"], "unlock": ["unlock", "open"],
              "retag": ["tag", "label"], "archive_kind": ["archive_all", "bulk_archive"], "move_kind": ["move_all", "migrate"],
              "lock_place": ["lock_all", "seal_all"], "clone": ["clone", "copy"], "swap": ["swap", "switch"],
              "purge_archive": ["purge", "empty"]}
FALSE_VERBS = {"list_place": ["count", "sum"], "list_kind": ["delete", "weigh"], "inspect": ["erase", "move"],
               "count": ["list", "weigh"], "weigh": ["count", "lock"], "extreme": ["oldest", "random"],
               "newest": ["heaviest", "first"], "find_tag": ["delete", "clone"], "checksum": ["count", "undo"],
               "archived": ["active", "live"], "census": ["purge", "reset"], "convert": ["delete", "melt"],
               "move": ["copy", "melt"], "archive": ["purge", "burn"], "delete": ["stash", "keep"],
               "restore": ["drop", "bury"], "lock": ["open", "free"], "unlock": ["seal", "bolt"],
               "retag": ["wipe", "move"], "archive_kind": ["restore_all", "count_all"], "move_kind": ["copy_all", "lock_all"],
               "lock_place": ["open_all", "free_all"], "clone": ["remove", "merge"], "swap": ["join", "split"],
               "purge_archive": ["restore", "backup"]}
VERB_MIX = (0.25, 0.30)          # P(truthful English verb), P(misleading English verb); rest: pseudo-word verb
VERBOSE_P = 0.5                  # P(a mutating tool reports its effect); otherwise it prints only "ok"
N_OBJ = (30, 60)


class SimError(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind          # "locked" | "missing" | "badarg"


class ProgramError(Exception):
    pass


# ------------------------------------------------------------------ generator
def pseudo(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(CONS) + rng.choice(VOWS) + (rng.choice(CONS) if rng.random() < 0.5 else "") for _ in range(n))


def _fresh(rng, used: set, make):
    for _ in range(1000):
        w = make()
        if w not in used and len(w) >= 3:
            used.add(w)
            return w
    raise RuntimeError("name space exhausted")


def gen_world(seed: int) -> dict:
    """Spec (private) of one world. Deterministic in `seed`."""
    rng = random.Random(f"fauxos-{seed}")
    used: set[str] = set()
    kinds = [_fresh(rng, used, lambda: pseudo(rng, 1).upper()) for _ in range(4)]
    places = [_fresh(rng, used, lambda: pseudo(rng, 2)) for _ in range(4)]
    tags = [_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(6)]
    unit, base_unit = (_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(2))
    C = rng.randint(2, 9)
    letter = rng.choice("ABDEFGHJKMNPQRTVWXYZ")
    codes = rng.sample(range(10, 100), 3)
    errors = {k: f"{letter}{c}" for k, c in zip(("locked", "missing", "badarg"), codes)}

    ops = list(OPS) + rng.sample(DECOY_OPS, rng.randint(1, 3))
    tools, names = [], set()
    for op in ops:
        u = rng.random()
        verbs = TRUE_VERBS[op] if u < VERB_MIX[0] else FALSE_VERBS[op] if u < sum(VERB_MIX) else None
        verb = rng.choice(verbs) if verbs else pseudo(rng, rng.randint(1, 2))
        name = _fresh(rng, names, lambda: f"{pseudo(rng, 1)}_{verb}")
        n_args = len(OPS[op][0])
        perm = list(range(n_args))
        rng.shuffle(perm)
        tools.append({"name": name, "op": op, "perm": perm, "var": _variant(rng, op), "verbose": rng.random() < VERBOSE_P})
    # the decoy must differ from the primary tool of its op (otherwise it is just an alias)
    for t in tools:
        prim = next(x for x in tools if x["op"] == t["op"])
        while t is not prim and t["var"] == prim["var"]:
            t["var"] = _variant(rng, t["op"])

    n = rng.randint(*N_OBJ)
    ids = rng.sample(range(100, 1000), n)
    made = list(range(1, n + 1))
    rng.shuffle(made)
    objs = {}
    for i, m in zip(ids, made):
        objs[str(i)] = {"kind": rng.choice(kinds), "place": rng.choice(places), "w": rng.randint(1, 9),
                        "locked": rng.random() < 0.2, "state": "archived" if rng.random() < 0.15 else "active",
                        "made": m, "tag": rng.choice(tags) if rng.random() < 0.5 else "-"}
    init = {"objs": objs, "clock": n, "next_id": 1000}
    return {"seed": seed, "kinds": kinds, "places": places, "tags": tags, "unit": unit, "base_unit": base_unit, "C": C,
            "errors": errors, "tools": tools, "init": init}


def _variant(rng, op) -> dict:
    v = {}
    if op in ("list_place", "list_kind"):
        v = {"order": rng.choice(["new", "old", "id"]), "skip_locked": rng.random() < 0.5}
    elif op == "count":
        v = {"skip_locked": rng.random() < 0.5}
    elif op == "weigh":
        v = {"unit": rng.choice(["display", "base"])}
    elif op == "extreme":
        v = {"which": rng.choice(["heaviest", "lightest"])}
    elif op == "newest":
        v = {"which": rng.choice(["newest", "oldest"])}
    elif op == "checksum":
        v = {"mul": rng.choice([3, 7, 11]), "mod": rng.choice([89, 97]), "with_locked": rng.random() < 0.5}
    elif op == "convert":
        v = {"dir": rng.choice(["d2b", "b2d"])}
    elif op == "lock":
        v = {"toggle": rng.random() < 0.4}
    elif op == "archive_kind":
        v = {"unit": rng.choice(["display", "base"]), "strict": rng.random() < 0.5}
    return v


# ------------------------------------------------------------------ simulator
def tool_map(spec: dict) -> dict:
    return {t["name"]: t for t in spec["tools"]}


def fmt_call(name: str, args: list) -> str:
    return f"{name}(" + ", ".join(json.dumps(a) if isinstance(a, str) else str(a) for a in args) + ")"


def _ids(xs) -> str:
    return ", ".join(str(x) for x in xs) if xs else "(none)"


def _active(state, pred=lambda o: True) -> list[tuple[int, dict]]:
    return sorted(((int(i), o) for i, o in state["objs"].items() if o["state"] == "active" and pred(o)), key=lambda x: x[0])


def _obj(state, i, active=True) -> dict:
    o = state["objs"].get(str(i))
    if o is None or (active and o["state"] != "active"):
        raise SimError("missing")
    return o


def _unlocked(o):
    if o["locked"]:
        raise SimError("locked")
    return o


def _order(items, order):
    if order == "new":
        return [i for i, o in sorted(items, key=lambda x: -x[1]["made"])]
    if order == "old":
        return [i for i, o in sorted(items, key=lambda x: x[1]["made"])]
    return [i for i, _ in items]


def apply(spec: dict, state: dict, tool: dict, a: dict) -> str:
    """Execute one op on `state` in place; returns the observation. Raises SimError."""
    op, v, s = tool["op"], tool["var"], spec
    verb = tool["verbose"]
    if op in ("list_place", "list_kind"):
        key = "place" if op == "list_place" else "kind"
        items = _active(state, lambda o: o[key] == a[key] and not (v["skip_locked"] and o["locked"]))
        return f"{a[key]}: {_ids(_order(items, v['order']))}"
    if op == "inspect":
        o = _obj(state, a["id"], active=False)
        return (f"#{a['id']} kind={o['kind']} place={o['place']} weight={o['w']} {s['unit']} made={o['made']} "
                f"locked={'yes' if o['locked'] else 'no'} state={o['state']} tag={o['tag']}")
    if op == "count":
        return f"count: {len(_active(state, lambda o: o['kind'] == a['kind'] and o['place'] == a['place'] and not (v['skip_locked'] and o['locked'])))}"
    if op == "weigh":
        tot = sum(o["w"] for _, o in _active(state, lambda o: o["kind"] == a["kind"]))
        return f"total: {tot} {s['unit']}" if v["unit"] == "display" else f"total: {tot * s['C']} {s['base_unit']}"
    if op == "extreme":
        items = _active(state, lambda o: o["place"] == a["place"])
        if not items:
            return f"{v['which']}: (none)"
        sign = -1 if v["which"] == "heaviest" else 1
        return f"{v['which']}: {min(items, key=lambda x: (sign * x[1]['w'], x[0]))[0]}"
    if op == "newest":
        items = _active(state, lambda o: o["kind"] == a["kind"])
        if not items:
            return f"{v['which']}: (none)"
        sign = -1 if v["which"] == "newest" else 1
        return f"{v['which']}: {min(items, key=lambda x: sign * x[1]['made'])[0]}"
    if op == "find_tag":
        return f"tag {a['tag']}: {_ids([i for i, _ in _active(state, lambda o: o['tag'] == a['tag'])])}"
    if op == "checksum":
        items = _active(state, lambda o: o["place"] == a["place"] and (v["with_locked"] or not o["locked"]))
        return f"checksum: {sum(i for i, _ in items) * v['mul'] % v['mod']}"
    if op == "archived":
        return f"archive: {_ids(sorted(int(i) for i, o in state['objs'].items() if o['state'] == 'archived'))}"
    if op == "census":
        return ", ".join(f"{p} {len(_active(state, lambda o: o['place'] == p))}" for p in s["places"])
    if op == "convert":
        if v["dir"] == "d2b":
            return f"{a['n']} {s['unit']} = {a['n'] * s['C']} {s['base_unit']}"
        return f"{a['n']} {s['base_unit']} = {a['n'] // s['C']} {s['unit']}"
    if op == "move":
        o = _unlocked(_obj(state, a["id"]))
        o["place"] = a["place"]
        return f"moved #{a['id']} to {a['place']}" if verb else "ok"
    if op == "archive":
        o = _unlocked(_obj(state, a["id"]))
        o["state"] = "archived"
        return f"archived #{a['id']}" if verb else "ok"
    if op == "delete":
        _unlocked(_obj(state, a["id"], active=False))
        del state["objs"][str(a["id"])]
        return f"deleted #{a['id']}" if verb else "ok"
    if op == "restore":
        o = _obj(state, a["id"], active=False)
        if o["state"] != "archived":
            raise SimError("badarg")
        o["state"] = "active"
        return f"restored #{a['id']}" if verb else "ok"
    if op == "lock":
        o = _obj(state, a["id"])
        o["locked"] = (not o["locked"]) if v["toggle"] else True
        return (f"{'locked' if o['locked'] else 'unlocked'} #{a['id']}") if verb else "ok"
    if op == "unlock":
        o = _obj(state, a["id"])
        o["locked"] = False
        return f"unlocked #{a['id']}" if verb else "ok"
    if op == "retag":
        o = _unlocked(_obj(state, a["id"]))
        o["tag"] = a["tag"]
        return f"tagged #{a['id']} {a['tag']}" if verb else "ok"
    if op == "archive_kind":
        t = a["threshold"] if v["unit"] == "display" else a["threshold"] / s["C"]
        hit = _active(state, lambda o: o["kind"] == a["kind"] and not o["locked"] and (o["w"] > t if v["strict"] else o["w"] >= t))
        for _, o in hit:
            o["state"] = "archived"
        return f"archived {len(hit)} items" if verb else f"ok {len(hit)}"
    if op == "move_kind":
        hit = _active(state, lambda o: o["kind"] == a["kind"] and o["place"] == a["src"] and not o["locked"])
        for _, o in hit:
            o["place"] = a["dst"]
        return f"moved {len(hit)} items" if verb else f"ok {len(hit)}"
    if op == "lock_place":
        hit = _active(state, lambda o: o["place"] == a["place"] and not o["locked"])
        for _, o in hit:
            o["locked"] = True
        return f"locked {len(hit)} items" if verb else f"ok {len(hit)}"
    if op == "clone":
        o = _obj(state, a["id"])
        state["clock"] += 1
        new = state["next_id"]
        state["next_id"] += 1
        state["objs"][str(new)] = {**o, "made": state["clock"], "locked": False}
        return f"created #{new}" if verb else "ok"
    if op == "swap":
        x, y = (_unlocked(_obj(state, a[k])) for k in ("a", "b"))
        if a["a"] == a["b"]:
            raise SimError("badarg")
        x["place"], y["place"] = y["place"], x["place"]
        return f"swapped #{a['a']} and #{a['b']}" if verb else "ok"
    if op == "purge_archive":
        gone = sorted(i for i, o in state["objs"].items() if o["state"] == "archived" and not o["locked"])
        for i in gone:
            del state["objs"][i]
        return f"deleted {len(gone)} items" if verb else f"ok {len(gone)}"
    raise AssertionError(op)


def call(spec: dict, state: dict, name: str, args: list) -> tuple[str, bool]:
    """One tool call -> (observation, ok). Errors are observations too ("error <code>")."""
    if name == "answer":
        if len(args) != 1:
            return "error: answer takes one argument", False
        return str(args[0]), True
    tool = tool_map(spec).get(name)
    if tool is None:
        return f"error: no such tool {name}", False
    canon = OPS[tool["op"]][0]
    try:
        if len(args) != len(canon):
            raise SimError("badarg")
        a = {}
        for pos, ci in enumerate(tool["perm"]):          # user position pos carries canonical argument ci
            (an, at), val = canon[ci], args[pos]
            if at in ("id", "num"):
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    raise SimError("badarg")
            elif not isinstance(val, str):
                raise SimError("badarg")
            elif at == "place" and val not in spec["places"] or at == "kind" and val not in spec["kinds"]:
                raise SimError("badarg")
            elif at == "tag" and not re.fullmatch(r"[a-z]{1,12}", val):
                raise SimError("badarg")
            a[an] = val
        return apply(spec, state, tool, a), True
    except SimError as e:
        return f"error {spec['errors'][e.kind]}", False


def parse_program(text: str) -> list[tuple[str, list]]:
    """Model output -> [(name, args)]. Code fences and blank lines are ignored; anything else must be a call."""
    lines = [ln.strip() for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("```")]
    if not lines:
        raise ProgramError("syntax error: empty program")
    if len(lines) > MAX_CALLS:
        raise ProgramError(f"syntax error: more than {MAX_CALLS} lines")
    out = []
    for n, ln in enumerate(lines, 1):
        try:
            node = ast.parse(ln, mode="eval").body
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords):
                raise ValueError
            args = []
            for x in node.args:
                if isinstance(x, ast.Constant) and type(x.value) in (int, str):
                    args.append(x.value)
                elif isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub) and isinstance(x.operand, ast.Constant) and type(x.operand.value) is int:
                    args.append(-x.operand.value)
                else:
                    raise ValueError
            out.append((node.func.id, args))
        except (SyntaxError, ValueError):
            raise ProgramError(f"syntax error on line {n}") from None
    return out


def run_program(spec: dict, state: dict, text: str) -> dict:
    """Execute a program on a COPY of `state`. Returns {state, outputs, values, error: None | {line, obs}}.
    values[i] is the typed answer value of line i (see value_of), None for lines that carry no answer."""
    st = copy.deepcopy(state)
    try:
        prog = parse_program(text)
    except ProgramError as e:
        return {"state": st, "outputs": [], "values": [], "error": {"line": 0, "obs": str(e)}}
    outs, vals = [], []
    for n, (name, args) in enumerate(prog, 1):
        obs, ok = call(spec, st, name, args)
        outs.append(obs)
        vals.append(value_of(spec, name, args, obs) if ok else None)
        if not ok:
            return {"state": st, "outputs": outs, "values": vals, "error": {"line": n, "obs": obs}}
    return {"state": st, "outputs": outs, "values": vals, "error": None}


_ID_LIST = r"(\(none\)|\d+(?:, \d+)*)"
_VALUE_RE = {  # op -> exact grammar of its (simulator-printed) output; group 1 is the answer value
    "list_place": r"[a-z]+: " + _ID_LIST, "list_kind": r"[A-Z]+: " + _ID_LIST, "find_tag": r"tag [a-z]+: " + _ID_LIST,
    "archived": r"archive: " + _ID_LIST, "count": r"count: (\d+)", "weigh": r"total: (\d+) [a-z]+",
    "extreme": r"(?:heaviest|lightest): (\(none\)|\d+)", "newest": r"(?:newest|oldest): (\(none\)|\d+)",
    "checksum": r"checksum: (\d+)", "convert": r"\d+ [a-z]+ = (\d+) [a-z]+",
}


def _typed(s: str):
    if s == "(none)":
        return []
    xs = [int(x) for x in s.split(", ")]
    return xs[0] if len(xs) == 1 else xs


def value_of(spec: dict, name: str, args: list, obs: str):
    """Typed answer value of one successful call: an int, a list of ints, or None (no answer).
    answer(x) has an exact grammar: an int literal, or a string that is exactly an integer ("42", "-3") or a
    comma-separated list of integers ("101, 202"); anything else ("42e999", "not 42", "42 or 43") is None.
    Tool outputs are parsed with the exact grammar of their op; mutations, inspect and census carry no answer."""
    if name == "answer":
        (x,) = args
        if isinstance(x, int):
            return x
        m = re.fullmatch(r"\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*", x)
        if not m:
            return None
        xs = [int(v) for v in re.split(r"\s*,\s*", m.group(1))]
        return xs[0] if len(xs) == 1 else xs
    rx = _VALUE_RE.get(tool_map(spec)[name]["op"])
    m = re.fullmatch(rx, obs) if rx else None
    return _typed(m.group(1)) if m else None


def score(task: dict, res: dict) -> float:
    """1.0 iff the program ran without error, the final state equals the gold state (state tasks) and the typed value
    of the last line equals the gold value (answer tasks; a set answer may come in any order, without duplicates).
    Collateral state changes make a state task fail."""
    if res["error"] is not None:
        return 0.0
    if task["check_state"] and res["state"] != task["gold_state"]:
        return 0.0
    if task["answer"] is not None:
        if not res["values"]:
            return 0.0
        got = res["values"][-1]
        kind, val = task["answer"]["kind"], task["answer"]["value"]
        if kind == "int" and not (type(got) is int and got == val):
            return 0.0
        if kind == "set":
            got = [got] if type(got) is int else got
            if not isinstance(got, list) or sorted(got) != sorted(val) or len(set(got)) != len(got):
                return 0.0
    return 1.0


# ------------------------------------------------------------------ explorer (frozen scripted policy)
def explore(spec: dict, n_calls: int, random_frac: float = 0.3, followup: float = 0.6) -> tuple[list[dict], dict]:
    """A scripted explorer: coverage pass over every tool, one error pass, then a mix of sensible calls (70%) and
    random probes (30%), with an inspect follow-up after many mutations. Returns (events, end state).
    Events are exactly {"call", "obs"} pairs: what an agent at the keyboard would see."""
    rng = random.Random(f"explore-{spec['seed']}")
    st = copy.deepcopy(spec["init"])
    events: list[dict] = []
    tools = spec["tools"]
    by_op = {}
    for t in tools:
        by_op.setdefault(t["op"], []).append(t)

    def do(tool, canon_args):
        args = list(canon_args)                       # wrong-arity probes are passed through as they are
        if len(canon_args) == len(tool["perm"]):
            args = [canon_args[ci] for ci in tool["perm"]]
        obs, _ = call(spec, st, tool["name"], args)
        events.append({"call": fmt_call(tool["name"], args), "obs": obs})
        return obs

    def pick(pred):
        xs = [int(i) for i, o in st["objs"].items() if pred(o)]
        return rng.choice(sorted(xs)) if xs else rng.randint(100, 999)

    def sensible(t):
        op, S = t["op"], spec
        act = lambda o: o["state"] == "active"  # noqa: E731
        free = lambda o: o["state"] == "active" and not o["locked"]  # noqa: E731
        vals = {"place": rng.choice(S["places"]), "kind": rng.choice(S["kinds"]), "tag": rng.choice(S["tags"]),
                "num": rng.randint(1, 9), "src": rng.choice(S["places"]), "dst": rng.choice(S["places"])}
        target = {"restore": lambda o: o["state"] == "archived", "unlock": lambda o: act(o) and o["locked"],
                  "inspect": lambda o: True, "lock": act, "clone": act, "delete": free}.get(op, free)
        out = []
        for an, at in OPS[op][0]:
            if at == "id":
                out.append(pick(target))
            elif an == "threshold":
                out.append(rng.randint(2, 8) * (S["C"] if t["var"].get("unit") == "base" else 1))
            elif an == "n":
                out.append(rng.randint(1, 12))
            else:
                out.append(vals[an if an in vals else at])
        return out

    def maybe_followup(t, a):
        if OPS[t["op"]][1] and rng.random() < followup:
            ids_ = [x for (an, at), x in zip(OPS[t["op"]][0], a) if at == "id"]
            if ids_:
                do(by_op["inspect"][0], [ids_[0]])
            elif by_op.get("list_place"):
                do(by_op["list_place"][0], [a[[an for an, _ in OPS[t["op"]][0]].index("dst")] if t["op"] == "move_kind"
                                            else rng.choice(spec["places"])])

    # 1) opening: overview calls
    do(by_op["census"][0], [])
    for t in by_op["list_place"]:
        for p in spec["places"][:2]:
            do(t, [p])
    do(by_op["archived"][0], [])
    # 2) coverage: every tool once with sensible args (destructive ops last)
    order = sorted(tools, key=lambda t: (t["op"] == "delete", rng.random()))
    for t in order:                               # the archive is emptied once, during this pass
        a = sensible(t)
        do(t, a)
        maybe_followup(t, a)
    # 3) errors: locked item, missing id, bad argument, each at least twice
    mut = [t for t in tools if t["op"] in ("move", "archive", "retag", "swap")]
    for _ in range(2):
        t = rng.choice(mut)
        a = sensible(t)
        for k, (an, at) in enumerate(OPS[t["op"]][0]):
            if at == "id":
                a[k] = pick(lambda o: o["state"] == "active" and o["locked"])
        do(t, a)
        do(by_op["inspect"][0], [rng.choice([x for x in range(100, 1000) if str(x) not in st["objs"]])])
        t = rng.choice(tools)
        do(t, [0 if isinstance(x, str) else "x" for x in sensible(t)] or ["x"])
    # 4) mixed exploration until the budget; the archive is emptied at most once, near the end
    cycle = []
    while len(events) < n_calls:
        if rng.random() < random_frac:
            t = rng.choice([x for x in tools if x["op"] != "purge_archive"])
            r = rng.random() * (0.4 if t["op"] == "delete" else 1.0)     # deletes only probe random ids
            if r < 0.4:
                a = [rng.randint(100, 999) if at in ("id",) else x for (an, at), x in zip(OPS[t["op"]][0], sensible(t))]
            elif r < 0.7:
                a = sensible(t)[::-1]                       # canonical order reversed: often a type error
            else:
                a = sensible(t) + [rng.randint(1, 9)]       # wrong arity
            do(t, a)
            continue
        if not cycle:
            cycle = [t for t in tools if t["op"] != "purge_archive"]
            rng.shuffle(cycle)
        t = cycle.pop()
        if t["op"] == "delete" and rng.random() < 0.7 or t["op"] == "archive_kind" and rng.random() < 0.5:
            continue                                  # keep the world populated
        a = sensible(t)
        do(t, a)
        maybe_followup(t, a)
    return events[:n_calls], replay(spec, events[:n_calls])


def replay(spec: dict, events: list[dict]) -> dict:
    """Re-execute the transcript's calls from the initial state; returns the end state."""
    st = copy.deepcopy(spec["init"])
    for e in events:
        prog = parse_program(e["call"])
        (name, args), = prog
        call(spec, st, name, args)
    return st


# ------------------------------------------------------------------ held-out tasks
def _tool(spec, op) -> dict:
    return next(t for t in spec["tools"] if t["op"] == op)


def _c(spec, op, *canon) -> str:
    t = _tool(spec, op)
    args = [None] * len(canon)
    for pos, ci in enumerate(t["perm"]):
        args[pos] = canon[ci]
    return fmt_call(t["name"], args)


TEMPLATES = {  # name -> weight (DIFFICULTY mix; tuned in the GPU pilot so that none <= 15% and ICL - none >= 15 pts)
    "archive": 3, "delete": 2, "move": 3, "lock": 2, "unlock": 2, "restore": 2, "retag": 2, "swap": 1, "clone": 1,
    "archive_kind": 2, "move_kind": 2, "lock_place": 1, "count": 1, "weigh": 2, "extreme": 2, "newest": 2,
    "checksum": 2, "find_tag": 1, "unlock_delete": 2, "move_lock": 2, "restore_move": 2, "archive_checksum": 2,
}


def make_tasks(spec: dict, end: dict, events: list[dict], n: int, salt: str) -> list[dict]:
    """n held-out goals starting from the post-exploration state `end`. Gold = the reference program's outcome on the
    simulator. Rejected: duplicate goals; mutation tasks whose reference lines ALL occur verbatim in the transcript
    (query tasks have a tiny argument space, so their calls usually do occur; their answers are computed at `end`);
    reference programs that error; state tasks whose gold state equals the start state."""
    rng = random.Random(f"tasks-{spec['seed']}-{salt}")
    S, U = spec, spec["unit"]
    seen_calls = {e["call"] for e in events}
    objs = end["objs"]

    def ids(pred):
        return sorted(int(i) for i, o in objs.items() if pred(o))

    act = lambda o: o["state"] == "active"  # noqa: E731
    free = lambda o: act(o) and not o["locked"]  # noqa: E731
    out, goals = [], set()
    names, weights = zip(*TEMPLATES.items())
    for _ in range(50 * n):
        if len(out) >= n:
            break
        tpl = rng.choices(names, weights)[0]
        P, K = rng.choice(S["places"]), rng.choice(S["kinds"])
        try:
            g = _task(tpl, S, U, rng, ids, act, free, P, K, objs)
        except IndexError:        # no suitable object for this template in this state
            continue
        if g is None:
            continue
        goal, prog, answer = g
        if goal in goals or answer is None and all(line in seen_calls for line in prog):
            continue
        res = run_program(spec, end, "\n".join(prog))
        if res["error"] is not None:
            continue
        check_state = any(OPS[_first_op(S, line)][1] for line in prog)
        if check_state and res["state"] == end:
            continue
        if answer is None and not check_state:
            continue
        task = {"goal": goal, "template": tpl, "reference": prog, "n_calls": len(prog), "check_state": bool(check_state),
                "gold_state": res["state"] if check_state else None, "answer": answer}
        if score(task, res) != 1.0:
            continue
        goals.add(goal)
        out.append(task)
    if len(out) < n:
        raise RuntimeError(f"world {spec['seed']}: only {len(out)} tasks")
    return out


def _first_op(spec, line: str) -> str:
    return tool_map(spec)[line.split("(")[0]]["op"]


def _task(tpl, S, U, rng, ids, act, free, P, K, objs):
    c = lambda op, *a: _c(S, op, *a)  # noqa: E731
    pick = lambda pred: rng.choice(ids(pred))  # noqa: E731
    var = lambda op: _tool(S, op)["var"]  # noqa: E731
    if tpl == "archive":
        i = pick(free)
        return f"Archive item {i}.", [c("archive", i)], None
    if tpl == "delete":
        i = pick(lambda o: not o["locked"])
        return f"Permanently delete item {i}.", [c("delete", i)], None
    if tpl == "move":
        i = pick(lambda o: free(o) and o["place"] != P)
        return f"Move item {i} to {P}.", [c("move", i, P)], None
    if tpl == "lock":
        i = pick(lambda o: act(o) and not o["locked"])
        return f"Lock item {i}.", [c("lock", i)], None
    if tpl == "unlock":
        i = pick(lambda o: act(o) and o["locked"])
        return f"Unlock item {i}.", [c("unlock", i)], None
    if tpl == "restore":
        i = pick(lambda o: o["state"] == "archived")
        return f"Restore item {i} from the archive.", [c("restore", i)], None
    if tpl == "retag":
        i = pick(lambda o: free(o))
        t = rng.choice([x for x in S["tags"] if x != objs[str(i)]["tag"]])
        return f'Tag item {i} with "{t}".', [c("retag", i, t)], None
    if tpl == "swap":
        a = pick(free)
        b = pick(lambda o: free(o) and o["place"] != objs[str(a)]["place"])
        return f"Swap the places of items {a} and {b}.", [c("swap", a, b)], None
    if tpl == "clone":
        i = pick(act)
        return f"Make a copy of item {i}.", [c("clone", i)], None
    if tpl == "archive_kind":
        k = rng.randint(2, 8)
        v = var("archive_kind")
        arg = k * S["C"] if v["unit"] == "base" else k
        cmp = "more than" if v["strict"] else "at least"
        return f"Archive every unlocked {K} item that weighs {cmp} {k} {U}.", [c("archive_kind", K, arg)], None
    if tpl == "move_kind":
        Q = rng.choice([p for p in S["places"] if p != P])
        return f"Move every unlocked {K} item from {P} to {Q}.", [c("move_kind", K, P, Q)], None
    if tpl == "lock_place":
        return f"Lock every item in {P}.", [c("lock_place", P)], None
    if tpl == "count":
        v = var("count")
        what = f"unlocked {K} items" if v["skip_locked"] else f"{K} items (locked or not)"
        n = len(ids(lambda o: act(o) and o["kind"] == K and o["place"] == P and not (v["skip_locked"] and o["locked"])))
        return f"How many {what} are in {P}?", [c("count", K, P)], {"kind": "int", "value": n}
    if tpl == "weigh":
        v = var("weigh")
        tot = sum(objs[str(i)]["w"] for i in ids(lambda o: act(o) and o["kind"] == K))
        unit, val = (U, tot) if v["unit"] == "display" else (S["base_unit"], tot * S["C"])
        return f"What is the total weight of all {K} items, in {unit}?", [c("weigh", K)], {"kind": "int", "value": val}
    if tpl == "extreme":
        v = var("extreme")
        items = [(int(i), o) for i, o in objs.items() if act(o) and o["place"] == P]
        if not items:
            return None
        sign = -1 if v["which"] == "heaviest" else 1
        best = min(items, key=lambda x: (sign * x[1]["w"], x[0]))[0]
        return (f"Which item in {P} is the {v['which']}? (ties: lowest id)", [c("extreme", P)], {"kind": "int", "value": best})
    if tpl == "newest":
        v = var("newest")
        items = [(int(i), o) for i, o in objs.items() if act(o) and o["kind"] == K]
        if not items:
            return None
        sign = -1 if v["which"] == "newest" else 1
        best = min(items, key=lambda x: sign * x[1]["made"])[0]
        word = "most recently" if v["which"] == "newest" else "least recently"
        return f"Which {K} item was created {word}?", [c("newest", K)], {"kind": "int", "value": best}
    if tpl == "checksum":
        return f"What is the tally checksum of {P}?", [c("checksum", P)], {"kind": "int", "value": _checksum(S, objs, P)}
    if tpl == "find_tag":
        t = rng.choice(S["tags"])
        got = ids(lambda o: act(o) and o["tag"] == t)
        if not got:
            return None
        return f'Which items carry the tag "{t}"?', [c("find_tag", t)], {"kind": "set", "value": got}
    if tpl == "unlock_delete":
        i = pick(lambda o: act(o) and o["locked"])
        return f"Unlock item {i}, then permanently delete it.", [c("unlock", i), c("delete", i)], None
    if tpl == "move_lock":
        i = pick(lambda o: free(o) and o["place"] != P)
        return f"Move item {i} to {P}, then lock it.", [c("move", i, P), c("lock", i)], None
    if tpl == "restore_move":
        i = pick(lambda o: o["state"] == "archived" and not o["locked"] and o["place"] != P)
        return f"Restore item {i} from the archive, then move it to {P}.", [c("restore", i), c("move", i, P)], None
    if tpl == "archive_checksum":
        i = pick(lambda o: free(o) and o["place"] == P)
        o2 = copy.deepcopy(objs)
        o2[str(i)]["state"] = "archived"
        return (f"Archive item {i}, then report the tally checksum of {P}.", [c("archive", i), c("checksum", P)],
                {"kind": "int", "value": _checksum(S, o2, P)})
    raise AssertionError(tpl)


def _checksum(S, objs, P) -> int:
    v = _tool(S, "checksum")["var"]
    return sum(int(i) for i, o in objs.items() if o["state"] == "active" and o["place"] == P
               and (v["with_locked"] or not o["locked"])) * v["mul"] % v["mod"]
