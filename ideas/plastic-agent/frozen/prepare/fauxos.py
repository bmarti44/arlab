"""FauxOS: a procedurally generated fictional tool world (deterministic, pure Python, no LLM inside).

This file is PREPARE/EVALUATE-only. It lives in frozen/prepare/ (mounted only into PREPARE) and PREPARE copies it into
each split's private/ dir for the evaluator. RUN never sees it: the surface gets only the exploration transcript
(calls + observations) and the tool names, so the semantics below can only be learned from what the transcript shows.

A world ("instance") is a spec generated from one integer seed:
  - 4 kinds, 4 places, 6 tags, a weight unit (display unit U, base unit B, 1 U = C B, C in 2..9), 3 error codes
    (locked / missing / bad-argument), 30-60 objects with hidden fields;
  - `inspect` plus N_OPS operations sampled from TASK_OPS, one tool each, with a pseudo-word name, a random positional
    argument order and per-op semantic variants; tool verbs are pseudo-words, truthful English verbs, or misleading
    English verbs ("purge" that archives) -- the "drive on the left". Every tool reports what it did (VERBOSE_P), so
    the effect of each call is visible in the transcript.
Programs (forgiving format): ONE call, the first line of the model output that starts like a call, `name(arg, ...)`
with positional int / "string" literals (a bare word counts as a string, a digit string as an int where an int is
expected); a leading "> " is dropped and every other line (prose, code fences, predicted outputs, further calls) is
ignored; a first call line that does not parse is a syntax error. Its output is the answer; the builtin answer(x)
returns x.
"""
from __future__ import annotations

import ast
import copy
import os
import sys
import json
import keyword
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wording  # noqa: E402  (the v2 wording families; copied next to this file wherever it is used)

CONS, VOWS = "bdfgklmnprstvz", "aeiou"

# op -> (canonical args [(name, type)], mutating)
OPS = {
    "inspect": ([("id", "id")], False),
    "count": ([("kind", "kind"), ("place", "place")], False),
    "weigh": ([("kind", "kind")], False),
    "extreme": ([("place", "place")], False),
    "newest": ([("kind", "kind")], False),
    "find_tag": ([("tag", "tag")], False),
    "checksum": ([("place", "place")], False),
    "move": ([("id", "id"), ("place", "place")], True),
    "archive": ([("id", "id")], True),
    "delete": ([("id", "id")], True),
    "restore": ([("id", "id")], True),
    "lock": ([("id", "id")], True),
    "unlock": ([("id", "id")], True),
    "retag": ([("id", "id"), ("tag", "tag")], True),
    "clone": ([("id", "id")], True),
    "swap": ([("a", "id"), ("b", "id")], True),
}
TASK_OPS = tuple(op for op in OPS if op != "inspect")      # every held-out task is ONE call of one of these
# v2 robustness split ("novel" style, 2026-10-01): operation types that never occur in the standard worlds, plus
# reworded observations and task goals for EVERY op. Standard worlds (style "std") are byte-identical to v1.
NOVEL_OPS = {
    "bump": ([("id", "id")], True),               # weight + 1
    "pin": ([("id", "id")], True),                # sets the object's pin flag (a field only novel worlds have)
    "count_tag": ([("tag", "tag")], False),       # number of active items carrying a tag
    "oldest_at": ([("place", "place")], False),   # oldest / newest active item at a place (variant)
}
OPS.update(NOVEL_OPS)
# v2 campaign worlds (style "fam", 2026-10-01, V2-DESIGN-sol.md): per-world wording families (wording.py; validation
# bank A, holdout bank B), tool names independent of semantics, 6 standard ops + 4 split-specific ops, balanced tasks.
DEV_OPS = {                                       # validation-only op types (feedback on handling unseen op types)
    "shrink": ([("id", "id")], True),             # weight - 1 (badarg at weight 1)
    "unpin": ([("id", "id")], True),
    "count_place": ([("place", "place")], False), # number of active items at a place
    "lightest_kind": ([("kind", "kind")], False), # lightest active item of a kind (ties: lowest id)
}
OPS.update(DEV_OPS)
RESERVED_OPS = tuple(NOVEL_OPS)                   # holdout-only op types in fam worlds (never in validation)
FAM_STD, FAM_TASKS_PER_OP = 6, 6
FAM_VOCAB, FAM_OBJ, FAM_MIN = 7, (60, 90), 18     # kinds/places/tags; objects; min active-locked and archived objects
N_NOVEL_STD = 6                   # novel worlds: all 4 NOVEL_OPS + 6 standard task ops (+ inspect)
N_OPS = 10                        # task ops per world (+ inspect): 11 tools. CALIBRATE with the gate
# English verbs: truthful ones a pretrained model would guess right, misleading ones it would guess wrong.
TRUE_VERBS = {"bump": ["bump", "grow"], "pin": ["pin", "star"], "count_tag": ["tagcount", "census"],
              "oldest_at": ["eldest", "vintage"], "inspect": ["inspect", "info"], "count": ["count", "tally"], "weigh": ["weigh", "mass"],
              "extreme": ["top", "pick"], "newest": ["latest", "recent"], "find_tag": ["find", "search"],
              "checksum": ["checksum", "digest"], "move": ["move", "send"], "archive": ["archive", "shelve"],
              "delete": ["delete", "remove"], "restore": ["restore", "revive"], "lock": ["lock", "seal"],
              "unlock": ["unlock", "open"], "retag": ["tag", "label"], "clone": ["clone", "copy"], "swap": ["swap", "switch"]}
FALSE_VERBS = {"bump": ["shrink", "hide"], "pin": ["drop", "unpin"], "count_tag": ["purge", "rename"],
               "oldest_at": ["heaviest", "newest"], "inspect": ["erase", "move"], "count": ["list", "weigh"], "weigh": ["count", "lock"],
               "extreme": ["oldest", "random"], "newest": ["heaviest", "first"], "find_tag": ["delete", "clone"],
               "checksum": ["count", "undo"], "move": ["copy", "melt"], "archive": ["purge", "burn"],
               "delete": ["stash", "keep"], "restore": ["drop", "bury"], "lock": ["open", "free"],
               "unlock": ["seal", "bolt"], "retag": ["wipe", "move"], "clone": ["remove", "merge"], "swap": ["join", "split"]}
VERB_MIX = (0.15, 0.35)          # P(truthful English verb), P(misleading English verb); rest: pseudo-word verb
VERBOSE_P = 1.0                  # P(a mutating tool reports its effect); otherwise it prints only "ok"
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
        if w not in used and len(w) >= 3 and not keyword.iskeyword(w):     # bare-word arguments must parse
            used.add(w)
            return w
    raise RuntimeError("name space exhausted")


def gen_world(seed: int, style: str = "std", bank: str = "A", extra: tuple = ()) -> dict:
    """Spec (private) of one world. Deterministic in `seed` (and `style`: "std" = v1 worlds, byte-identical; "novel" =
    the v2 robustness split: every NOVEL_OPS op + N_NOVEL_STD standard ops, reworded outputs and goals, pin flags)."""
    if style == "fam":
        return _gen_fam(seed, bank, extra)
    rng = random.Random(f"fauxos-{seed}" if style == "std" else f"fauxos-{style}-{seed}")
    used: set[str] = set()
    kinds = [_fresh(rng, used, lambda: pseudo(rng, 1).upper()) for _ in range(4)]
    places = [_fresh(rng, used, lambda: pseudo(rng, 2)) for _ in range(4)]
    tags = [_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(6)]
    unit, base_unit = (_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(2))
    C = rng.randint(2, 9)
    letter = rng.choice("ABDEFGHJKMNPQRTVWXYZ")
    codes = rng.sample(range(10, 100), 3)
    errors = {k: f"{letter}{c}" for k, c in zip(("locked", "missing", "badarg"), codes)}

    if style == "std":
        ops = ["inspect"] + sorted(rng.sample(TASK_OPS, N_OPS), key=list(OPS).index)
    else:
        ops = ["inspect"] + sorted(rng.sample(TASK_OPS, N_NOVEL_STD) + list(NOVEL_OPS), key=list(OPS).index)
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
    n = rng.randint(*N_OBJ)
    ids = rng.sample(range(100, 1000), n)
    made = list(range(1, n + 1))
    rng.shuffle(made)
    objs = {}
    for i, m in zip(ids, made):
        objs[str(i)] = {"kind": rng.choice(kinds), "place": rng.choice(places), "w": rng.randint(1, 9),
                        "locked": rng.random() < 0.2, "state": "archived" if rng.random() < 0.15 else "active",
                        "made": m, "tag": rng.choice(tags) if rng.random() < 0.5 else "-"}
    for i in sorted(objs, key=int):          # at least 3 active locked objects, so the explorer can show the locked error
        if sum(o["state"] == "active" and o["locked"] for o in objs.values()) >= 3:
            break
        if objs[i]["state"] == "active":
            objs[i]["locked"] = True
    if style != "std":
        for o in objs.values():
            o["pin"] = rng.random() < 0.2
    init = {"objs": objs, "clock": n, "next_id": 1000}
    return {"seed": seed, **({"style": style} if style != "std" else {}), "kinds": kinds, "places": places, "tags": tags, "unit": unit, "base_unit": base_unit, "C": C,
            "errors": errors, "tools": tools, "init": init}


def gen_twin(spec: dict, seed: int) -> dict:
    """The paired CONTROL world ("twin") of target `spec`, deterministic in `seed`: IDENTICAL tool names, operation set
    (so the same task templates), kinds, places, tags and units (the argument vocabulary), but independently drawn
    semantics: a derangement of which name does what (every name gets a different operation), fresh argument orders
    and per-op variants, fresh objects, error codes and unit conversion (from gen_world(seed), renamed into the target's
    vocabulary). A surface that learns only name / vocabulary priors adapts the same way to both; only what it learns
    from the transcript differs."""
    style = spec.get("style", "std")
    if style == "fam":
        return _gen_fam_twin(spec, seed)
    base = gen_world(seed, style)
    rng = random.Random(f"fauxos-twin-{seed}")
    ren = {"-": "-"}
    for k in ("kinds", "places", "tags"):
        ren.update(zip(base[k], spec[k]))
    objs = {i: {**o, "kind": ren[o["kind"]], "place": ren[o["place"]], "tag": ren[o["tag"]]} for i, o in base["init"]["objs"].items()}
    ops, names = [t["op"] for t in spec["tools"]], [t["name"] for t in spec["tools"]]
    while True:
        sigma = rng.sample(range(len(ops)), len(ops))
        if all(sigma[k] != k for k in range(len(ops))):
            break
    tools = []
    for k, name in enumerate(names):
        op = ops[sigma[k]]
        perm = list(range(len(OPS[op][0])))
        rng.shuffle(perm)
        tools.append({"name": name, "op": op, "perm": perm, "var": _variant(rng, op), "verbose": rng.random() < VERBOSE_P})
    tools.sort(key=lambda t: list(OPS).index(t["op"]))
    return {"seed": seed, "twin_of": spec["seed"], **({"style": style} if style != "std" else {}), **{k: copy.deepcopy(spec[k]) for k in ("kinds", "places", "tags", "unit", "base_unit")},
            "C": base["C"], "errors": base["errors"], "tools": tools, "init": {**base["init"], "objs": objs}}


def _variant(rng, op) -> dict:
    v = {}
    if op == "count":
        v = {"skip_locked": rng.random() < 0.5}
    elif op == "weigh":
        v = {"unit": rng.choice(["display", "base"])}
    elif op == "extreme":
        v = {"which": rng.choice(["heaviest", "lightest"])}
    elif op == "newest":
        v = {"which": rng.choice(["newest", "oldest"])}
    elif op == "checksum":
        v = {"mul": rng.choice([3, 7, 11]), "mod": rng.choice([89, 97]), "with_locked": rng.random() < 0.5}
    elif op == "oldest_at":
        v = {"which": rng.choice(["newest", "oldest"])}
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


def apply(spec: dict, state: dict, tool: dict, a: dict) -> str:
    """Execute one op on `state` in place; returns the observation. Raises SimError."""
    if spec.get("style") == "novel":
        return _apply_novel(spec, state, tool, a)
    if spec.get("style") == "fam":
        return _apply_fam(spec, state, tool, a)[0]
    op, v, s = tool["op"], tool["var"], spec
    verb = tool["verbose"]
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
        o["locked"] = True
        return f"locked #{a['id']}" if verb else "ok"
    if op == "unlock":
        o = _obj(state, a["id"])
        o["locked"] = False
        return f"unlocked #{a['id']}" if verb else "ok"
    if op == "retag":
        o = _unlocked(_obj(state, a["id"]))
        o["tag"] = a["tag"]
        return f"tagged #{a['id']} {a['tag']}" if verb else "ok"
    if op == "clone":
        o = _obj(state, a["id"])
        state["clock"] += 1
        new = state["next_id"]
        state["next_id"] += 1
        state["objs"][str(new)] = {**o, "made": state["clock"], "locked": False}
        return f"copied #{a['id']} as #{new}" if verb else "ok"
    if op == "swap":
        x, y = (_unlocked(_obj(state, a[k])) for k in ("a", "b"))
        if a["a"] == a["b"]:
            raise SimError("badarg")
        x["place"], y["place"] = y["place"], x["place"]
        return f"swapped #{a['a']} and #{a['b']}" if verb else "ok"
    raise AssertionError(op)


def _apply_novel(spec: dict, state: dict, tool: dict, a: dict) -> str:
    """The same semantics as apply() for standard ops (plus NOVEL_OPS), with different observation wording."""
    op, v, s = tool["op"], tool["var"], spec
    verb = tool["verbose"]
    if op == "inspect":
        o = _obj(state, a["id"], active=False)
        return (f"item {a['id']} | {o['kind']} @ {o['place']} | {o['w']}{s['unit']} | born {o['made']} | "
                f"{'sealed' if o['locked'] else 'open'} | {o['state']} | label {o['tag']} | pinned {'Y' if o['pin'] else 'N'}")
    if op == "count":
        return f"n = {len(_active(state, lambda o: o['kind'] == a['kind'] and o['place'] == a['place'] and not (v['skip_locked'] and o['locked'])))}"
    if op == "weigh":
        tot = sum(o["w"] for _, o in _active(state, lambda o: o["kind"] == a["kind"]))
        return f"sum = {tot}{s['unit']}" if v["unit"] == "display" else f"sum = {tot * s['C']}{s['base_unit']}"
    if op in ("extreme", "newest", "oldest_at"):
        if op == "extreme":
            items, key = _active(state, lambda o: o["place"] == a["place"]), lambda x: (-x[1]["w"] if v["which"] == "heaviest" else x[1]["w"], x[0])
        else:
            pred = (lambda o: o["kind"] == a["kind"]) if op == "newest" else (lambda o: o["place"] == a["place"])
            items, key = _active(state, pred), lambda x: -x[1]["made"] if v["which"] == "newest" else x[1]["made"]
        return f"=> {min(items, key=key)[0] if items else 'nothing'} ({v['which']})"
    if op == "find_tag":
        got = [i for i, _ in _active(state, lambda o: o["tag"] == a["tag"])]
        return f"[{' '.join(str(i) for i in got)}]"
    if op == "count_tag":
        return f"n = {len(_active(state, lambda o: o['tag'] == a['tag']))}"
    if op == "checksum":
        items = _active(state, lambda o: o["place"] == a["place"] and (v["with_locked"] or not o["locked"]))
        return f"digest = {sum(i for i, _ in items) * v['mul'] % v['mod']}"
    if op == "move":
        o = _unlocked(_obj(state, a["id"]))
        o["place"] = a["place"]
        return f"item {a['id']} now sits at {a['place']}" if verb else "done"
    if op == "archive":
        o = _unlocked(_obj(state, a["id"]))
        o["state"] = "archived"
        return f"item {a['id']} put away" if verb else "done"
    if op == "delete":
        _unlocked(_obj(state, a["id"], active=False))
        del state["objs"][str(a["id"])]
        return f"item {a['id']} is gone for good" if verb else "done"
    if op == "restore":
        o = _obj(state, a["id"], active=False)
        if o["state"] != "archived":
            raise SimError("badarg")
        o["state"] = "active"
        return f"item {a['id']} is back in service" if verb else "done"
    if op == "lock":
        o = _obj(state, a["id"])
        o["locked"] = True
        return f"item {a['id']} sealed" if verb else "done"
    if op == "unlock":
        o = _obj(state, a["id"])
        o["locked"] = False
        return f"item {a['id']} opened up" if verb else "done"
    if op == "retag":
        o = _unlocked(_obj(state, a["id"]))
        o["tag"] = a["tag"]
        return f"item {a['id']} labelled {a['tag']}" if verb else "done"
    if op == "clone":
        o = _obj(state, a["id"])
        state["clock"] += 1
        new = state["next_id"]
        state["next_id"] += 1
        state["objs"][str(new)] = {**o, "made": state["clock"], "locked": False}
        return f"item {new} created from item {a['id']}" if verb else "done"
    if op == "swap":
        x, y = (_unlocked(_obj(state, a[k])) for k in ("a", "b"))
        if a["a"] == a["b"]:
            raise SimError("badarg")
        x["place"], y["place"] = y["place"], x["place"]
        return f"items {a['a']} and {a['b']} traded places" if verb else "done"
    if op == "bump":
        o = _unlocked(_obj(state, a["id"]))
        o["w"] += 1
        return f"item {a['id']} weighs {o['w']}{s['unit']} now" if verb else "done"
    if op == "pin":
        o = _obj(state, a["id"])
        o["pin"] = True
        return f"item {a['id']} pinned" if verb else "done"
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
                if isinstance(val, str) and re.fullmatch(r"\d{1,6}", val):
                    val = int(val)                        # forgiving: "412" where an int is expected
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
        if spec.get("style") == "fam":
            return wording.render_error(spec["family"], spec["errors"][e.kind]), False
        return f"error {spec['errors'][e.kind]}", False


_CALL_START = re.compile(r"[A-Za-z_]\w*\s*\(")


def parse_program(text: str) -> list[tuple[str, list]]:
    """Model output -> [(name, args)] with exactly ONE call: the first line that starts like a call. Forgiving: a
    leading '>' (the log's prompt marker) and a trailing ';' are dropped; other lines (prose, code fences, predicted
    outputs, further calls) are ignored; a bare word argument is a string. That first line must be exactly one call
    with literal arguments, else it is a syntax error; so is an output without any call."""
    for n, ln in enumerate(text.strip().splitlines(), 1):
        ln = ln.strip()
        ln = ln[1:].strip() if ln.startswith(">") else ln
        ln = ln[:-1].rstrip() if ln.endswith(";") else ln
        if not _CALL_START.match(ln):
            continue
        try:
            node = ast.parse(ln, mode="eval").body
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords):
                raise ValueError
            args = []
            for x in node.args:
                if isinstance(x, ast.Constant) and type(x.value) in (int, str):
                    args.append(x.value)
                elif isinstance(x, ast.Name):
                    args.append(x.id)                     # forgiving: bare word = string
                elif isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub) and isinstance(x.operand, ast.Constant) and type(x.operand.value) is int:
                    args.append(-x.operand.value)
                else:
                    raise ValueError
            return [(node.func.id, args)]
        except (SyntaxError, ValueError):
            raise ProgramError(f"syntax error on line {n}") from None
    raise ProgramError("syntax error: no tool call")


def run_program(spec: dict, state: dict, text: str) -> dict:
    """Execute a program on a COPY of `state`. Returns {state, outputs, values, error: None | {line, obs}}.
    values[i] is the typed answer value of line i (see value_of), None for lines that carry no answer."""
    st = copy.deepcopy(state)
    try:
        prog = parse_program(text)
    except ProgramError as e:
        return {"state": st, "outputs": [], "values": [], "error": {"line": 0, "obs": str(e)}}
    outs, vals = [], []
    fam = spec.get("style") == "fam"
    for n, (name, args) in enumerate(prog, 1):
        before = copy.deepcopy(st) if fam else None
        obs, ok = call(spec, st, name, args)
        outs.append(obs)
        vals.append((_fam_value(spec, before, name, args) if fam else value_of(spec, name, args, obs)) if ok else None)
        if not ok:
            return {"state": st, "outputs": outs, "values": vals, "error": {"line": n, "obs": obs}}
    return {"state": st, "outputs": outs, "values": vals, "error": None}


_ID_LIST = r"(\(none\)|\d+(?:, \d+)*)"
_VALUE_RE = {  # op -> exact grammar of its (simulator-printed) output; group 1 is the answer value
    "find_tag": r"tag [a-z]+: " + _ID_LIST, "count": r"count: (\d+)", "weigh": r"total: (\d+) [a-z]+",
    "extreme": r"(?:heaviest|lightest): (\(none\)|\d+)", "newest": r"(?:newest|oldest): (\(none\)|\d+)",
    "checksum": r"checksum: (\d+)",
}


def _typed(s: str):
    if s == "(none)":
        return []
    xs = [int(x) for x in s.split(", ")]
    return xs[0] if len(xs) == 1 else xs


_VALUE_RE_NOVEL = {  # novel-style output grammars; group 1 is the answer value
    "find_tag": r"\[(\d+(?: \d+)*)?\]", "count": r"n = (\d+)", "count_tag": r"n = (\d+)", "weigh": r"sum = (\d+)[a-z]+",
    "extreme": r"=> (nothing|\d+) \((?:heaviest|lightest)\)", "newest": r"=> (nothing|\d+) \((?:newest|oldest)\)",
    "oldest_at": r"=> (nothing|\d+) \((?:newest|oldest)\)", "checksum": r"digest = (\d+)",
}


def _typed_novel(s):
    if s is None or s == "nothing":
        return []
    xs = [int(x) for x in s.split()]
    return xs[0] if len(xs) == 1 else xs


def value_of(spec: dict, name: str, args: list, obs: str):
    """Typed answer value of one successful call: an int, a list of ints, or None (no answer).
    answer(x) has an exact grammar: an int literal, or a string that is exactly an integer ("42", "-3") or a
    comma-separated list of integers ("101, 202"); anything else ("42e999", "not 42", "42 or 43") is None.
    Tool outputs are parsed with the exact grammar of their op; mutations and inspect carry no answer."""
    if name == "answer":
        (x,) = args
        if isinstance(x, int):
            return x
        m = re.fullmatch(r"\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*", x)
        if not m:
            return None
        xs = [int(v) for v in re.split(r"\s*,\s*", m.group(1))]
        return xs[0] if len(xs) == 1 else xs
    if spec.get("style") == "fam":
        raise AssertionError("fam worlds: typed values come from run_program (_fam_value), never from text")
    if spec.get("style") == "novel":
        rx = _VALUE_RE_NOVEL.get(tool_map(spec)[name]["op"])
        m = re.fullmatch(rx, obs) if rx else None
        return _typed_novel(m.group(1)) if m else None
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
def explore(spec: dict, n_calls: int, random_frac: float = 0.15, followup: float = 0.2, rounds: int = 3) -> tuple[list[dict], dict]:
    """A scripted explorer: `rounds` coverage passes over every tool with sensible arguments, a few error probes
    (locked item, missing id, bad argument), then a round-robin of sensible calls (1 - random_frac) and random probes
    (random_frac: random id, reversed arguments or wrong arity), with an inspect follow-up after some mutations.
    Returns (events, end state). Events are exactly {"call", "obs"} pairs: what an agent at the keyboard would see."""
    rng = random.Random(f"explore-{spec['seed']}")
    st = copy.deepcopy(spec["init"])
    events: list[dict] = []
    tools = spec["tools"]
    by_op = {t["op"]: t for t in tools}

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

    act = lambda o: o["state"] == "active"  # noqa: E731
    free = lambda o: act(o) and not o["locked"]  # noqa: E731
    TARGET = {"restore": lambda o: o["state"] == "archived", "unlock": lambda o: act(o) and o["locked"],
              "lock": lambda o: act(o) and not o["locked"], "inspect": lambda o: True, "clone": act,
              "pin": lambda o: act(o) and not o.get("pin")}

    def sensible(t):
        op, S = t["op"], spec
        out, used = [], set()
        for an, at in OPS[op][0]:
            if at == "id":
                i = pick(TARGET.get(op, free))
                if op == "swap" and used:
                    first = next(iter(used))
                    i = pick(lambda o: free(o) and o["place"] != st["objs"].get(str(first), {}).get("place"))
                used.add(i)
                out.append(i)
            else:
                out.append(rng.choice(S[{"place": "places", "kind": "kinds", "tag": "tags"}[at]]))
        return out

    def maybe_followup(t, a):
        if OPS[t["op"]][1] and rng.random() < followup:
            ids_ = [x for (an, at), x in zip(OPS[t["op"]][0], a) if at == "id"]
            if ids_ and t["op"] != "delete":
                do(by_op["inspect"], [ids_[0]])

    def coverage():
        order = list(tools)
        rng.shuffle(order)
        for t in order:
            a = sensible(t)
            do(t, a)
            maybe_followup(t, a)

    # 1) coverage: every tool once with sensible args
    coverage()
    # 2) errors: locked item, missing id, bad argument (each twice)
    lockable = ("move", "archive", "retag", "swap", "delete") + (("bump", "shrink") if spec.get("style") == "fam" else ())
    mut = [t for t in tools if t["op"] in lockable] or [by_op["inspect"]]
    for _ in range(2):
        t = rng.choice(mut)
        a = sensible(t)
        for k, (an, at) in enumerate(OPS[t["op"]][0]):
            if at == "id":
                a[k] = pick(lambda o: o["state"] == "active" and o["locked"])
        do(t, a)
        do(by_op["inspect"], [rng.choice([x for x in range(100, 1000) if str(x) not in st["objs"]])])
        t = rng.choice(tools)
        do(t, [0 if isinstance(x, str) else "x" for x in sensible(t)])
    # 3) coverage: every tool `rounds - 1` more times
    for _ in range(rounds - 1):
        coverage()
    # 4) round-robin of sensible calls with random probes until the budget
    cycle = []
    while len(events) < n_calls:
        if rng.random() < random_frac:
            t = rng.choice(tools)
            r = rng.random()
            if r < 0.4:
                a = [rng.randint(100, 999) if at == "id" else x for (an, at), x in zip(OPS[t["op"]][0], sensible(t))]
            elif r < 0.7 and len(OPS[t["op"]][0]) > 1:
                a = sensible(t)[::-1]                       # canonical order reversed: often a type error
            else:
                a = sensible(t) + [rng.randint(1, 9)]       # wrong arity
            do(t, a)
            continue
        if not cycle:
            cycle = list(tools)
            rng.shuffle(cycle)
        t = cycle.pop()
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


TEMPLATES = {  # name -> weight; each template is ONE call of the op of the same name (only ops present in the world)
    "archive": 2, "delete": 2, "move": 2, "lock": 2, "unlock": 2, "restore": 2, "retag": 2, "swap": 1, "clone": 1,
    "count": 1, "weigh": 1, "extreme": 1, "newest": 1, "checksum": 1, "find_tag": 1,
    "bump": 2, "pin": 2, "count_tag": 1, "oldest_at": 1,       # novel worlds only (absent from standard worlds)
    "shrink": 2, "unpin": 2, "count_place": 1, "lightest_kind": 1,   # fam validation worlds only
}


def make_tasks(spec: dict, end: dict, events: list[dict], n: int, salt: str) -> list[dict]:
    """n held-out goals starting from the post-exploration state `end`. Gold = the reference program's outcome on the
    simulator. Rejected: duplicate goals; mutation tasks whose reference lines ALL occur verbatim in the transcript
    (query tasks have a tiny argument space, so their calls usually do occur; their answers are computed at `end`);
    reference programs that error; state tasks whose gold state equals the start state."""
    if spec.get("style") == "fam":
        return _fam_tasks(spec, end, events, n, salt)
    rng = random.Random(f"tasks-{spec['seed']}-{salt}")
    S, U = spec, spec["unit"]
    seen_calls = {e["call"] for e in events}
    objs = end["objs"]

    def ids(pred):
        return sorted(int(i) for i, o in objs.items() if pred(o))

    act = lambda o: o["state"] == "active"  # noqa: E731
    free = lambda o: act(o) and not o["locked"]  # noqa: E731
    out, goals = [], set()
    present = {t["op"] for t in spec["tools"]}
    names, weights = zip(*((k, w) for k, w in TEMPLATES.items() if k in present))
    for _ in range(50 * n):
        if len(out) >= n:
            break
        tpl = rng.choices(names, weights)[0]
        P, K = rng.choice(S["places"]), rng.choice(S["kinds"])
        try:
            g = (_task_novel if S.get("style") == "novel" else _task)(tpl, S, U, rng, ids, act, free, P, K, objs)
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
    raise AssertionError(tpl)


def _task_novel(tpl, S, U, rng, ids, act, free, P, K, objs):
    """Reworded goals for the novel split (same reference programs and gold as _task for standard ops)."""
    c = lambda op, *a: _c(S, op, *a)  # noqa: E731
    pick = lambda pred: rng.choice(ids(pred))  # noqa: E731
    var = lambda op: _tool(S, op)["var"]  # noqa: E731
    if tpl == "archive":
        i = pick(free)
        return f"Put item {i} away into storage.", [c("archive", i)], None
    if tpl == "delete":
        i = pick(lambda o: not o["locked"])
        return f"Get rid of item {i} for good.", [c("delete", i)], None
    if tpl == "move":
        i = pick(lambda o: free(o) and o["place"] != P)
        return f"Item {i} should now sit at {P}.", [c("move", i, P)], None
    if tpl == "lock":
        i = pick(lambda o: act(o) and not o["locked"])
        return f"Seal item {i} so it cannot be changed.", [c("lock", i)], None
    if tpl == "unlock":
        i = pick(lambda o: act(o) and o["locked"])
        return f"Open up the sealed item {i}.", [c("unlock", i)], None
    if tpl == "restore":
        i = pick(lambda o: o["state"] == "archived")
        return f"Bring item {i} back into service.", [c("restore", i)], None
    if tpl == "retag":
        i = pick(lambda o: free(o))
        t = rng.choice([x for x in S["tags"] if x != objs[str(i)]["tag"]])
        return f"Item {i} needs the label {t}.", [c("retag", i, t)], None
    if tpl == "swap":
        a = pick(free)
        b = pick(lambda o: free(o) and o["place"] != objs[str(a)]["place"])
        return f"Items {a} and {b} should trade places.", [c("swap", a, b)], None
    if tpl == "clone":
        i = pick(act)
        return f"Create a new item from item {i}.", [c("clone", i)], None
    if tpl == "bump":
        i = pick(free)
        return f"Item {i} should weigh one {U} more.", [c("bump", i)], None
    if tpl == "pin":
        i = pick(lambda o: act(o) and not o["pin"])
        return f"Mark item {i} as pinned.", [c("pin", i)], None
    if tpl == "count":
        v = var("count")
        what = f"open {K} items" if v["skip_locked"] else f"{K} items, sealed ones included,"
        n = len(ids(lambda o: act(o) and o["kind"] == K and o["place"] == P and not (v["skip_locked"] and o["locked"])))
        return f"Tell me the number of {what} at {P}.", [c("count", K, P)], {"kind": "int", "value": n}
    if tpl == "count_tag":
        t = rng.choice(S["tags"])
        n = len(ids(lambda o: act(o) and o["tag"] == t))
        return f"Tell me the number of items in service labelled {t}.", [c("count_tag", t)], {"kind": "int", "value": n}
    if tpl == "weigh":
        v = var("weigh")
        tot = sum(objs[str(i)]["w"] for i in ids(lambda o: act(o) and o["kind"] == K))
        unit, val = (U, tot) if v["unit"] == "display" else (S["base_unit"], tot * S["C"])
        return f"Report the combined weight of every {K} item, in {unit}.", [c("weigh", K)], {"kind": "int", "value": val}
    if tpl in ("extreme", "newest", "oldest_at"):
        v = var(tpl)
        if tpl == "extreme":
            items = [(int(i), o) for i, o in objs.items() if act(o) and o["place"] == P]
            key = lambda x: (-x[1]["w"] if v["which"] == "heaviest" else x[1]["w"], x[0])  # noqa: E731
            goal = f"Report the {v['which']} item at {P} (ties: lowest id)."
        else:
            items = [(int(i), o) for i, o in objs.items() if act(o) and (o["kind"] == K if tpl == "newest" else o["place"] == P)]
            key = lambda x: -x[1]["made"] if v["which"] == "newest" else x[1]["made"]  # noqa: E731
            age = "youngest" if v["which"] == "newest" else "eldest"
            goal = f"Report the {age} {K} item." if tpl == "newest" else f"Report the {age} item at {P}."
        if not items:
            return None
        return goal, [c(tpl, K if tpl == "newest" else P)], {"kind": "int", "value": min(items, key=key)[0]}
    if tpl == "checksum":
        return f"Report the digest of {P}.", [c("checksum", P)], {"kind": "int", "value": _checksum(S, objs, P)}
    if tpl == "find_tag":
        t = rng.choice(S["tags"])
        got = ids(lambda o: act(o) and o["tag"] == t)
        if not got:
            return None
        return f"List the items in service labelled {t}.", [c("find_tag", t)], {"kind": "set", "value": got}
    raise AssertionError(tpl)


def _checksum(S, objs, P) -> int:
    v = _tool(S, "checksum")["var"]
    return sum(int(i) for i, o in objs.items() if o["state"] == "active" and o["place"] == P
               and (v["with_locked"] or not o["locked"])) * v["mul"] % v["mod"]


# ------------------------------------------------------------------ v2 "fam" worlds (V2-DESIGN-sol.md)
def _gen_fam(seed: int, bank: str, extra: tuple) -> dict:
    """A v2 campaign world: FAM_STD standard task ops + the `extra` op types, wording family from `bank`, tool names
    drawn independently of semantics (two pseudo-words, no English verb), pin flags on every object."""
    rng = random.Random(f"fauxos-fam-{bank}-{seed}")
    used: set[str] = set()
    kinds = [_fresh(rng, used, lambda: pseudo(rng, 1).upper()) for _ in range(FAM_VOCAB)]
    places = [_fresh(rng, used, lambda: pseudo(rng, 2)) for _ in range(FAM_VOCAB)]
    tags = [_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(FAM_VOCAB)]
    unit, base_unit = (_fresh(rng, used, lambda: pseudo(rng, 1)) for _ in range(2))
    C = rng.randint(2, 9)
    letter = rng.choice("ABDEFGHJKMNPQRTVWXYZ")
    codes = rng.sample(range(10, 100), 3)
    errors = {k: f"{letter}{c}" for k, c in zip(("locked", "missing", "badarg"), codes)}
    ops = ["inspect"] + sorted(rng.sample(TASK_OPS, FAM_STD) + list(extra), key=list(OPS).index)
    names: set[str] = set()
    tools = []
    for op in ops:
        name = _fresh(rng, names, lambda: f"{pseudo(rng, 1)}_{pseudo(rng, rng.randint(1, 2))}")
        perm = list(range(len(OPS[op][0])))
        rng.shuffle(perm)
        tools.append({"name": name, "op": op, "perm": perm, "var": _variant(rng, op), "verbose": True})
    objs = _fam_objs(rng, kinds, places, tags)
    return {"seed": seed, "style": "fam", "kinds": kinds, "places": places, "tags": tags, "unit": unit,
            "base_unit": base_unit, "C": C, "errors": errors, "tools": tools, "family": wording.make_family(rng, bank),
            "init": {"objs": objs, "clock": len(objs), "next_id": 1000}}


def regenerate(spec: dict) -> dict:
    """The world `spec` regenerated from its seed and style (tests: data matches this generator)."""
    st = spec.get("style", "std")
    if st == "fam":
        ops = {t["op"] for t in spec["tools"]}
        extra = tuple(op for op in list(DEV_OPS) + list(RESERVED_OPS) if op in ops)
        return gen_world(spec["seed"], "fam", spec["family"]["bank"], extra)
    return gen_world(spec["seed"], st)


def error_text(spec: dict, code: str) -> str:
    """The observation of an error with `code` in this world's wording."""
    return wording.render_error(spec["family"], code) if spec.get("style") == "fam" else f"error {code}"


def _fam_objs(rng, kinds, places, tags) -> dict:
    n = rng.randint(*FAM_OBJ)
    ids = rng.sample(range(100, 1000), n)
    made = list(range(1, n + 1))
    rng.shuffle(made)
    objs = {}
    for i, m in zip(ids, made):
        objs[str(i)] = {"kind": rng.choice(kinds), "place": rng.choice(places), "w": rng.randint(1, 9),
                        "locked": rng.random() < 0.2, "state": "archived" if rng.random() < 0.15 else "active",
                        "made": m, "tag": rng.choice(tags) if rng.random() < 0.5 else "-", "pin": rng.random() < 0.3}
    keys = sorted(objs, key=int)
    for i in keys:                                # enough archived and active-locked objects for balanced tasks
        if sum(o["state"] == "archived" for o in objs.values()) >= FAM_MIN:
            break
        if not objs[i]["locked"]:
            objs[i]["state"] = "archived"
    for i in reversed(keys):
        if sum(o["state"] == "active" and o["locked"] for o in objs.values()) >= FAM_MIN:
            break
        if objs[i]["state"] == "active":
            objs[i]["locked"] = True
    return objs


def _gen_fam_twin(spec: dict, seed: int) -> dict:
    """Twin of a fam world: same tool names, op set, argument vocabulary and WORDING FAMILY; deranged name -> op
    semantics, fresh argument orders, variants, objects, error codes and unit conversion."""
    rng = random.Random(f"fauxos-fam-twin-{seed}")
    ops, names = [t["op"] for t in spec["tools"]], [t["name"] for t in spec["tools"]]
    while True:
        sigma = rng.sample(range(len(ops)), len(ops))
        if all(sigma[k] != k for k in range(len(ops))):
            break
    tools = []
    for k, name in enumerate(names):
        op = ops[sigma[k]]
        perm = list(range(len(OPS[op][0])))
        rng.shuffle(perm)
        tools.append({"name": name, "op": op, "perm": perm, "var": _variant(rng, op), "verbose": True})
    tools.sort(key=lambda t: list(OPS).index(t["op"]))
    letter = rng.choice("ABDEFGHJKMNPQRTVWXYZ")
    codes = rng.sample(range(10, 100), 3)
    objs = _fam_objs(rng, spec["kinds"], spec["places"], spec["tags"])
    return {"seed": seed, "twin_of": spec["seed"], "style": "fam",
            **{k: copy.deepcopy(spec[k]) for k in ("kinds", "places", "tags", "unit", "base_unit", "family")},
            "C": rng.randint(2, 9), "errors": {k: f"{letter}{c}" for k, c in zip(("locked", "missing", "badarg"), codes)},
            "tools": tools, "init": {"objs": objs, "clock": len(objs), "next_id": 1000}}


def _fam_query(spec: dict, state: dict, op: str, v: dict, a: dict):
    """Typed answer of a query op (int, id or [] for none, sorted id list)."""
    if op == "count":
        return len(_active(state, lambda o: o["kind"] == a["kind"] and o["place"] == a["place"] and not (v["skip_locked"] and o["locked"])))
    if op == "weigh":
        tot = sum(o["w"] for _, o in _active(state, lambda o: o["kind"] == a["kind"]))
        return tot if v["unit"] == "display" else tot * spec["C"]
    if op == "extreme":
        items = _active(state, lambda o: o["place"] == a["place"])
        sign = -1 if v["which"] == "heaviest" else 1
        return min(items, key=lambda x: (sign * x[1]["w"], x[0]))[0] if items else []
    if op in ("newest", "oldest_at"):
        pred = (lambda o: o["kind"] == a["kind"]) if op == "newest" else (lambda o: o["place"] == a["place"])
        items = _active(state, pred)
        sign = -1 if v["which"] == "newest" else 1
        return min(items, key=lambda x: sign * x[1]["made"])[0] if items else []
    if op == "lightest_kind":
        items = _active(state, lambda o: o["kind"] == a["kind"])
        return min(items, key=lambda x: (x[1]["w"], x[0]))[0] if items else []
    if op == "find_tag":
        return [i for i, _ in _active(state, lambda o: o["tag"] == a["tag"])]
    if op == "count_tag":
        return len(_active(state, lambda o: o["tag"] == a["tag"]))
    if op == "count_place":
        return len(_active(state, lambda o: o["place"] == a["place"]))
    if op == "checksum":
        items = _active(state, lambda o: o["place"] == a["place"] and (v["with_locked"] or not o["locked"]))
        return sum(i for i, _ in items) * v["mul"] % v["mod"]
    raise AssertionError(op)


def _apply_fam(spec: dict, state: dict, tool: dict, a: dict) -> tuple[str, object]:
    """(observation rendered with the world's wording family, typed value or None). Same semantics as apply()."""
    op, v, f = tool["op"], tool["var"], spec["family"]
    if op == "inspect":
        return wording.render_inspect(f, a["id"], _obj(state, a["id"], active=False), spec["unit"]), None
    if not OPS[op][1]:
        val = _fam_query(spec, state, op, v, a)
        if op == "find_tag":
            txt = wording.flist(f, val)
        elif op in ("extreme", "newest", "oldest_at", "lightest_kind"):
            txt = f["empty"] if val == [] else str(val)
        elif op == "weigh":
            txt = f["weight"].format(w=val, u=spec["unit"] if v["unit"] == "display" else spec["base_unit"])
        else:
            txt = str(val)
        return wording.render_query(f, op, txt, v.get("which")), val
    i = a.get("id", a.get("a"))
    kw = {}
    if op == "move":
        o = _unlocked(_obj(state, a["id"]))
        o["place"] = kw["place"] = a["place"]
    elif op == "archive":
        _unlocked(_obj(state, a["id"]))["state"] = "archived"
    elif op == "delete":
        _unlocked(_obj(state, a["id"], active=False))
        del state["objs"][str(a["id"])]
    elif op == "restore":
        o = _obj(state, a["id"], active=False)
        if o["state"] != "archived":
            raise SimError("badarg")
        o["state"] = "active"
    elif op == "lock":
        _obj(state, a["id"])["locked"] = True
    elif op == "unlock":
        _obj(state, a["id"])["locked"] = False
    elif op == "retag":
        o = _unlocked(_obj(state, a["id"]))
        o["tag"] = kw["tag"] = a["tag"]
    elif op == "clone":
        o = _obj(state, a["id"])
        state["clock"] += 1
        new = state["next_id"]
        state["next_id"] += 1
        state["objs"][str(new)] = {**o, "made": state["clock"], "locked": False}
        kw["new"] = wording.fid(f, new)
    elif op == "swap":
        x, y = (_unlocked(_obj(state, a[k])) for k in ("a", "b"))
        if a["a"] == a["b"]:
            raise SimError("badarg")
        x["place"], y["place"] = y["place"], x["place"]
        kw["other"] = wording.fid(f, a["b"])
    elif op in ("bump", "shrink"):
        o = _unlocked(_obj(state, a["id"]))
        if op == "shrink" and o["w"] <= 1:
            raise SimError("badarg")
        o["w"] += 1 if op == "bump" else -1
        kw["w"] = f["weight"].format(w=o["w"], u=spec["unit"])
    elif op in ("pin", "unpin"):
        _obj(state, a["id"])["pin"] = op == "pin"
    else:
        raise AssertionError(op)
    return (wording.render_mut(f, op, i, **kw) if tool["verbose"] else f["quiet"]), None


def _fam_value(spec: dict, before: dict, name: str, args: list):
    """Typed value of one successful call in a fam world, from the state BEFORE the call (queries do not mutate)."""
    if name == "answer":
        return value_of({}, name, args, "")
    tool = tool_map(spec)[name]
    if OPS[tool["op"]][1] or tool["op"] == "inspect":
        return None
    canon = OPS[tool["op"]][0]
    a = {}
    for pos, ci in enumerate(tool["perm"]):
        (an, at), val = canon[ci], args[pos]
        a[an] = int(val) if at == "id" and isinstance(val, str) else val
    val = _fam_query(spec, before, tool["op"], tool["var"], a)
    return val[0] if isinstance(val, list) and len(val) == 1 and tool["op"] == "find_tag" else val


def _fam_tasks(spec: dict, end: dict, events: list[dict], n_per_op: int, salt: str) -> list[dict]:
    """Balanced held-out goals: exactly n_per_op tasks for every task op of the world, goals in the world's family
    wording. Same rejection rules as make_tasks. Raises if an op cannot reach n_per_op (prepare then skips the seed)."""
    rng = random.Random(f"tasks-fam-{spec['seed']}-{salt}")
    S, f = spec, spec["family"]
    seen_calls = {e["call"] for e in events}
    objs = end["objs"]
    act = lambda o: o["state"] == "active"  # noqa: E731
    free = lambda o: act(o) and not o["locked"]  # noqa: E731

    def ids(pred):
        return sorted(int(i) for i, o in objs.items() if pred(o))

    logged = _fam_logged_values(spec, events, end)
    out, goals = [], set()
    for op in [t["op"] for t in S["tools"] if t["op"] != "inspect"]:
        got = 0
        for _ in range(400):
            if got >= n_per_op:
                break
            try:
                g = _fam_task(op, S, f, rng, ids, act, free, objs)
            except IndexError:
                continue
            if g is None:
                continue
            goal, prog, answer = g
            if goal in goals or answer is None and all(line in seen_calls for line in prog):
                continue
            if answer is not None and len(prog) == 1 and prog[0] in logged and logged[prog[0]] == answer:
                continue                             # the transcript already shows this call with this answer
            res = run_program(spec, end, "\n".join(prog))
            if res["error"] is not None:
                continue
            check_state = OPS[op][1]
            if check_state and res["state"] == end:
                continue
            task = {"goal": goal, "template": op, "reference": prog, "n_calls": len(prog), "check_state": check_state,
                    "gold_state": res["state"] if check_state else None, "answer": answer,
                    "stratum": "reserved" if op in RESERVED_OPS else "dev" if op in DEV_OPS else "familiar"}
            if score(task, res) != 1.0:
                continue
            goals.add(goal)
            out.append(task)
            got += 1
        if got < n_per_op:
            raise RuntimeError(f"world {spec['seed']}: op {op} reached only {got} tasks")
    return out


def _fam_logged_values(spec: dict, events: list[dict], end: dict) -> dict:
    """call string -> typed value of its LAST successful logged occurrence (replays the transcript from init)."""
    st, logged = copy.deepcopy(spec["init"]), {}
    for e in events:
        res = run_program(spec, st, e["call"])
        if res["error"] is None:
            if res["values"] and res["values"][0] is not None:
                logged[e["call"]] = res["values"][0]
            st = res["state"]
    assert st == end, "transcript replay diverged from the explorer's end state"
    return logged


def _fam_task(op, S, f, rng, ids, act, free, objs):
    c = lambda o, *a: _c(S, o, *a)  # noqa: E731
    pick = lambda pred: rng.choice(ids(pred))  # noqa: E731
    v = _tool(S, op)["var"]
    P, K, T, U = rng.choice(S["places"]), rng.choice(S["kinds"]), rng.choice(S["tags"]), S["unit"]
    G = f["goal"][op]
    if op in ("archive", "lock", "pin", "bump", "shrink", "unpin", "delete", "unlock", "restore", "clone"):
        pred = {"archive": free, "bump": free, "delete": lambda o: not o["locked"],
                "lock": lambda o: act(o) and not o["locked"], "unlock": lambda o: act(o) and o["locked"],
                "restore": lambda o: o["state"] == "archived", "clone": act,
                "pin": lambda o: act(o) and not o["pin"], "unpin": lambda o: act(o) and o["pin"],
                "shrink": lambda o: free(o) and o["w"] > 1}[op]
        i = pick(pred)
        return G.format(i=i, U=U), [c(op, i)], None
    if op == "move":
        i = pick(lambda o: free(o) and o["place"] != P)
        return G.format(i=i, P=P), [c(op, i, P)], None
    if op == "retag":
        i = pick(free)
        T = rng.choice([x for x in S["tags"] if x != objs[str(i)]["tag"]])
        return G.format(i=i, T=T), [c(op, i, T)], None
    if op == "swap":
        i = pick(free)
        j = pick(lambda o: free(o) and o["place"] != objs[str(i)]["place"])
        return G.format(i=i, j=j), [c(op, i, j)], None
    a = {"count": {"kind": K, "place": P}, "weigh": {"kind": K}, "extreme": {"place": P}, "newest": {"kind": K},
         "find_tag": {"tag": T}, "checksum": {"place": P}, "count_tag": {"tag": T}, "oldest_at": {"place": P},
         "count_place": {"place": P}, "lightest_kind": {"kind": K}}[op]
    st = {"objs": objs}
    val = _fam_query(S, st, op, v, a)
    if val == [] and op != "find_tag" or op == "find_tag" and not val:
        return None
    canon = [a[an] for an, _ in OPS[op][0]]
    kw = {"P": P, "K": K, "T": T, "U": U}
    if op == "count":
        kw["what"] = f["what"][str(v["skip_locked"])].format(K=K)
    if op == "weigh":
        kw["U"] = U if v["unit"] == "display" else S["base_unit"]
    if op == "extreme":
        kw["which"] = f["which"][v["which"]]
    if op in ("newest", "oldest_at"):
        kw["age"] = f["age"][v["which"]]
    ans = {"kind": "set", "value": val} if op == "find_tag" else {"kind": "int", "value": val}
    return G.format(**kw), [c(op, *canon)], ans
