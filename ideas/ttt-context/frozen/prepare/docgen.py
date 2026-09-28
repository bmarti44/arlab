"""PREPARE-only (mounted at /prepare, never in RUN): frozen, deterministic generator of long synthetic documents
with one question each (pure Python, no LLM).

A document is the operations log of a fictional depot: ~1,100 timestamped one-line events over several days (crates
moved between docks, shipments approved/rejected/reviewed, people assigned to teams, teams moving rooms, locker codes
set, plus routine filler). All names are fresh pseudo-words drawn per document, so no answer can come from world
knowledge. Every document asks ONE question of one of four kinds (answers are short and exact):

  state  "At the end of the log, at which dock is crate K-482?"      last of 3-5 moves of that crate      -> dock name
  hop2   "... which room does the team of <person> occupy?"           person -> last team -> its last room -> number
  count  "How many shipments did <person> approve in total?"          1-6 approvals among many events      -> number
  kv     "What is the latest code for locker <name>?"                 last of 1-3 settings of that locker  -> code

Hard distractors: every question has near-duplicate entities (crate K-428 / R-482, locker Marel / Marell, a person
with the same first or last name) with their own events, the asked entity has confusing events (earlier values,
rejections), and all other events use the same templates. The answer is computed by replaying the event list
(`replay`), and the event list is what gets rendered, so the answer is a function of the document text only.

Length: `build` renders the log, counts tokens with the caller's tokenizer, and removes random unprotected
distractor events until the document is at most `target_tokens` (and within ~40 tokens of it).
"""
from __future__ import annotations

import random

from common import HEAD

# ---- sizes (calibrated in the GPU pilot; a change here changes data_hash -> new data, new tag)
CONTEXT_TOKENS = 8192      # document prompt length target (chat head + log); every doc is within 128 tokens below it
N_ITEMS = 400              # items (one question per document) per split: validation and holdout

KINDS = ("state", "hop2", "count", "kv")
DIFFICULTY = {                      # tune in the GPU pilot so that no_ttt lands at 30-70 % (a change = new data)
                                    # pilot 1 (16K, moves 3-5, count 1-6, 2 decoys): no_ttt 0.175 (n=40)
    "state_moves": (2, 3),          # moves of the asked crate (the last one is the answer)
    "hop2_assigns": (1, 2),         # team assignments of the asked person (the last one counts)
    "hop2_rooms": (1, 2),           # room moves of that team (the last one is the answer)
    "count_approvals": (1, 6),      # approvals by the asked person (= the answer)
    "kv_sets": (1, 2),              # code settings of the asked locker (the last one is the answer)
    "near_duplicates": 2,           # confusable entities per question, each with 1-2 events of its own
}
N_PEOPLE, N_TEAMS, N_DOCKS, N_CRATES, N_LOCKERS = 32, 10, 8, 48, 24

TEMPLATES = {
    "move": "Crate {0} was moved to dock {1}.",
    "approve": "{0} approved shipment {1}.",
    "reject": "{0} rejected shipment {1}.",
    "review": "{0} reviewed shipment {1}.",
    "assign": "{0} was assigned to team {1}.",
    "room": "Team {0} moved into room {1}.",
    "code": "The code for locker {0} was set to {1}.",
    "inspect": "Dock {0} was inspected; no issues were found.",
    "leave": "{0} requested leave for day {1}.",
    "forklift": "Forklift {0} was serviced at dock {1}.",
    "badge": "{0} reported a faulty badge reader near room {1}.",
}
WEIGHTS = {"move": 24, "approve": 10, "reject": 6, "review": 5, "assign": 10, "room": 6, "code": 14,
           "inspect": 8, "leave": 5, "forklift": 6, "badge": 6}
FREE, PROTECTED, CRITICAL = 0, 1, 2     # event flags: only FREE events may be removed when fitting the length

_ONSETS = ("b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr",
           "v", "z", "sh", "th")
_VOWELS = ("a", "e", "i", "o", "u", "ai", "ea", "io", "ou")
_CODAS = ("n", "r", "l", "s", "m", "th", "x", "nd", "rk", "", "", "")
_LETTERS = "ABCDEFGHJKLMNPRSTVWXZ"


def _word(rng: random.Random) -> str:
    n = rng.choice((2, 2, 3))
    return ("".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(n)) + rng.choice(_CODAS)).capitalize()


def _crate(rng: random.Random) -> str:
    return f"{rng.choice(_LETTERS)}-{rng.randint(100, 999)}"


def _unique(rng, make, n, taken: set) -> list:
    out = []
    while len(out) < n:
        x = make(rng)
        if x not in taken:
            taken.add(x)
            out.append(x)
    return out


def _near(rng, x: str, kind: str, n: int, taken: set) -> list[str]:
    """n confusable variants of an entity name that are not used yet."""
    out, tries = [], 0
    while len(out) < n and tries < 1000:
        tries += 1
        if kind == "crate":
            letter, digits = x.split("-")
            if rng.random() < 0.5:
                d = list(digits)
                i, j = rng.sample(range(3), 2)
                d[i], d[j] = d[j], d[i]
                y = f"{letter}-{''.join(d)}"
            else:
                y = f"{rng.choice(_LETTERS)}-{digits}"
        elif kind == "word":
            i = rng.randrange(1, len(x))
            y = (x[:i] + rng.choice("aeiou") + x[i + 1:]) if rng.random() < 0.5 else (x + rng.choice("aeilnrs"))
        else:  # person: same first or same last name
            first, last = x.split(" ")
            y = f"{first} {_word(rng)}" if len(out) % 2 == 0 else f"{_word(rng)} {last}"
        if y != x and y not in taken:
            taken.add(y)
            out.append(y)
    return out


def _world(rng: random.Random) -> dict:
    taken: set = set()
    w = {"depot": _unique(rng, _word, 1, taken)[0]}
    firsts, lasts = _unique(rng, _word, N_PEOPLE, taken), _unique(rng, _word, N_PEOPLE, taken)
    w["people"] = [f"{f} {s}" for f, s in zip(firsts, lasts)]
    taken.update(w["people"])
    w["teams"] = _unique(rng, _word, N_TEAMS, taken)
    w["docks"] = _unique(rng, _word, N_DOCKS, taken)
    w["lockers"] = _unique(rng, _word, N_LOCKERS, taken)
    w["crates"] = _unique(rng, _crate, N_CRATES, taken)
    w["rooms"] = rng.sample(range(100, 900), 3 * N_TEAMS)
    w["taken"] = taken
    return w


class _Ships:
    """Fresh shipment ids S-1000..S-9999, never repeated within a document."""

    def __init__(self, rng):
        self.ids = [f"S-{n}" for n in rng.sample(range(1000, 10000), 4000)]

    def __call__(self):
        return self.ids.pop()


def _chain(rng, pool: list, n: int, avoid_first_last: bool = False) -> list:
    """n values from pool, consecutive ones different (and last != first if asked and n >= 2)."""
    while True:
        seq = [rng.choice(pool)]
        while len(seq) < n:
            seq.append(rng.choice([v for v in pool if v != seq[-1]]))
        if not (avoid_first_last and n >= 2 and seq[0] == seq[-1]):
            return seq


def _question(rng, w: dict, kind: str, ship) -> dict:
    """The asked entity, its critical event chains (order matters), protected confusers and the distractor exclusions."""
    D, nd = DIFFICULTY, DIFFICULTY["near_duplicates"]
    chains, prot, excl = [], [], {}
    if kind == "state":
        crate = rng.choice(w["crates"])
        docks = _chain(rng, w["docks"], rng.randint(*D["state_moves"]), avoid_first_last=True)
        chains.append([("move", (crate, d), CRITICAL) for d in docks])
        for dup in _near(rng, crate, "crate", nd, w["taken"]):
            w["crates"].append(dup)
            prot += [("move", (dup, rng.choice(w["docks"])), PROTECTED) for _ in range(rng.randint(1, 2))]
        excl, target = {"crate": crate}, crate
        q, ans, aliases = (f"At the end of the log, at which dock is crate {crate}? Answer with the dock name only.",
                           docks[-1], [docks[-1], f"dock {docks[-1]}"])
    elif kind == "hop2":
        person = rng.choice(w["people"])
        teams = _chain(rng, w["teams"], rng.randint(*D["hop2_assigns"]))
        team = teams[-1]
        rooms = rng.sample(w["rooms"], rng.randint(*D["hop2_rooms"]))
        chains.append([("assign", (person, t), CRITICAL) for t in teams])
        chains.append([("room", (team, r), CRITICAL) for r in rooms])
        if len(teams) > 1:
            prot.append(("room", (teams[0], rng.choice([r for r in w["rooms"] if r not in rooms])), PROTECTED))
        for dup in _near(rng, person, "person", nd, w["taken"]):
            w["people"].append(dup)
            prot += [("assign", (dup, rng.choice(w["teams"])), PROTECTED) for _ in range(rng.randint(1, 2))]
        excl, target = {"assign_person": person, "room_team": team}, person
        q, ans, aliases = (f"At the end of the log, which room does the team of {person} occupy? "
                           "Answer with the room number only.", str(rooms[-1]), [str(rooms[-1]), f"room {rooms[-1]}"])
    elif kind == "count":
        person = rng.choice(w["people"])
        k = rng.randint(*D["count_approvals"])
        chains.append([("approve", (person, ship()), CRITICAL) for _ in range(k)])
        prot += [(rng.choice(("reject", "review")), (person, ship()), PROTECTED) for _ in range(rng.randint(1, 3))]
        for dup in _near(rng, person, "person", nd, w["taken"]):
            w["people"].append(dup)
            prot += [("approve", (dup, ship()), PROTECTED) for _ in range(rng.randint(1, 2))]
        excl, target = {"approve_person": person}, person
        q, ans, aliases = (f"How many shipments did {person} approve in total? Answer with a number only.",
                           str(k), [str(k)])
    elif kind == "kv":
        locker = rng.choice(w["lockers"])
        codes = [str(c) for c in rng.sample(range(1000, 100000), rng.randint(*D["kv_sets"]))]
        chains.append([("code", (locker, c), CRITICAL) for c in codes])
        for dup in _near(rng, locker, "word", nd, w["taken"]):
            w["lockers"].append(dup)
            prot += [("code", (dup, str(rng.randint(1000, 99999))), PROTECTED) for _ in range(rng.randint(1, 2))]
        excl, target = {"locker": locker}, locker
        q, ans, aliases = (f"What is the latest code for locker {locker}? Answer with the code only.", codes[-1], [codes[-1]])
    else:
        raise ValueError(kind)
    return {"chains": chains, "protected": prot, "excl": excl, "target": target, "question": q, "answer": ans,
            "aliases": aliases}


def _distractor(rng, w: dict, excl: dict, ship, pools: dict):
    typ = rng.choices(pools["types"], pools["weights"])[0]
    if typ == "move":
        a = (rng.choice(pools["crates"]), rng.choice(w["docks"]))
    elif typ == "approve":
        a = (rng.choice(pools["approvers"]), ship())
    elif typ in ("reject", "review"):
        a = (rng.choice(w["people"]), ship())
    elif typ == "assign":
        a = (rng.choice(pools["assignees"]), rng.choice(w["teams"]))
    elif typ == "room":
        a = (rng.choice(pools["room_teams"]), rng.choice(w["rooms"]))
    elif typ == "code":
        a = (rng.choice(pools["lockers"]), str(rng.randint(1000, 99999)))
    elif typ == "inspect":
        a = (rng.choice(w["docks"]),)
    elif typ == "leave":
        a = (rng.choice(w["people"]), rng.randint(1, 30))
    elif typ == "forklift":
        a = (rng.randint(1, 20), rng.choice(w["docks"]))
    else:  # badge
        a = (rng.choice(w["people"]), rng.choice(w["rooms"]))
    return (typ, a, FREE)


def generate(seed: int, kind: str, n_events: int) -> dict:
    """World + question + n_events free distractors, with the question's chains and confusers at random positions."""
    rng = random.Random(seed)
    ship = _Ships(rng)
    w = _world(rng)
    q = _question(rng, w, kind, ship)
    ex = q["excl"]
    pools = {"types": list(WEIGHTS), "weights": list(WEIGHTS.values()),
             "crates": [c for c in w["crates"] if c != ex.get("crate")],
             "approvers": [p for p in w["people"] if p != ex.get("approve_person")],
             "assignees": [p for p in w["people"] if p != ex.get("assign_person")],
             "room_teams": [t for t in w["teams"] if t != ex.get("room_team")],
             "lockers": [x for x in w["lockers"] if x != ex.get("locker")]}
    placed = [((i + 0.5) / n_events, _distractor(rng, w, ex, ship, pools)) for i in range(n_events)]
    for chain in q["chains"]:
        placed += zip(sorted(rng.random() for _ in chain), chain)
    placed += [(rng.random(), e) for e in q["protected"]]
    placed.sort(key=lambda x: x[0])
    return {"seed": seed, "kind": kind, "depot": w["depot"], "events": [e for _, e in placed], "target": q["target"],
            "question": q["question"], "answer": q["answer"], "aliases": q["aliases"]}


def render(doc: dict) -> str:
    """The document prompt text (chat head + log). Timestamps are drawn from the document seed at render time."""
    rng = random.Random(doc["seed"] * 7 + 3)
    lines = [f"Operations log of the {doc['depot']} depot. One event per line, in time order.", "", "Day 1"]
    day, minute = 1, 8 * 60
    for typ, args, _ in doc["events"]:
        minute += rng.randint(1, 12)
        if minute > 18 * 60:
            day, minute = day + 1, 8 * 60 + rng.randint(0, 20)
            lines.append(f"Day {day}")
        lines.append(f"{minute // 60:02d}:{minute % 60:02d} " + TEMPLATES[typ].format(*args))
    return HEAD + "\n".join(lines) + "\n\n"


def replay(doc: dict) -> str:
    """The answer, recomputed from the event list (what the document says), not from how the question was built."""
    ev, t = doc["events"], doc["target"]
    if doc["kind"] == "state":
        return [a[1] for typ, a, _ in ev if typ == "move" and a[0] == t][-1]
    if doc["kind"] == "hop2":
        team = [a[1] for typ, a, _ in ev if typ == "assign" and a[0] == t][-1]
        return str([a[1] for typ, a, _ in ev if typ == "room" and a[0] == team][-1])
    if doc["kind"] == "count":
        return str(sum(1 for typ, a, _ in ev if typ == "approve" and a[0] == t))
    return [a[1] for typ, a, _ in ev if typ == "code" and a[0] == t][-1]


def build(seed: int, kind: str, target_tokens: int, count_fn) -> dict:
    """A document of at most target_tokens tokens (count_fn(text) -> int) and its question/answer."""
    n = max(8, target_tokens // 11)
    for _ in range(8):
        doc = generate(seed, kind, n)
        cnt = count_fn(render(doc))
        if cnt > target_tokens:
            break
        n = int(n * 1.3) + 8
    else:
        raise RuntimeError(f"seed {seed}: cannot reach {target_tokens} tokens")
    free = [i for i, e in enumerate(doc["events"]) if e[2] == FREE]
    random.Random(seed * 13 + 5).shuffle(free)
    events, removed, ptr = doc["events"], set(), 0
    while cnt > target_tokens:
        per_event = cnt / max(1, len(events) - len(removed))
        r = max(1, int((cnt - target_tokens) / per_event))
        if ptr + r > len(free):
            raise RuntimeError(f"seed {seed}: ran out of removable events")
        removed.update(free[ptr:ptr + r])
        ptr += r
        doc["events"] = [e for i, e in enumerate(events) if i not in removed]
        text = render(doc)
        cnt = count_fn(text)
    text = render(doc)
    if replay(doc) != doc["answer"]:
        raise AssertionError(f"seed {seed}: replayed answer {replay(doc)!r} != constructed {doc['answer']!r}")
    return {"seed": seed, "kind": kind, "text": text, "n_tokens": cnt, "question": doc["question"],
            "answer": doc["answer"], "aliases": doc["aliases"], "target": doc["target"], "n_events": len(doc["events"])}
