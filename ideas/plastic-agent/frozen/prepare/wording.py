"""Frozen wording families for FauxOS v2 ("fam" style worlds; PREPARE/EVALUATE-only, never visible to RUN).

A family fixes, for one world, how every observation and every goal is phrased: id / list / empty-result formats,
the structure of mutation reports and query results, per-op verbs and labels, the field labels, separators and field
order of `inspect`, the error format, and one goal template per op. Bank "A" (validation) and bank "B" (holdout) share
NO template string, so wording learned from validation feedback does not transfer verbatim to the holdout. A world's
twin uses the same family. Semantics live in fauxos.py; this file only renders.
"""
from __future__ import annotations

import random

# ------------------------------------------------------------------ formats (bank -> options)
ID = {"A": ["#{i}", "item {i}", "obj {i}"], "B": ["[{i}]", "unit {i}", "id:{i}"]}
LIST = {"A": [(", ", "(none)"), (" ", "[]")], "B": [("; ", "nothing"), (" / ", "--")]}   # (separator, empty)
EMPTY = {"A": ["(none)", "n/a"], "B": ["nothing", "--"]}
MUT = {"A": ["{id} {verb}", "{verb}: {id}"], "B": ["ok - {id} {verb}", "{id} => {verb}"]}
QUERY = {"A": ["{label}: {val}", "{label} = {val}"], "B": ["{val} ({label})", "result {label} -> {val}"]}
ERROR = {"A": ["error {c}", "err {c}"], "B": ["{c}: refused", "fail [{c}]"]}
QUIET = {"A": ["ok"], "B": ["done"]}
WEIGHT = {"A": ["{w} {u}"], "B": ["{w}{u}"]}
INSPECT_SEP = {"A": [(" ", "{k}={v}")], "B": [(", ", "{k}: {v}"), (" | ", "{k} {v}")]}
FIELDS = {  # field -> per bank label options
    "kind": {"A": ["kind", "type"], "B": ["class", "sort"]},
    "place": {"A": ["place", "loc"], "B": ["site", "where"]},
    "w": {"A": ["weight", "wt"], "B": ["mass", "load"]},
    "made": {"A": ["made", "born"], "B": ["created", "age"]},
    "locked": {"A": ["locked", "lock"], "B": ["sealed", "fixed"]},
    "state": {"A": ["state", "status"], "B": ["mode", "phase"]},
    "tag": {"A": ["tag", "label"], "B": ["mark", "badge"]},
    "pin": {"A": ["pinned", "pin"], "B": ["flagged", "bookmarked"]},
}
YESNO = {"A": [("yes", "no"), ("y", "n")], "B": [("true", "false"), ("on", "off")]}
STATE_WORDS = {"A": [("active", "archived")], "B": [("live", "stored"), ("in use", "shelved")]}

# mutation verbs: op -> bank -> options; placeholders {place} {tag} {new} {other} {w}
VERB = {
    "move": {"A": ["moved to {place}", "sent to {place}"], "B": ["relocated to {place}", "now located at {place}"]},
    "archive": {"A": ["archived", "shelved"], "B": ["put in storage", "filed away"]},
    "delete": {"A": ["deleted", "removed"], "B": ["destroyed", "gone for good"]},
    "restore": {"A": ["restored", "revived"], "B": ["back in service", "taken out of storage"]},
    "lock": {"A": ["locked", "secured"], "B": ["sealed shut", "frozen"]},
    "unlock": {"A": ["unlocked", "released"], "B": ["unsealed", "thawed"]},
    "retag": {"A": ["tagged {tag}", "labelled {tag}"], "B": ["marked {tag}", "badge set to {tag}"]},
    "clone": {"A": ["copied as {new}", "cloned to {new}"], "B": ["duplicated into {new}", "twin created: {new}"]},
    "swap": {"A": ["swapped with {other}", "switched places with {other}"], "B": ["traded spots with {other}", "exchanged with {other}"]},
    "bump": {"A": ["weight raised to {w}", "grew to {w}"], "B": ["now weighs {w}", "heavier: {w}"]},
    "pin": {"A": ["pinned", "starred"], "B": ["flagged", "bookmarked"]},
    "shrink": {"A": ["weight lowered to {w}", "shrank to {w}"], "B": ["now lighter: {w}", "reduced to {w}"]},
    "unpin": {"A": ["unpinned", "unstarred"], "B": ["unflagged", "bookmark removed"]},
}
# query labels: op -> bank -> options ({which} = the variant word where the op has one)
LABEL = {
    "count": {"A": ["count", "tally"], "B": ["number", "how many"]},
    "weigh": {"A": ["total", "sum"], "B": ["combined", "overall"]},
    "extreme": {"A": ["{which}", "{which} item"], "B": ["the {which}", "{which} one"]},
    "newest": {"A": ["{which}", "{which} item"], "B": ["the {which}", "{which} one"]},
    "find_tag": {"A": ["tagged", "matches"], "B": ["carrying it", "hits"]},
    "checksum": {"A": ["checksum", "digest"], "B": ["hash", "fingerprint"]},
    "count_tag": {"A": ["tag count", "with tag"], "B": ["bearing it", "tag total"]},
    "oldest_at": {"A": ["{which} here", "{which} at site"], "B": ["the {which} there", "{which} in place"]},
    "count_place": {"A": ["present", "here"], "B": ["on site", "located"]},
    "lightest_kind": {"A": ["lightest", "least heavy"], "B": ["the lightest", "smallest load"]},
}
WHICH = {"heaviest": {"A": ["heaviest"], "B": ["weightiest"]}, "lightest": {"A": ["lightest"], "B": ["featherweight"]},
         "newest": {"A": ["newest"], "B": ["youngest"]}, "oldest": {"A": ["oldest"], "B": ["eldest"]}}

# goal templates: op -> bank -> options. Placeholders: {i} {j} {P} {K} {T} {U} {what} {which} {age}
GOAL = {
    "archive": {"A": ["Archive item {i}.", "Move item {i} into the archive."], "B": ["Put {i} into storage.", "File item {i} away."]},
    "delete": {"A": ["Permanently delete item {i}.", "Remove item {i} for good."], "B": ["Destroy {i}.", "Item {i} must be gone permanently."]},
    "move": {"A": ["Move item {i} to {P}.", "Send item {i} to {P}."], "B": ["Relocate {i} so it is at {P}.", "Item {i} belongs at {P} now."]},
    "lock": {"A": ["Lock item {i}.", "Secure item {i} against changes."], "B": ["Seal {i} shut.", "Make item {i} unchangeable."]},
    "unlock": {"A": ["Unlock item {i}.", "Release the lock on item {i}."], "B": ["Unseal {i}.", "Make item {i} changeable again."]},
    "restore": {"A": ["Restore item {i} from the archive.", "Revive archived item {i}."], "B": ["Take {i} out of storage.", "Bring item {i} back into service."]},
    "retag": {"A": ['Tag item {i} with "{T}".', 'Give item {i} the label "{T}".'], "B": ['Mark {i} as "{T}".', 'Set the badge of item {i} to "{T}".']},
    "swap": {"A": ["Swap the places of items {i} and {j}.", "Items {i} and {j} should switch places."], "B": ["Exchange the locations of {i} and {j}.", "Let {i} and {j} trade spots."]},
    "clone": {"A": ["Make a copy of item {i}.", "Clone item {i}."], "B": ["Duplicate {i}.", "Create a twin of item {i}."]},
    "bump": {"A": ["Raise the weight of item {i} by one {U}.", "Item {i} should get one {U} heavier."], "B": ["Add one {U} to the mass of {i}.", "Make {i} weigh one {U} more."]},
    "pin": {"A": ["Pin item {i}.", "Star item {i}."], "B": ["Flag {i}.", "Bookmark item {i}."]},
    "shrink": {"A": ["Lower the weight of item {i} by one {U}.", "Item {i} should get one {U} lighter."], "B": ["Take one {U} off the mass of {i}.", "Make {i} weigh one {U} less."]},
    "unpin": {"A": ["Unpin item {i}.", "Remove the star from item {i}."], "B": ["Unflag {i}.", "Drop the bookmark on item {i}."]},
    "count": {"A": ["How many {what} are in {P}?", "Count the {what} in {P}."], "B": ["What number of {what} sit at {P}?", "Report how many {what} {P} holds."]},
    "weigh": {"A": ["What is the total weight of all {K} items, in {U}?", "Sum the weight of every {K} item, in {U}."], "B": ["Report the combined mass of the {K} items in {U}.", "How much do all {K} items weigh together, in {U}?"]},
    "extreme": {"A": ["Which item in {P} is the {which}? (ties: lowest id)", "Name the {which} item in {P} (ties: lowest id)."], "B": ["Report the {which} item at {P}; break ties by lowest id.", "At {P}, which item is {which}? Lowest id wins ties."]},
    "newest": {"A": ["Which {K} item was created {age}?", "Name the {K} item made {age}."], "B": ["Report the {K} item whose creation was {age}.", "Among {K} items, which one came {age}?"]},
    "find_tag": {"A": ['Which items carry the tag "{T}"?', 'List the items tagged "{T}".'], "B": ['Report every item marked "{T}".', 'Which items bear "{T}"?']},
    "checksum": {"A": ["What is the checksum of {P}?", "Compute the digest of {P}."], "B": ["Report the hash of {P}.", "What fingerprint does {P} have?"]},
    "count_tag": {"A": ['How many active items carry the tag "{T}"?', 'Count the active items tagged "{T}".'], "B": ['Report how many items in service are marked "{T}".', 'What number of live items bear "{T}"?']},
    "oldest_at": {"A": ["Which item in {P} was created {age}?", "Name the item at {P} made {age}."], "B": ["Report the item at {P} whose creation was {age}.", "Among items at {P}, which came {age}?"]},
    "count_place": {"A": ["How many active items are in {P}?", "Count the active items in {P}."], "B": ["Report how many items in service sit at {P}.", "What number of live items does {P} hold?"]},
    "lightest_kind": {"A": ["Which {K} item is the lightest? (ties: lowest id)", "Name the lightest {K} item (ties: lowest id)."], "B": ["Report the {K} item with the smallest mass; lowest id wins ties.", "Among {K} items, which weighs least? Break ties by lowest id."]},
}
AGE = {"newest": {"A": ["most recently", "last"], "B": ["latest", "most lately"]},
       "oldest": {"A": ["least recently", "first"], "B": ["earliest", "longest ago"]}}
WHAT = {  # count variant: skip_locked -> bank -> options ({K} kind)
    True: {"A": ["unlocked {K} items", "{K} items that are not locked"], "B": ["unsealed {K} items", "{K} items that can still change"]},
    False: {"A": ["{K} items (locked or not)", "{K} items, locked ones included"], "B": ["{K} items, sealed or not", "{K} items counting sealed ones"]},
}


def make_family(rng: random.Random, bank: str) -> dict:
    """One world's wording: a JSON-able dict of chosen options (all from `bank`)."""
    ch = lambda table: rng.choice(table[bank])  # noqa: E731
    order = list(FIELDS)
    rng.shuffle(order)
    return {"bank": bank, "id": ch(ID), "list": list(ch(LIST)), "empty": ch(EMPTY), "mut": ch(MUT), "query": ch(QUERY),
            "error": ch(ERROR), "quiet": ch(QUIET), "weight": ch(WEIGHT), "inspect_sep": list(ch(INSPECT_SEP)),
            "fields": {f: rng.choice(FIELDS[f][bank]) for f in FIELDS}, "field_order": order,
            "yesno": list(ch(YESNO)), "state_words": list(ch(STATE_WORDS)),
            "verb": {op: rng.choice(v[bank]) for op, v in VERB.items()},
            "label": {op: rng.choice(v[bank]) for op, v in LABEL.items()},
            "which": {w: rng.choice(v[bank]) for w, v in WHICH.items()},
            "goal": {op: rng.choice(v[bank]) for op, v in GOAL.items()},
            "age": {w: rng.choice(v[bank]) for w, v in AGE.items()},
            "what": {str(k): rng.choice(v[bank]) for k, v in WHAT.items()}}


# ------------------------------------------------------------------ rendering
def fid(f: dict, i) -> str:
    return f["id"].format(i=i)


def flist(f: dict, ids: list) -> str:
    sep, empty = f["list"]
    return sep.join(str(i) for i in ids) if ids else empty


def render_mut(f: dict, op: str, i, **kw) -> str:
    return f["mut"].format(id=fid(f, i), verb=f["verb"][op].format(**kw))


def render_query(f: dict, op: str, val: str, which: str | None = None) -> str:
    label = f["label"][op].format(which=f["which"][which] if which else "")
    return f["query"].format(label=label, val=val)


def render_inspect(f: dict, i, o: dict, unit: str) -> str:
    sep, kv = f["inspect_sep"]
    yes, no = f["yesno"]
    act, arch = f["state_words"]
    vals = {"kind": o["kind"], "place": o["place"], "w": f["weight"].format(w=o["w"], u=unit), "made": o["made"],
            "locked": yes if o["locked"] else no, "state": act if o["state"] == "active" else arch, "tag": o["tag"],
            "pin": yes if o.get("pin") else no}
    return fid(f, i) + sep + sep.join(kv.format(k=f["fields"][k], v=vals[k]) for k in f["field_order"])


def render_error(f: dict, code: str) -> str:
    return f["error"].format(c=code)


def all_strings(bank: str) -> set:
    """Every literal option of a bank (tests: banks A and B share none)."""
    out = set()
    for table in (ID, EMPTY, MUT, QUERY, ERROR, QUIET):
        out |= set(table[bank])
    for table in (VERB, LABEL, GOAL, FIELDS, WHICH, AGE):
        for v in table.values():
            out |= set(v[bank])
    for v in WHAT.values():
        out |= set(v[bank])
    for sep, empty in LIST[bank]:
        out.add(empty)
    return out
