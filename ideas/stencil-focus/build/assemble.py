"""Validate the four Codex outputs (build/gen.sh) and freeze them into frozen/prepare/paraphrases.json.

  python3 build/assemble.py <gen-dir> <snapshot-dir>      (gen-dir holds <task>/<task>.json)

Checks (a string failing any is dropped and reported; lists are then re-checked for size):
  ASCII, one line, exact placeholder set per kind, VPs start lowercase and carry no final punctuation, frames and
  talk end with . ! or ?, every rendering (all slot values x a frame) is exactly one focus3.sentences span, no
  5-word run from topics.json instruction texts or eval queries, no duplicates.
"""
import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "frozen" / "prepare"))
import bank  # noqa: E402

gen, snap = Path(sys.argv[1]), Path(sys.argv[2])
topics = json.loads((snap / "vendor/memorycode/topics.json").read_text())
raw = {t: json.loads((gen / t / f"{t}.json").read_text()) for t in ("conventions", "fillers", "codetalk", "frames")}


def words(s):
    return re.findall(r"[a-z0-9_@']+", s.lower().replace("'", " ' "))


def ngrams(s, n=5):
    w = words(s)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


CANON = set()
for t in topics["instructions"]:
    for x in t["text"]:
        CANON |= ngrams(x)
    CANON |= ngrams(t["eval_query"])
PH = re.compile(r"\{[a-z0-9]+\}")
NEED = {"case": {"{case}"}, "chx": {"{obj}"}, "prefix": {"{obj}", "{affix}"}, "suffix": {"{obj}", "{affix}"},
        "annotation": {"{objs}"}, "try": {"{objs}"}, "assert": {"{objs}"}, "docstring": {"{objs}"}, "comment": set(),
        "import": {"{module}"}, "decorator": {"{decorator}", "{objs}"}, "digit": {"{obj}"}}
PROBE_FRAMES = ["{vp}.", "I'd like you to {vp}, since it keeps reviews quick.", "Please {vp} and {vp2}."]
problems = []


def one_span(sentence):
    sp = bank.sentences(sentence)
    return len(sp) == 1 and sp[0][1] == sentence and bank.TERMINAL.search(sentence)


def common_ok(s, where):
    why = []
    if not isinstance(s, str) or not s.isascii() or "\n" in s or s != s.strip() or "  " in s:
        why.append("format")
    elif ngrams(s) & CANON:
        why.append("copies 5 canonical words")
    if why:
        problems.append((where, s, why))
    return not why


def vp_ok(s, where, need=frozenset()):
    if not common_ok(s, where):
        return False
    why = []
    if set(PH.findall(s)) != set(need) or len(PH.findall(s)) != len(need):
        why.append(f"placeholders {PH.findall(s)} != {sorted(need)}")
    if not s[:1].islower() or bank.TERMINAL.search(s) or s.endswith((",", ";", ":")):
        why.append("not a bare VP")
    values = {"{obj}": list(bank.SINGULAR.values()), "{objs}": list(bank.PLURAL.values()), "{affix}": ["a_", "_vr", "fn_"],
              "{module}": ["secrets"], "{decorator}": ["@retry", "@timer_class"], "{case}": bank.CASES}
    for combo in itertools.product(*[values[k] for k in sorted(need)]):
        r = bank.fill(s, dict(zip(sorted(need), combo)))
        for fr in PROBE_FRAMES:
            if not one_span(bank.frame(fr, vp=r, vp2="run the tests first")):
                why.append(f"not one sentence: {bank.frame(fr, vp=r, vp2='x')!r}")
                break
    if why:
        problems.append((where, s, why))
    return not why


def frame_ok(s, where, keys):
    if not common_ok(s, where):
        return False
    why = []
    if sorted(PH.findall(s)) != sorted("{" + k + "}" for k in keys):
        why.append(f"placeholders {PH.findall(s)}")
    else:
        r = bank.frame(s, **{k: "prefix every method name with 'x_'" for k in keys})
        if not one_span(r):
            why.append(f"not one sentence: {r!r}")
    if why:
        problems.append((where, s, why))
    return not why


def dedup(xs):
    return list(dict.fromkeys(xs))


out = {"conventions": {}, "fillers": {}, "codetalk": {}, "frames": {}}
for kind, need in NEED.items():
    out["conventions"][kind] = dedup(v for v in raw["conventions"][kind] if vp_ok(v, f"conv:{kind}", need))
fvar = {str(f["id"]): f["text"] for f in topics["fillers_instruction"]}
for fid, variants in fvar.items():
    got = raw["fillers"][fid]
    assert len(got) == len(variants), (fid, len(got))
    for k, triple in enumerate(got):
        ok = [v for v in triple if vp_ok(v, f"filler:{fid}:{k}")]
        assert len(ok) == 3 and len(set(ok)) == 3, (fid, k, triple)
    out["fillers"][fid] = got
out["codetalk"]["advice"] = dedup(v for v in raw["codetalk"]["advice"] if vp_ok(v, "advice"))
out["codetalk"]["talk"] = dedup(v for v in raw["codetalk"]["talk"] if common_ok(v, "talk") and (one_span(v) or problems.append(("talk", v, ["not one sentence"]))))
out["frames"]["single"] = dedup(v for v in raw["frames"]["single"] if frame_ok(v, "single", ["vp"]))
out["frames"]["double"] = dedup(v for v in raw["frames"]["double"] if frame_ok(v, "double", ["vp1", "vp2"]))
out["frames"]["update"] = dedup(v for v in raw["frames"]["update"] if frame_ok(v, "update", ["vp"]))

for w, s, why in problems:
    print(f"DROP {w}: {s!r}: {why}")
sizes = {f"conv:{k}": len(v) for k, v in out["conventions"].items()} | {f"codetalk:{k}": len(v) for k, v in out["codetalk"].items()} \
    | {f"frames:{k}": len(v) for k, v in out["frames"].items()}
print(json.dumps(sizes))
assert min(v for k, v in sizes.items() if k.startswith("conv:")) >= 6, "fewer than 2 templates per slice for a kind"
assert sizes["frames:double"] >= 12 and sizes["frames:single"] >= 30 and sizes["frames:update"] >= 15
out["_provenance"] = ("Authored once by Codex (gpt-6-sol, containerized, build/gen.sh + build/prompts/*) on 2026-09-27, "
                      "validated by build/assemble.py; frozen. Never regenerated inside PREPARE.")
dest = HERE.parent / "frozen" / "prepare" / "paraphrases.json"
dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(f"wrote {dest}")
