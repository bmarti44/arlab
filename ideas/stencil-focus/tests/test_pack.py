"""Deterministic CPU checks for stencil-focus (run by `arlab check` as root, no GPU/network, data at /data)."""
import gzip  # noqa: F401  (kept for parity with prepare's readers)
import importlib.util
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tarfile
from collections import Counter

import pytest

sys.path.insert(0, "/frozen")
sys.path.insert(0, "/pack/frozen/prepare")
import bank  # noqa: E402
import window  # noqa: E402

D = "/data"
SPLITS = ("validation", "holdout")
SLICES = ("validation", "holdout", "pilot")
mc, f3 = window._stencil()
TOPICS = json.load(open(f"{D}/validation/private/vendor/memorycode/topics.json"))
BY_ID = {int(t["id"]): t for t in TOPICS["instructions"]}


def rows(path):
    return [json.loads(line) for line in open(path)]


ITEMS = {s: rows(f"{D}/{s}/public/items.jsonl") for s in SLICES}
LABELS = {s: {x["id"]: x for x in rows(f"{D}/{s}/private/labels.jsonl")} for s in SLICES}
REPORT = json.load(open(f"{D}/prepare_report.json"))


def load_surface(path):
    spec = importlib.util.spec_from_file_location("surf_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mentor_spans(item, i):
    dia = window.as_dialogue(item)
    return [x for sp, line in mc.speaker_lines(dia, i) if sp == "mentor" for _, x in f3.sentences(line)]


# ---------------------------------------------------------------- splitter
def test_bank_splitter_is_stencils_and_breaks_on_abbreviations():
    for t in ("Use it, e.g. here. Then stop!", "One. Two? Three", "Name it 'a_'. Done.", "No end"):
        assert bank.sentences(t) == f3.sentences(t)
    assert len(f3.sentences("Prefix names, e.g. with 'x_'.")) == 2       # why every inserted sentence is re-checked


def test_every_inserted_sentence_is_one_splitter_span():
    for sl in SLICES:
        for it in ITEMS[sl]:
            lab = LABELS[sl][it["id"]]
            spans = {i: Counter(mentor_spans(it, i)) for i in range(lab["n_sessions"])}
            for u in lab["units"]:
                assert spans[u["session"]][u["text"]] >= 1, (sl, it["id"], u["text"])


# ---------------------------------------------------------------- data hygiene
def test_g0_trees_excluded():
    snap = [p for p in os.listdir("/pack/frozen/prepare") if p.startswith("stencil-") and p.endswith(".tar.gz")]
    with tarfile.open(f"/pack/frozen/prepare/{snap[0]}") as tf:
        g0 = tf.extractfile("data/g0/chat.jsonl").read().decode()
    roots = {json.loads(line)["id"].split(":")[0] for line in io.StringIO(g0) if line.strip()}
    assert len(roots) == 29 and roots == set(REPORT["g0_root_ids"])
    used = [src for v in REPORT["slices"].values() for src in v["sources"]]
    oasst = {s.split(":", 1)[1] for s in used if s.startswith("oasst2:")}
    assert oasst, "expected some oasst2 chatter"
    assert not oasst & roots
    assert REPORT["chatter_dropped"]["oasst2_g0"] == 29
    for root, _dirs, files in os.walk(D):
        assert "chat.jsonl" not in files, root                      # the g0 file is not kept


def test_splits_sized_and_disjoint():
    sp = json.load(open(f"{D}/splits.json"))
    assert sp == {"validation": len(ITEMS["validation"]), "holdout": len(ITEMS["holdout"])}
    assert min(sp.values()) >= 250
    ids = [set(LABELS[s]) for s in SLICES]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    srcs = [set(REPORT["slices"][s]["sources"]) for s in SLICES]
    assert not (srcs[0] & srcs[1] or srcs[0] & srcs[2] or srcs[1] & srcs[2]), "chatter threads reused across slices"
    for s in SLICES:                                                 # threads never reused within a slice either
        assert len(srcs[SLICES.index(s)]) == len(REPORT["slices"][s]["sources"])
    pv = {s: {p for x in LABELS[s].values() for ev in x["template"] for p, *_ in [e for e in ev if e != -1]} for s in SLICES}
    assert pv["validation"] <= set(bank.SIDE_A) and pv["holdout"] <= set(bank.SIDE_B)
    assert pv["pilot"] & set(bank.SIDE_A) and pv["pilot"] & set(bank.SIDE_B)
    assert not set(bank.SIDE_A) & set(bank.SIDE_B) and len(set(bank.SIDE_A) | set(bank.SIDE_B)) == 51
    conv = {s: {u["text"] for x in LABELS[s].values() for u in x["units"]} for s in SLICES}
    assert not (conv["validation"] & conv["holdout"] or conv["validation"] & conv["pilot"] or conv["holdout"] & conv["pilot"])


def test_split_stratification():
    def stats(sl):
        labs = list(LABELS[sl].values())
        fam = Counter(f for x in labs for f in x["required"])
        return {"updates": statistics.mean(sum(k == "instruction-update" for ks in x["kinds"] for k in ks) for x in labs),
                "live": statistics.mean(len(x["live"]) for x in labs),
                "class": sum(x["structure"] == ["class"] for x in labs) / len(labs),
                "hist": statistics.median(x["history_tokens"] for x in labs),
                "fam": {f: v / len(labs) for f, v in fam.items()}}
    v, h = stats("validation"), stats("holdout")
    assert abs(v["updates"] - h["updates"]) <= 0.25 * max(v["updates"], h["updates"]), (v["updates"], h["updates"])
    assert abs(v["live"] - h["live"]) <= 0.2 * max(v["live"], h["live"])
    assert abs(v["class"] - h["class"]) <= 0.15
    assert abs(v["hist"] - h["hist"]) <= 0.15 * max(v["hist"], h["hist"])
    for f in ("variable", "import"):                                  # always-required families present on both sides
        assert abs(v["fam"].get(f, 0) - h["fam"].get(f, 0)) <= 0.15, f
    upd = {s: {p for p in (bank.SIDE_A if s == "A" else bank.SIDE_B) if p in bank.UPDATABLE} for s in "AB"}
    objs = {s: sorted(BY_ID[p]["regex"][0][0] for p in upd[s]) for s in "AB"}
    assert objs["A"] == objs["B"] and len(upd["A"]) == len(upd["B"]) == 5   # one updatable pivot per object per side
    for s, lo in (("validation", 8000), ("holdout", 8000)):
        assert min(x["history_tokens"] for x in LABELS[s].values()) >= lo - 1000
        assert max(x["current_tokens"] for x in LABELS[s].values()) <= 2300


def test_no_labels_in_public():
    for sl in SLICES:
        pub = f"{D}/{sl}/public"
        allowed = {"items.jsonl", "real.jsonl", "stencil"}
        assert set(os.listdir(pub)) <= allowed, os.listdir(pub)
        mods = sorted(m for m in os.listdir(f"{pub}/stencil/src/stencil") if m != "__pycache__")
        assert mods == sorted(["__init__.py", "memorycode.py", "focus3.py", "focus2.py", "stats.py"])
        assert not os.path.exists(f"{pub}/stencil/vendor")
        for it in ITEMS[sl]:
            assert set(it) == {"id", "mentor", "mentee", "sessions", "query"} and re.fullmatch(r"[0-9a-f]{12}", it["id"])
            text = "\n".join(it["sessions"]).lower()
            for t in TOPICS["instructions"]:
                for x in t["text"]:
                    assert x.lower() not in text, (sl, it["id"], x)          # conventions are never verbatim topics text
    assert not os.path.exists(f"{D}/train") or not os.listdir(f"{D}/train")
    for root, _dirs, files in os.walk(D):
        if "topics.json" in files:
            assert "/private/" in root + "/", root


# ---------------------------------------------------------------- ceiling: a trivial selector must not solve the bank
SELECTORS = {
    "keyword": re.compile(r"\b(names?|naming|prefix\w*|suffix\w*|start\w*|begin\w*|end\w*|annotat\w*|type hints?|try|except|"
                          r"assert\w*|docstrings?|comments?|import\w*|decorat\w*|digits?|uppercase|camelcase|snake_case|"
                          r"always|never|pedantic)\b", re.I),
    "quote": re.compile(r"'[^'\s]{1,20}'"),
    "cue": re.compile(r"\b(i'd like you to|i would like you to|please|make sure|remember to|from now on|going forward|"
                      r"i want you to|i expect you to)\b", re.I),
}
CEILING_F1 = 0.8


@pytest.mark.parametrize("split", SPLITS)
def test_trivial_selector_ceiling(split):
    """Frozen trivial selectors (keyword / quoted-token / frame-cue over mentor sentences of sessions 0..s-1) scored
    against the label-derived live set (the latest statement of every live convention): F1 < 0.8 on both splits,
    else the bank must be regenerated."""
    for name, rx in SELECTORS.items():
        tp = fp = fn = 0
        for it in ITEMS[split]:
            lab = LABELS[split][it["id"]]
            gold = {(lab["units"][k]["session"], lab["units"][k]["text"]) for k in lab["oracle_units"]}
            pred = {(i, x) for i in range(len(it["sessions"]) - 1) for x in mentor_spans(it, i) if rx.search(x)}
            tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"{split} {name}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
        assert f1 < CEILING_F1, (split, name, f1)


# ---------------------------------------------------------------- one bank dialogue end to end
def test_bank_dialogue_end_to_end():
    for sl in SLICES:
        for it in ITEMS[sl][:20]:
            lab = LABELS[sl][it["id"]]
            s = lab["n_sessions"] - 1
            dia = {"instructions": lab["template"], "sessions": [{} for _ in range(s + 1)]}
            live = mc.live_instructions(dia, s)
            assert [list(x) for x in live] == lab["live"]
            want = sorted(json.dumps(r) for r in mc.live_regexes(dia, s, TOPICS))
            assert sorted(json.dumps(r) for r in lab["history_regex"]) == want        # labels == checker expectations
            assert lab["query"] in {BY_ID[p]["eval_query"] for p, _ in live}
            assert lab["required"] == mc.required_families(lab["query"], sorted({str(o) for o, _ in lab["history_regex"]}))
            for p, u in live:                                                    # template -> text: the slot is rendered
                stated = [x for x in lab["units"] if [p, u] in x["conv"]]
                assert stated, (p, u)
                for key, val in bank.slots(BY_ID[p], u).items():
                    if key in ("{affix}", "{module}", "{decorator}"):
                        assert f"'{val}'" in stated[-1]["text"], (val, stated[-1]["text"])
            for k in lab["oracle_units"]:
                assert lab["units"][k]["session"] < s and any(pu in lab["live"] for pu in lab["units"][k]["conv"])


# ---------------------------------------------------------------- baseline == stencil Exp 4C, byte for byte
@pytest.mark.parametrize("split", SPLITS)
def test_baseline_reproduces_4c_policy(split):
    tok = window.tokenizer()
    base, ref = load_surface("/work/stencil.py"), load_surface("/frozen/ref_evicted_1024/stencil.py")
    for it in ITEMS[split][:6]:
        dia, s = window.as_dialogue(it), len(it["sessions"]) - 1
        cut = mc.build_long_prompt(dia, s, it["query"], tok, "")["cut_chars"]
        evicted = mc.evicted_mentor_sentences(dia, s, cut=cut)
        win = window.Window(it)
        assert [r["text"] for r in win.sentences if r["speaker"] == "mentor" and r["evicted_at_base_window"]] == evicted
        assert sorted(win.displaced(0)) == [r["id"] for r in win.sentences if r["evicted_at_base_window"]]
        for surf, budget in ((base, 256), (ref, 1024)):
            plan = win.check(surf.plan([dict(r) for r in win.sentences], it["sessions"][-1], it["query"], win))
            want = mc.render_long_reminder(mc.pack_long(evicted, tok, budget)[0])
            assert win.render(plan.ids, plan.header) == want
            built = win.build(plan)
            assert built["prompt"] == mc.build_long_prompt(dia, s, it["query"], tok, want)["prompt"]
            assert built["prompt_tokens"] <= window.W and built["reminder_tokens"] <= budget
        assert len(win.displaced(600)) >= len(win.displaced(0))


def test_plan_checks():
    it = ITEMS["validation"][0]
    win = window.Window(it)
    n = len(win.sentences)
    P = window.Plan
    for bad in (P(ids=[n]), P(ids=[0, 0]), P(ids=[-1]), P(header=len(window.HEADERS)), P(placement="middle"),
                P(budget=1025), P(ids=list(range(min(n, 200))), budget=20), "text"):
        with pytest.raises(window.PlanError):
            win.check(bad)
    ok = win.check({"ids": [0, 1], "header": 1, "placement": "before_thread", "budget": 1024})
    b = win.build(ok)
    assert b["prompt"].startswith("<|im_start|>user\n- " + win.sentences[0]["text"]) and b["prompt_tokens"] <= window.W


# ---------------------------------------------------------------- scorer and evaluator (hand-written outputs)
sys.path.insert(0, "/eval")


EV = load_surface("/eval/evaluate.py")
CS = EV.load_checker(f"{D}/validation/private")


def fr(text, regexes, query):
    fams = sorted({o for o, _ in regexes})
    return mc.score_generation(text, regexes, CS, mc.required_families(query, fams), mc.required_structure(query))["fraction_required"]


def test_scorer_required_parent_missing_and_optional_ignored():
    rx = [["function", "^a_.*"], ["variable", ".*_x$"], ["method", "^md_.*"]]
    q = "function that implements merge sort"                                # function-side: method is optional
    good = "```python\ndef a_sort(arr):\n    out_x = sorted(arr)\n    return out_x\n```"
    assert fr(good, rx, q) == 1.0
    cls_only = "```python\nclass Sorter:\n    def md_run(self, arr):\n        out_x = sorted(arr)\n        return out_x\n```"
    assert fr(cls_only, rx, q) == 0.0                                        # required function missing -> 0
    extra = good + "\n```python\n```" + "\n"  # second block ignored by the checker (first block only)
    assert fr(extra, rx, q) == 1.0
    with_bad_method = "```python\ndef a_sort(arr):\n    out_x = sorted(arr)\n    return out_x\n\nclass K:\n    def run(self):\n        return 1\n```"
    assert fr(with_bad_method, rx, q) == 1.0                                 # optional (method) family ignored
    half = "```python\ndef sort(arr):\n    out_x = sorted(arr)\n    return out_x\n```"
    assert fr(half, rx, q) == 0.5
    assert fr("I cannot help with that.", rx, q) == 0.0
    imp = [["import", ["secrets", True]], ["function", "^a_.*"]]
    assert fr("```python\nimport secrets\ndef a_f(x):\n    return x\n```", imp, q) == 1.0
    assert fr("```python\ndef a_f(x):\n    return x\n```", imp, q) == 0.5


def _mini_run(tmp_path, mutate=None, surface_src=None, n=2):
    """A 2-item validation subset, the baseline plans, canned generations; returns the evaluator's metrics.json."""
    data = tmp_path / "data"
    (data / "public").mkdir(parents=True)
    (data / "private").mkdir()
    items = ITEMS["validation"][:n]
    (data / "public" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (data / "private" / "labels.jsonl").write_text("".join(json.dumps(LABELS["validation"][i["id"]]) + "\n" for i in items))
    os.symlink(f"{D}/validation/private/vendor", data / "private" / "vendor")
    work = tmp_path / "work"
    work.mkdir()
    (work / "stencil.py").write_text(surface_src or open("/work/stencil.py").read())
    surf = load_surface(str(work / "stencil.py"))
    out = tmp_path / "out"
    out.mkdir()
    code = "```python\nimport secrets\n\ndef f(x):\n    return x\n```"
    with open(out / "outputs.jsonl", "w") as f:
        for it in items:
            win = window.Window(it)
            plan = win.check(surf.plan([dict(r) for r in win.sentences], it["sessions"][-1], it["query"], win))
            b = win.build(plan)
            row = {"id": it["id"], "kind": "main", "W": window.W, "plan": plan.__dict__, "prompt": b["prompt"],
                   "text": code, "finish_reason": "stop", "timed_out": False, "completion_tokens": 20}
            if mutate:
                mutate(row, win)
            f.write(json.dumps(row) + "\n")
    res = tmp_path / "metrics.json"
    subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(out), "--out", str(res), "--max-new", "1024",
                    "--data", str(data), "--work", str(work)], check=True, capture_output=True, text=True)
    return json.loads(res.read_text())


def test_evaluator_valid_on_baseline(tmp_path):
    m = _mini_run(tmp_path)
    assert m["valid"] and len(m["items"]) == 2 and 0 <= m["primary"] <= 1
    assert m["metrics"]["reminder_tokens_mean"] <= 256


def test_evaluator_rejects_paraphrased_reminder_line(tmp_path):
    def paraphrase(row, win):
        line = row["prompt"].split("Earlier instructions still in force:\n- ", 1)[1].split("\n", 1)[0]
        row["prompt"] = row["prompt"].replace("- " + line, "- Always start function names with 'a_'.", 1)
    m = _mini_run(tmp_path, paraphrase)
    assert not m["valid"] and "differs" in m["message"]


def test_evaluator_rejects_overlong_prompt_and_cap(tmp_path):
    def longer(row, win):
        row["prompt"] = row["prompt"].replace("Based on information provided", "Based on " + "all the " * 50 + "information provided")
    assert not _mini_run(tmp_path / "a", longer)["valid"]
    def over_cap(row, win):
        row["completion_tokens"] = 1025
    m = _mini_run(tmp_path / "b", over_cap)
    assert not m["valid"] and "cap" in m["message"]
    def plan_error(row, win):
        row["plan_error"] = "PlanError: duplicate sentence id"
    assert not _mini_run(tmp_path / "c", plan_error)["valid"]
    def bad_plan(row, win):
        row["plan"]["budget"] = 2048
    assert not _mini_run(tmp_path / "d", bad_plan)["valid"]


def test_evaluator_rejects_topics_ngrams_in_surface(tmp_path):
    src = open("/work/stencil.py").read() + "\nKNOWN = ['always start function names with']\n"
    m = _mini_run(tmp_path, surface_src=src)
    assert not m["valid"] and "topics.json" in m["message"]
    for p in ("/work/stencil.py", "/frozen/ref_off/stencil.py", "/frozen/ref_evicted_1024/stencil.py"):
        assert EV.surface_violation(os.path.dirname(p), TOPICS) is None, p


def test_real_dialogues_descriptive_only():
    real = REPORT["real"]
    assert real["counts"]["free"] == 28 and real["counts"]["short_long_overlap"] == 0
    if real["used"]:
        pub = rows(f"{D}/holdout/public/real.jsonl")
        assert len(pub) == 28 and not os.path.exists(f"{D}/validation/public/real.jsonl")
        assert set(pub[0]) == {"id", "mentor", "mentee", "sessions", "query"}


def test_surface_runs_only_in_the_planner_process():
    """The trusted harness never imports the surface (it cannot patch generation or the recorded outputs)."""
    h = open("/frozen/harness.py").read()
    assert "/work" not in h and "import stencil" not in h and "spec_from_file_location" not in h
    assert '"/frozen/planner.py"' in h and "/work/stencil.py" in open("/frozen/planner.py").read()
    for src in ("/frozen/harness.py", "/frozen/planner.py", "/frozen/window.py"):
        s = open(src).read()
        assert "private" not in s and "/eval" not in s, src
