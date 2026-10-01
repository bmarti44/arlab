"""Deterministic CPU checks for plastic-agent (run by `arlab check` as root, no GPU, no network; data at /data,
pack at /pack, surface at /work). Model tests use a tiny random-init Qwen3 with the real tokenizer's vocabulary."""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "/pack/frozen/prepare")
sys.path.insert(0, "/frozen")
sys.path.insert(0, "/eval")
import fauxos  # noqa: E402  (the PREPARE-only generator + simulator)
import lora  # noqa: E402
from common import MODEL_DIR, arm_of, chat_prefix, chat_user, render_transcript, system_text  # noqa: E402
from engine import Budget, BudgetExceeded, Engine, Gen  # noqa: E402
from scoring import gsm_answer, gsm_score  # noqa: E402

D = "/data"
SPLITS = ("validation", "holdout")
ENV = {**os.environ, "PYTHONPATH": "/frozen:/arlab_lib", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "PYTHONDONTWRITEBYTECODE": "1"}


def js(p):
    return json.load(open(p))


def pub_worlds(root) -> list[dict]:
    """A split's public worlds in harness order (public/order.json + one file per world)."""
    return [js(f"{root}/public/worlds/{w}.json") for w in js(f"{root}/public/order.json")]


def pub_twins(root) -> list[dict]:
    """The twins (control worlds) of a split's worlds, in the same order."""
    return [js(f"{root}/public/twins/{w}.json") for w in js(f"{root}/public/order.json")]


# ================================================================ generator: deterministic, well-formed, disjoint
def test_generator_deterministic():
    a, b = fauxos.gen_world(123), fauxos.gen_world(123)
    assert a == b and a != fauxos.gen_world(124)
    ea, enda = fauxos.explore(a, 120)
    eb, endb = fauxos.explore(b, 120)
    assert ea == eb and enda == endb
    assert fauxos.make_tasks(a, enda, ea, 20, "x") == fauxos.make_tasks(b, endb, eb, 20, "x")


def test_world_shape():
    for seed in range(40, 60):
        w = fauxos.gen_world(seed)
        names = [t["name"] for t in w["tools"]]
        ops = [t["op"] for t in w["tools"]]
        assert len(names) == fauxos.N_OPS + 1 and len(set(names)) == len(names)
        assert ops[0] == "inspect" and len(set(ops)) == len(ops) and set(ops[1:]) <= set(fauxos.TASK_OPS)
        for t in w["tools"]:
            assert sorted(t["perm"]) == list(range(len(fauxos.OPS[t["op"]][0])))
        assert 30 <= len(w["init"]["objs"]) <= 60
        assert len(set(w["errors"].values())) == 3 and 2 <= w["C"] <= 9


def test_prepared_data_matches_generator_and_splits_are_disjoint():
    info, sp = js(f"{D}/info.json"), js(f"{D}/splits.json")
    seeds, toolsets = {}, {}
    for split in SPLITS:
        pub, priv = pub_worlds(f"{D}/{split}"), js(f"{D}/{split}/private/worlds.json")
        assert [w["id"] for w in pub] == list(priv) and sp[split] == sum(len(p["tasks"]) for p in priv.values())
        for w in pub:
            p = priv[w["id"]]
            seed = info["worlds"][w["id"]]["seed"]
            assert p["spec"] == json.loads(json.dumps(fauxos.regenerate(p["spec"]))), "data was prepared with another generator"
            seeds[w["id"]] = seed
            toolsets[w["id"]] = frozenset(w["tools"])
        assert open(f"{D}/{split}/private/fauxos.py").read() == open("/pack/frozen/prepare/fauxos.py").read()
    g = {s: js(f"{D}/{s}/private/guard.json") for s in SPLITS}
    gseeds = {info["guard"][s]["seed"] for s in SPLITS}
    assert len(gseeds) == 2 and not gseeds & set(seeds.values()) and g["validation"]["id"] != g["holdout"]["id"]
    assert g["validation"]["spec"] == json.loads(json.dumps(fauxos.gen_world(info["guard"]["validation"]["seed"])))
    assert len(set(seeds.values())) == len(seeds), "world seeds must be disjoint across and within splits"
    assert {s // 100 for k, s in seeds.items() if k[0] == "v"}.isdisjoint({s // 100 for k, s in seeds.items() if k[0] == "h"})
    assert len(set(toolsets.values()) | {frozenset(x["tools"]) for x in g.values()}) == len(toolsets) + 2
    ids = [t["id"] for s in SPLITS for p in js(f"{D}/{s}/private/worlds.json").values() for t in p["tasks"]]
    assert len(ids) == len(set(ids))
    # the forgetting batteries of validation and holdout are disjoint: guard world (above), GSM8K items, text rows
    gv, gh = ({(x["id"], x["q"]) for x in js(f"{D}/{s}/private/gsm8k.json")} for s in SPLITS)
    assert gv and gh and not gv & gh and not {q for _, q in gv} & {q for _, q in gh}
    rows = {s: {r.tobytes() for r in np.load(f"{D}/{s}/private/text.npy")} for s in SPLITS}
    replay = {r.tobytes() for r in np.load(f"{D}/train/replay.npy")}
    assert not rows["validation"] & rows["holdout"] and not (rows["validation"] | rows["holdout"]) & replay


# ================================================================ v2 campaign worlds ("fam" style)
def test_v2_splits_balanced_disjoint_wording_and_op_types():
    """Validation = bank A + dev-only op types; holdout = bank B + reserved op types; every op exactly FAM_TASKS_PER_OP
    tasks per world; reference programs score 1.0; banks A and B share no template string; twins share the family."""
    import wording
    assert not wording.all_strings("A") & wording.all_strings("B")
    want = {"validation": ("A", set(fauxos.DEV_OPS)), "holdout": ("B", set(fauxos.RESERVED_OPS))}
    for split, (bank, extra) in want.items():
        priv, tw = js(f"{D}/{split}/private/worlds.json"), js(f"{D}/{split}/private/twins.json")
        assert open(f"{D}/{split}/private/wording.py").read() == open("/pack/frozen/prepare/wording.py").read()
        for wid, p in priv.items():
            spec = p["spec"]
            ops = {t["op"] for t in spec["tools"]} - {"inspect"}
            assert spec["style"] == "fam" and spec["family"]["bank"] == bank and extra <= ops and len(ops) == 10
            other = set(fauxos.DEV_OPS) | set(fauxos.RESERVED_OPS)
            assert not (ops & other) - extra, (split, ops)
            counts = {}
            for t in p["tasks"]:
                counts[t["template"]] = counts.get(t["template"], 0) + 1
                assert fauxos.score(t, fauxos.run_program(spec, p["end"], "\n".join(t["reference"]))) == 1.0
                assert t["stratum"] == ("familiar" if t["template"] in fauxos.TASK_OPS else
                                        "dev" if split == "validation" else "reserved")
            assert set(counts) == ops and set(counts.values()) == {fauxos.FAM_TASKS_PER_OP}
            pub = js(f"{D}/{split}/public/worlds/{wid}.json")
            logged = fauxos._fam_logged_values(spec, pub["transcript"], p["end"])   # (also: the replay reaches `end`)
            for t in p["tasks"]:    # no question is answered by a logged call whose logged value is still right
                assert t["answer"] is None or logged.get(t["reference"][0], object()) != t["answer"], (wid, t["goal"])
            assert tw[wid]["spec"]["family"] == spec["family"]
            assert all(not any(v in t["name"] for vs in fauxos.TRUE_VERBS.values() for v in vs if len(v) > 3)
                       for t in spec["tools"])                     # names carry no English verb
    hold = js(f"{D}/holdout/private/worlds.json")
    val = js(f"{D}/validation/private/worlds.json")
    goals = lambda ws: {t["goal"] for p in ws.values() for t in p["tasks"]}  # noqa: E731
    assert not goals(hold) & goals(val)


def test_v2_typed_values_do_not_depend_on_wording():
    """A fam world's query answers come from the semantics, not from parsing text: re-rendering the same world with
    another family leaves every reference value and final state unchanged."""
    import wording, random as _r
    spec = js(f"{D}/holdout/private/worlds.json")
    wid, p = next(iter(spec.items()))
    other = {**p["spec"], "family": wording.make_family(_r.Random(5), "A")}
    for t in p["tasks"]:
        a = fauxos.run_program(p["spec"], p["end"], "\n".join(t["reference"]))
        b = fauxos.run_program(other, p["end"], "\n".join(t["reference"]))
        assert a["values"] == b["values"] and a["state"] == b["state"] and a["outputs"] != b["outputs"] or not a["outputs"]


# ================================================================ novel split (v2 robustness check)
STD_OBS = ("archived #", "deleted #", "restored #", "locked #", "unlocked #", "moved #", "tagged #", "copied #",
           "swapped #", "count: ", "total: ", "checksum: ", "heaviest: ", "lightest: ", "newest: ", "oldest: ", "kind=")


def test_novel_split_new_ops_new_wording_and_holdout_battery():
    """Every novel world has all NOVEL_OPS (absent from every standard world), its reference programs score 1.0, none
    of its observations uses a standard-world output phrase, and its forgetting battery is the holdout's."""
    priv, pubs = js(f"{D}/novel/private/worlds.json"), pub_worlds(f"{D}/novel")
    assert [w["id"] for w in pubs] == list(priv) and len(priv) >= 2
    for w in js(f"{D}/validation/private/worlds.json").values():      # (v2 holdout reserves them by design)
        assert not {t["op"] for t in w["spec"]["tools"]} & set(fauxos.NOVEL_OPS)
    for w in pubs:
        p = priv[w["id"]]
        assert p["spec"]["style"] == "novel" and set(fauxos.NOVEL_OPS) <= {t["op"] for t in p["spec"]["tools"]}
        for e in w["transcript"]:
            assert not any(x in e["obs"] for x in STD_OBS), e
        for t in p["tasks"]:
            assert fauxos.score(t, fauxos.run_program(p["spec"], p["end"], "\n".join(t["reference"]))) == 1.0
    assert any(t["template"] in fauxos.NOVEL_OPS for p in priv.values() for t in p["tasks"])
    for f in ("guard.json", "gsm8k.json"):
        assert js(f"{D}/novel/private/{f}") == js(f"{D}/holdout/private/{f}")
    assert (np.load(f"{D}/novel/private/text.npy") == np.load(f"{D}/holdout/private/text.npy")).all()
    assert js(f"{D}/novel/private/twins.json").keys() == priv.keys()


# ================================================================ twin (control) worlds
def test_twin_pass_is_indistinguishable_by_seed_or_scratch_name():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = ["/frozen/run/harness.py", os.path.join(here, "..", "frozen", "run", "harness.py")]
    src = open(next(p for p in cand if os.path.exists(p))).read()
    assert 'mkdtemp(prefix="pa-")' in src and "+ 500" not in src
    assert 'adapt_world(f"{wid}x", twin, a.seed * 1000 + wi)' in src


def test_twins_are_deterministic_and_disjoint_from_every_other_world():
    """(c) Each twin is regenerated exactly from its own seed and its target; twin seeds are disjoint from the target,
    guard and other twin seeds (and seed ranges); no twin transcript equals any other transcript."""
    info = js(f"{D}/info.json")
    target_seeds = {v["seed"] for v in info["worlds"].values()} | {v["seed"] for v in info["guard"].values()}
    twin_seeds, transcripts = [], []
    for split in SPLITS:
        priv, tw = js(f"{D}/{split}/private/worlds.json"), js(f"{D}/{split}/private/twins.json")
        assert list(tw) == js(f"{D}/{split}/public/order.json")
        for wid, t in tw.items():
            assert t["id"] == f"{wid}x" and info["twins"][t["id"]] == {"seed": t["seed"], "twin_of": wid}
            again = fauxos.gen_twin(fauxos.regenerate(priv[wid]["spec"]), t["seed"])
            assert t["spec"] == json.loads(json.dumps(again)) and again == fauxos.gen_twin(priv[wid]["spec"], t["seed"])
            ev, end = fauxos.explore(again, len(js(f"{D}/{split}/public/twins/{wid}.json")["transcript"]))
            assert ev == js(f"{D}/{split}/public/twins/{wid}.json")["transcript"] and json.loads(json.dumps(end)) == t["end"]
            twin_seeds.append(t["seed"])
        transcripts += [json.dumps(w["transcript"]) for w in pub_worlds(f"{D}/{split}") + pub_twins(f"{D}/{split}")]
    assert len(set(twin_seeds)) == len(twin_seeds) and not set(twin_seeds) & target_seeds
    assert not {s // 100 for s in twin_seeds} & {s // 100 for s in target_seeds}
    assert len(set(transcripts)) == len(transcripts)


def test_twins_share_names_vocabulary_and_templates_but_not_semantics():
    """(a) Twin and target have exactly the same tool names, operation set (so the same task templates and wording
    apply), kinds, places, tags and units, and the twin's transcript uses only those names; but every name does
    something else (a derangement), with fresh argument orders, variants and objects."""
    n_names = n_diff = 0
    for split in SPLITS:
        priv, tw = js(f"{D}/{split}/private/worlds.json"), js(f"{D}/{split}/private/twins.json")
        for w, t in zip(pub_worlds(f"{D}/{split}"), pub_twins(f"{D}/{split}")):
            a, b = priv[w["id"]]["spec"], tw[w["id"]]["spec"]
            assert t["tools"] == w["tools"] == sorted(x["name"] for x in b["tools"])
            for k in ("kinds", "places", "tags", "unit", "base_unit"):
                assert a[k] == b[k], k
            assert sorted(x["op"] for x in a["tools"]) == sorted(x["op"] for x in b["tools"])     # same templates
            assert {x["template"] for x in priv[w["id"]]["tasks"]} <= {x["op"] for x in b["tools"]}
            assert {e["call"].split("(")[0] for e in t["transcript"]} == set(w["tools"])
            vocab = set(a["kinds"]) | set(a["places"]) | set(a["tags"])
            for e in t["transcript"]:
                (_, args), = fauxos.parse_program(e["call"])
                assert all(isinstance(x, int) or x in vocab or x == "x" for x in args), e
            ta, tb = fauxos.tool_map(a), fauxos.tool_map(b)
            n_names += len(ta)
            n_diff += sum(ta[n]["op"] != tb[n]["op"] for n in ta)
            assert b["init"]["objs"] != a["init"]["objs"]
    assert n_diff == n_names          # every name -> behaviour mapping differs (well above a "nontrivial fraction")


VERB_OP = {v: op for op, vs in fauxos.TRUE_VERBS.items() for v in vs}
VERB_OP.update({v: op for op, vs in fauxos.FALSE_VERBS.items() for v in vs})     # misleading verbs are likelier (35%)
OBS_OP = [(r"#\d+ kind=", "inspect"), (r"count: ", "count"), (r"total: ", "weigh"), (r"(heaviest|lightest): ", "extreme"),
          (r"(newest|oldest): ", "newest"), (r"tag \w+: ", "find_tag"), (r"checksum: ", "checksum"), (r"moved #", "move"),
          (r"archived #", "archive"), (r"deleted #", "delete"), (r"restored #", "restore"), (r"locked #", "lock"),
          (r"unlocked #", "unlock"), (r"tagged #", "retag"), (r"copied #", "clone"), (r"swapped #", "swap")]


def names_only_policy(transcript, tools):
    """The review's counterexample: fixed verb-family priors over tool names, the transcript ignored."""
    out = {}
    for n in tools:
        op = VERB_OP.get(n.split("_", 1)[1])
        if op:
            out.setdefault(op, (n, None))
    return out


def transcript_oracle_policy(transcript, tools):
    """Reads each tool's operation and argument order from what the transcript's observations show."""
    import re
    kind = lambda x: "id" if isinstance(x, int) else "kind" if x.isupper() else "str"  # noqa: E731
    out = {}
    for e in transcript:
        op = next((o for rx, o in OBS_OP if re.match(rx, e["obs"])), None)
        if op is None or op in out:
            continue
        (name, args), = fauxos.parse_program(e["call"])
        canon = [t if t in ("id", "kind") else "str" for _, t in fauxos.OPS[op][0]]
        free, perm = list(range(len(canon))), []
        for x in args:
            ci = next(c for c in free if canon[c] == kind(x))
            free.remove(ci)
            perm.append(ci)
        out[op] = (name, perm)
    return out


def _policy_success(pol, spec, end, tasks) -> float:
    """Mean task success of a {op: (tool name, argument order or None = canonical)} policy on the target world."""
    got = []
    for t in tasks:
        (name, args), = fauxos.parse_program(t["reference"][0])
        canon = [None] * len(args)
        for pos, ci in enumerate(fauxos.tool_map(spec)[name]["perm"]):
            canon[ci] = args[pos]
        if t["template"] not in pol:
            got.append(0.0)
            continue
        n, perm = pol[t["template"]]
        prog = fauxos.fmt_call(n, canon if perm is None else [canon[ci] for ci in perm])
        got.append(fauxos.score(t, fauxos.run_program(spec, end, prog)))
    return float(np.mean(got))


def test_twin_control_rejects_name_priors_and_credits_transcript_learning():
    """(b) In memory, on the prepared data and the real simulator / scorer: the names-only policy scores the same with
    the target's and the twin's transcript -> world_specific_share 0 (fails min 0.5), although the old donor-world
    control would have credited it (share 1). A policy that reads the TARGET's transcript scores ~1 on the target and
    ~0 with the twin's transcript -> share ~1. none = no adaptation (0 for these policies)."""
    from scoring import world_specific_share
    for split, (priv, order, pubs, twins) in _std_world_sets().items():
        res = {}
        for name, pol in (("names", names_only_policy), ("oracle", transcript_oracle_policy)):
            own, twin, donor = [], [], []
            for i, (w, t) in enumerate(zip(pubs, twins)):
                p = priv[w["id"]]
                d = pubs[(i + 1) % len(pubs)]
                n = len(p["tasks"])
                own += [_policy_success(pol(w["transcript"], w["tools"]), p["spec"], p["end"], p["tasks"])] * n
                twin += [_policy_success(pol(t["transcript"], t["tools"]), p["spec"], p["end"], p["tasks"])] * n
                donor += [_policy_success(pol(d["transcript"], d["tools"]), p["spec"], p["end"], p["tasks"])] * n
            res[name] = s, m, md = float(np.mean(own)), float(np.mean(twin)), float(np.mean(donor))
        s, m, md = res["names"]
        assert s >= 0.2 and m == s and world_specific_share(s, m, 0.0) == 0.0 < 0.5, (split, res)
        assert md < 0.05 and world_specific_share(s, md, 0.0) > 0.9                  # the flaw of the donor control
        s, m, _ = res["oracle"]
        assert s >= 0.95 and m <= 0.1 and world_specific_share(s, m, 0.0) >= 0.9, (split, res)
        assert len(order) == len(twins)
    hits = n = 0                          # v2 fam worlds: names are drawn independently of semantics
    for split in SPLITS:
        for p in js(f"{D}/{split}/private/worlds.json").values():
            for t in p["spec"]["tools"]:
                n += 1
                hits += VERB_OP.get(t["name"].split("_", 1)[1]) == t["op"]
    assert hits <= max(2, n // 50), (hits, n)


def _std_world_sets() -> dict:
    """v1-style ("std") worlds built in memory (seeds of the v1 holdout): the twin-control mechanics test needs verb
    names and std wording, which the v2 data splits no longer have."""
    out = {}
    pubs, twins, priv = [], [], {}
    for i in range(4):
        spec = fauxos.gen_world(26_092_800 + i)
        ev, end = fauxos.explore(spec, 120)
        wid = f"s{i}"
        tasks = fauxos.make_tasks(spec, end, ev, 40, wid)
        tw = fauxos.gen_twin(spec, 26_093_100 + i)
        tev, _ = fauxos.explore(tw, 120)
        names = sorted(t["name"] for t in spec["tools"])
        pubs.append({"id": wid, "tools": names, "transcript": ev})
        twins.append({"id": f"{wid}x", "tools": names, "transcript": tev})
        priv[wid] = {"spec": spec, "end": end, "tasks": tasks}
    out["std"] = (priv, [p["id"] for p in pubs], pubs, twins)
    return out


# ================================================================ transcripts reveal only observable facts
def test_transcript_is_a_faithful_replay():
    """Every observation is exactly what the simulator prints for that call, in order, from the initial state; the
    task start state is where the replay ends. Nothing else is in the transcript."""
    for split in SPLITS:
        pub, priv = pub_worlds(f"{D}/{split}"), js(f"{D}/{split}/private/worlds.json")
        tw = js(f"{D}/{split}/private/twins.json")
        pub = pub + pub_twins(f"{D}/{split}")          # the twins' transcripts come from the same explorer
        priv = {**priv, **{t["id"]: t for t in tw.values()}}
        for w in pub:
            spec = priv[w["id"]]["spec"]
            st = copy.deepcopy(spec["init"])
            for e in w["transcript"]:
                assert set(e) == {"call", "obs"}
                (name, args), = fauxos.parse_program(e["call"])
                obs, _ = fauxos.call(spec, st, name, args)
                assert obs == e["obs"], (w["id"], e)
            assert st == priv[w["id"]]["end"]
            assert w["tools"] == sorted(t["name"] for t in spec["tools"])
            obs_all = {e["obs"] for e in w["transcript"]}
            assert all(fauxos.error_text(spec, c) in obs_all for c in spec["errors"].values()), \
                "every error code must be observed at least once"
            called = {e["call"].split("(")[0] for e in w["transcript"]}
            assert called == set(w["tools"]), "every tool must appear in the transcript"


PRIVATE_MARKERS = ('"op"', '"perm"', '"var"', '"verbose"', "skip_locked", "with_locked", "gold_state", '"reference"',
                   '"spec"', "find_tag", "badarg", "Goal:")


def test_public_and_train_data_reveal_no_hidden_state_or_labels():
    for split in SPLITS:
        order = js(f"{D}/{split}/public/order.json")
        raw = "".join(open(f"{D}/{split}/public/{d}/{w}.json").read() for w in order for d in ("worlds", "twins"))
        for m in PRIVATE_MARKERS:
            assert m not in raw, f"{m} in public data"
        pub, priv = pub_worlds(f"{D}/{split}"), js(f"{D}/{split}/private/worlds.json")
        assert all(set(w) == {"id", "tools", "transcript"} for w in pub + pub_twins(f"{D}/{split}"))
        assert sorted(os.listdir(f"{D}/{split}/public")) == ["order.json", "twins", "worlds"]
        for d in ("worlds", "twins"):
            assert sorted(os.listdir(f"{D}/{split}/public/{d}")) == sorted(f"{w}.json" for w in order)
        for w in pub:
            text = render_transcript(w["transcript"])
            for t in priv[w["id"]]["tasks"]:
                assert t["goal"] not in raw
                if t["answer"] is None:      # a mutation task's full reference program never appears in the transcript
                    assert not all(f"> {line}\n" in text for line in t["reference"]), t
            assert str(priv[w["id"]]["spec"]["seed"]) not in raw
        for t in js(f"{D}/{split}/private/twins.json").values():
            assert str(t["seed"]) not in raw
    assert sorted(os.listdir(f"{D}/train")) == ["replay.npy"]


def test_known_limits_are_recorded():
    """v2: validation and holdout share only the familiar op types; their extra op types are disjoint (dev vs
    reserved). IDEA.md must state the v2 contract, and record that campaigns mount the whole HF cache (the surface
    sandbox is what keeps it unreadable)."""
    tpl = {s: {t["template"] for p in js(f"{D}/{s}/private/worlds.json").values() for t in p["tasks"]} for s in SPLITS}
    assert tpl["holdout"] <= set(fauxos.TEMPLATES) and tpl["validation"] <= set(fauxos.TEMPLATES)
    assert not (tpl["validation"] - set(fauxos.TASK_OPS)) & (tpl["holdout"] - set(fauxos.TASK_OPS))
    assert set(fauxos.RESERVED_OPS) <= tpl["holdout"] and set(fauxos.DEV_OPS) <= tpl["validation"]
    idea = " ".join(open("/pack/IDEA.md").read().split())
    assert "disjoint wording families" in idea and "reserved operation types" in idea
    assert "mounts the whole HF cache read-only at `/hf` in RUN" in idea and "no longer depends on them being unreadable" in idea


def _code(path) -> str:
    """Source without comments and docstrings (the checks below are about what the code does, not what it says)."""
    import ast
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.unparse(tree)


def test_run_side_code_cannot_reach_the_simulator_or_private_data():
    run_files = [f for f in os.listdir("/frozen") if f.endswith(".py")]
    assert "fauxos.py" not in run_files and "prepare.py" not in run_files
    for f in run_files + [f"ref_{r}/adapt.py" for r in ("none", "icl", "placebo")]:
        src = _code(f"/frozen/{f}")
        for bad in ("fauxos", "private", "holdout", "/eval", "/prepare", "gold"):
            assert bad not in src, f"{bad} in frozen/run/{f}"
    src = _code("/frozen/harness.py")
    # the supervisor opens exactly: the world order, the replay rows, and ONE world file per world (the donor's)
    # (adapt_world opens exactly the one file it is handed: the world's or its twin's)
    assert src.count("open(f'{a.data}") == 1 and "json.load(open(f'{a.data}/public/order.json'))" in src
    assert src.count("json.load(open(") == 2 and "json.load(open(src))" in src and "np.load(f'{a.data}/train/replay.npy')" in src
    assert "(f'{a.data}/public/worlds/{wid}.json', f'{a.data}/public/twins/{wid}.json')" in src
    assert "import adapt" not in src and "importlib" not in src and "/work" not in src.replace("default='/work'", "")
    # the sandboxed child reads no file: the world arrives over its stdin
    child = _code("/frozen/child.py") + _code("/frozen/surface_api.py")
    assert "a.data" not in child and "np.load" not in child and "/data" not in child and "/hf" not in child


def test_run_side_code_references_no_dataset_under_hf():
    """A campaign mounts the whole HF cache read-only into RUN; frozen RUN code references only the model snapshot.
    The GSM8K / OASST2 items the evaluator needs are copied into private/ at PREPARE."""
    files = [f for f in os.listdir("/frozen") if f.endswith(".py")] + [f"ref_{r}/adapt.py" for r in ("none", "icl", "placebo")]
    for f in files:
        src = open(f"/frozen/{f}").read()
        for bad in ("datasets--", "parquet", "jsonl", "/hf/hub/datasets"):
            assert bad not in src, (bad, f)
    common = _code("/frozen/common.py")
    assert "datasets--" not in common and common.count("/hf/") == 1 and "MODEL_DIR = '/hf/hub/models--Qwen--Qwen3-1.7B/" in common
    ev = _code("/eval/evaluate.py")
    assert "/hf" not in ev and "datasets--" not in ev and "private/gsm8k.json" in ev


# ================================================================ simulator and scoring
@pytest.fixture(scope="module")
def world():
    """The first prepared world with a move tool, an int-answer task and a state task (the edge-case tests use them)."""
    for split in SPLITS:
        p = js(f"{D}/{split}/private/worlds.json")
        for w in pub_worlds(f"{D}/{split}"):
            q = p[w["id"]]
            if any(t["op"] == "move" for t in q["spec"]["tools"]) and \
                    any(t["answer"] and t["answer"]["kind"] == "int" for t in q["tasks"]) and any(t["check_state"] for t in q["tasks"]):
                return w, q
    raise AssertionError("no suitable world")


def test_every_reference_program_scores_one_and_trivial_programs_score_zero(world):
    for split in SPLITS:
        for wid, p in js(f"{D}/{split}/private/worlds.json").items():
            spec, end = p["spec"], p["end"]
            for t in p["tasks"]:
                assert fauxos.score(t, fauxos.run_program(spec, end, "\n".join(t["reference"]))) == 1.0, t
                assert fauxos.score(t, fauxos.run_program(spec, end, "answer(-1)")) == 0.0, t
                if t["check_state"]:
                    assert t["gold_state"] != end
                    assert fauxos.score(t, fauxos.run_program(spec, end, f"answer({json.dumps(t['goal'])})")) == 0.0


def test_scorer_edge_cases(world):
    _, p = world
    spec, end = p["spec"], p["end"]
    ans = next(t for t in p["tasks"] if t["answer"] is not None and t["answer"]["kind"] == "int" and not t["check_state"])
    st = next(t for t in p["tasks"] if t["check_state"] and t["answer"] is None)
    ref = "\n".join(st["reference"])
    run = lambda txt: fauxos.score(st, fauxos.run_program(spec, end, txt))  # noqa: E731
    assert run(ref) == 1.0
    assert run(f"```\n{ref}\n```") == 1.0                                   # code fences are ignored
    assert run(f"Here is the call:\n{ref}") == 1.0                             # prose lines are ignored
    assert run(f"> {ref}\nok\n> {ref}") == 1.0                                # log-style: only the first call runs
    assert run(f"{ref};") == 1.0
    inspect = next(t["name"] for t in spec["tools"] if t["op"] == "inspect")
    some_id = next(iter(end["objs"]))
    assert run(f"{inspect}({some_id})\n{ref}") == 0.0                         # the FIRST call is the program
    assert run("I cannot do that.") == 0.0
    assert fauxos.run_program(spec, end, "no call here")["error"]["obs"] == "syntax error: no tool call"
    kw = st["reference"][0].replace("(", "(x=", 1)
    assert fauxos.run_program(spec, end, kw)["error"]["obs"].startswith("syntax error")
    assert fauxos.run_program(spec, end, f"{inspect}({some_id}")["error"]["obs"].startswith("syntax error")
    # forgiving literals: a bare word is a string, a digit string is an int where an int is expected
    name, args = fauxos.parse_program(ref)[0]
    loose = f"{name}(" + ", ".join(f'"{x}"' if isinstance(x, int) else x for x in args) + ")"
    assert fauxos.parse_program(loose) == [(name, [str(x) if isinstance(x, int) else x for x in args])]
    assert run(loose) == 1.0
    v = ans["answer"]["value"]
    sa = lambda txt: fauxos.score(ans, fauxos.run_program(spec, end, txt))  # noqa: E731
    assert sa(f"answer({v})") == 1.0 and sa(f'answer("{v}")') == 1.0 and sa(f'answer(" {v} ")') == 1.0
    assert sa(f"answer({v + 1})") == 0.0 and sa(f'answer("{v} or {v + 1}")') == 0.0   # hedged answers are wrong
    # exact integer grammar: no exponent, no prose, no extra numbers
    for bad in (f'"{v}e999"', f'"{v}e0"', f'"not {v}"', f'"{v}.0"', f'"{v}, {v}"', f'"{v}/1"', f'"~{v}"'):
        assert sa(f"answer({bad})") == 0.0, bad
    assert sa(f'answer("{v}")\nanswer("{v + 1}")') == 1.0 and sa(f'answer("{v + 1}")\nanswer("{v}")') == 0.0  # first call
    assert fauxos.value_of(spec, "answer", ["42e999"], "42e999") is None
    assert fauxos.value_of(spec, "answer", ["not 42"], "not 42") is None
    assert fauxos.value_of(spec, "answer", ["-3"], "-3") == -3 and fauxos.value_of(spec, "answer", ["1, 2"], "1, 2") == [1, 2]
    assert fauxos.score(ans, {"error": None, "values": [True if v == 1 else float(v)], "state": end}) == 0.0
    s = {"answer": {"kind": "set", "value": [101, 202]}, "check_state": False}
    ok = lambda val: fauxos.score(s, {"error": None, "values": [val], "state": end})  # noqa: E731
    assert ok([202, 101]) == 1.0 and ok([101, 202, 202]) == 0.0 and ok(101) == 0.0 and ok(None) == 0.0
    assert fauxos.score(s, {"error": {"line": 1, "obs": "error"}, "values": [[101, 202]], "state": end}) == 0.0


def test_simulator_errors_and_argument_order(world):
    _, p = world
    spec = p["spec"]
    st = copy.deepcopy(p["end"])
    E = spec["errors"]
    move = next(t for t in spec["tools"] if t["op"] == "move")
    locked = next((i for i, o in st["objs"].items() if o["state"] == "active" and o["locked"]), None)
    free = next(i for i, o in st["objs"].items() if o["state"] == "active" and not o["locked"])
    place = next(x for x in spec["places"] if x != st["objs"][free]["place"])

    def args(i, pl):
        canon = [int(i), pl]
        return [canon[ci] for ci in move["perm"]]

    err = lambda k: (fauxos.error_text(spec, E[k]), False)  # noqa: E731
    if locked:
        assert fauxos.call(spec, st, move["name"], args(locked, place)) == err("locked")
    assert fauxos.call(spec, st, move["name"], args(99, place)) == err("missing")
    assert fauxos.call(spec, st, move["name"], args(free, place)[::-1]) == err("badarg")
    assert fauxos.call(spec, st, move["name"], [int(free)]) == err("badarg")
    obs, ok = fauxos.call(spec, st, move["name"], args(free, place))
    assert ok and st["objs"][free]["place"] == place
    assert fauxos.call(spec, st, "no_such_tool", [])[1] is False
    assert fauxos.call(spec, st, "answer", ["x"]) == ("x", True)


def test_gsm8k_scorer():
    assert gsm_answer("so 3+4=7\nAnswer: 7") == 7.0 and gsm_score("Answer: 1,234", 1234) == 1.0
    assert gsm_score("Answer: $18.00", 18) == 1.0 and gsm_score("the answer is 18", 18) == 0.0
    assert gsm_score("Answer: 17\nAnswer: 18", 18) == 1.0 and gsm_score("Answer: 18.5", 18) == 0.0
    assert gsm_score("Answer: 18\nAnswer: 17", 18) == 0.0                     # the last answer line wins
    assert gsm_score("**Answer:** 18.", 18) == 1.0 and gsm_score("Answer: -4", -4) == 1.0
    for bad in ("Answer: 42e9", "Answer: 42/7", "Answer: not 42", "Answer: 42 or 43", "Answer: 1,23", "Answer: 42 apples",
                "Answer: 18\nAnswer: 18e0"):
        assert gsm_answer(bad) is None, bad
    gsm = js(f"{D}/validation/private/gsm8k.json")
    assert len(gsm) == len({x["id"] for x in gsm}) and all(isinstance(x["a"], int) for x in gsm)


# ================================================================ tiny model: engine, LoRA, harness, evaluator
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=40960, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for prm in m.parameters():
            prm.mul_(8.0)                 # sharper random logits: fewer argmax near-ties in the decode test
    path = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def test_engine_matches_uncached_reference_decode(tiny, tok):
    m = _load(tiny)
    E = Engine(m, tok, "cpu")
    pre = E.encode("<|im_start|>system\nhello tools a, b<|im_end|>\n")
    sufs = [E.encode("<|im_start|>user\nGoal: x<|im_end|>\n"), [11], list(range(300, 309)), [5, 6]]

    def ref(s, n):
        ids, out = pre + s, []
        for _ in range(n):
            with torch.no_grad():
                t = int(m(input_ids=torch.tensor([ids])).logits[0, -1].argmax())
            out.append(t)
            ids = ids + [t]
            if t in (151645, 151643):
                break
        return out

    assert E.generate(E.prefix_kv(pre), sufs, max_new=6, batch_size=3) == [ref(s, 6) for s in sufs]
    pairs = [(sufs[0], [20, 21, 22]), ([11], [5, 6])]
    for (p, c), (ix, lp) in zip(pairs, E.topk(E.prefix_kv(pre), pairs, k=5)):
        with torch.no_grad():
            full = torch.log_softmax(m(input_ids=torch.tensor([pre + p + c])).logits[0].float(), -1)
        v, i = full[len(pre) + len(p) - 1:len(pre) + len(p) - 1 + len(c)].topk(5, -1)
        assert torch.equal(i.int(), ix) and (v - lp).abs().max() < 1e-4


def test_lora_attach_merge_and_exact_reset(tiny):
    m = _load(tiny)
    fp = lora.fingerprint(m)
    x = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        ref = m(input_ids=x).logits
    mods = lora.attach(m, {"rank": 4, "alpha": 8, "targets": ["q_proj", "down_proj"]}, seed=0)
    with torch.no_grad():
        assert (m(input_ids=x).logits - ref).abs().max() == 0                 # B = 0 at init: identity
        for mm in mods.values():
            mm.B.normal_(0, 0.05)
        wrapped = m(input_ids=x).logits
    cfg, tens = lora.check_config({"rank": 4, "alpha": 8, "targets": ["q_proj", "down_proj"]}), lora.tensors(mods)
    lora.detach(m)
    assert lora.n_wrapped(m) == 0 and lora.fingerprint(m) == fp
    assert lora.validate(m, cfg, tens) is None
    with lora.Merged(m, cfg, tens), torch.no_grad():
        assert (m(input_ids=x).logits - wrapped).abs().max() < 1e-3
        assert lora.fingerprint(m) != fp
    assert lora.fingerprint(m) == fp                                          # restored bit for bit
    bad = dict(tens)
    k = next(iter(bad))
    bad[k] = bad[k].clone()
    bad[k][0, 0] = float("nan")
    assert "non-finite" in lora.validate(m, cfg, bad)
    with pytest.raises(ValueError):
        lora.check_config({"rank": 65, "alpha": 1, "targets": ["q_proj"]})
    with pytest.raises(ValueError):
        lora.check_config({"rank": 4, "alpha": 1, "targets": ["embed_tokens"]})
    # the fingerprint is a byte hash: a permutation (which a weighted sum misses) and a 1-ulp change are both caught
    w = m.model.layers[0].mlp.down_proj.weight
    with torch.no_grad():
        a0, a1 = w[0, 0].item(), w[0, 1].item()
        w[0, 0], w[0, 1] = a1, a0
        assert lora.fingerprint(m) != fp
        w[0, 0], w[0, 1] = a0, a1
        assert lora.fingerprint(m) == fp
        w[1, 1] = torch.nextafter(w[1, 1], torch.tensor(float("inf")))
        assert lora.fingerprint(m) != fp


PROBE_SURFACE = '''
import json, os, sys
ARM = "icl"          # ignored: the arm is decided by frozen code from the file hash
CALLS = []           # module state: a fresh process per world must start empty


def attempt(fn):
    try:
        fn()
        return "OPEN"
    except Exception as e:
        return type(e).__name__


def adapt(transcript, tool_names, gen, train):
    import socket, subprocess, threading
    import surface_api
    CALLS.append(1)
    rec = {"pid": os.getpid(), "calls": len(CALLS), "tools": tool_names, "transcript": transcript,
           "keys": sorted({k for e in transcript for k in e}), "gen_attrs": sorted(vars(gen)),
           "torch_loaded": "torch" in sys.modules, "seconds_left": gen.seconds_left()}
    rec["reads"] = {p: attempt(lambda p=p: open(p, "rb").read(1)) for p in @READS@}
    rec["listdir"] = {p: attempt(lambda p=p: os.listdir(p)) for p in ("@DATA@", "/hf", "/tmp", "/data")}
    rec["writes"] = {p: attempt(lambda p=p: open(p, "wb").write(b"x"))
                     for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "x"), "/tmp/pa_probe", "@OUT@/x")}
    import tempfile
    tmpd = tempfile.gettempdir()
    rec["scratch"] = {"dir": tmpd, "empty": os.listdir(tmpd) == []}
    open(os.path.join(tmpd, "carry"), "w").write("state")          # writable, but deleted before the next world
    rec["procmem"] = attempt(lambda: open("/proc/%d/mem" % os.getppid(), "rb").read(1))
    rec["environ"] = attempt(lambda: open("/proc/%d/environ" % os.getppid(), "rb").read(1))
    rec["thread"] = attempt(lambda: threading.Thread(target=print).start())
    rec["fork"] = attempt(os.fork)
    rec["subprocess"] = attempt(lambda: subprocess.run(["true"]))

    def net():
        s = socket.socket()
        s.settimeout(1)
        s.connect(("127.0.0.1", 9))
    rec["net"] = attempt(net)
    # every request is metered by the supervisor; extra fields (budget=None) are ignored, local counters are copies
    left0 = gen.tokens_left
    out = gen.generate([gen.task_prompt("Archive item 100.")], max_new_tokens=3, context=False)
    gen._c.call("generate", prompts=["x"], max_new_tokens=3, temperature=0.0, context=False, batch_size=1, budget=None)
    rec["metered"] = left0 - gen.tokens_left
    gen._deadline += 1e6
    gen._c.gen_left = 10 ** 12
    lora_cfg = {"rank": 1, "alpha": 1, "max_len": 64}
    rec["bad"] = {
        "unknown_fn": attempt(lambda: gen._c.call("reset_budget")),
        "prompts_str": attempt(lambda: gen._c.call("generate", prompts="abc", max_new_tokens=3, temperature=0.0,
                                                   context=False, batch_size=1)),
        "max_new_huge": attempt(lambda: gen.generate(["x"], max_new_tokens=10 ** 6)),
        "teacher_forged": attempt(lambda: train([{"prompt": "p", "completion": "c", "teacher": 12345}], lora_cfg)),
        "init_forged": attempt(lambda: train([{"text": "> a(1)\\nok\\n"}], lora_cfg, init=surface_api.AdapterRef(999, {}))),
        "replay_neg": attempt(lambda: train([{"text": "a b c"}], {**lora_cfg, "kl_base": 1.0, "replay_rows": -5})),
    }
    p, c = gen.task_prompt("Lock item 7."), tool_names[0] + "(7)"
    t = gen.teacher([p], [c], k=4, context=True)[0]
    ad = train([{"prompt": p, "completion": c, "teacher": t}, {"text": "> x(1)\\nok\\n", "weight": 0.5}],
               {"rank": 2, "alpha": 4, "epochs": 3, "batch_size": 2, "max_len": 256, "kl_base": 0.1, "replay_rows": 1})
    ad2 = train([{"prompt": p, "completion": c}], {"rank": 2, "alpha": 4, "epochs": 1, "max_len": 256}, init=ad)
    rec.update(out=out, ids=[ad.id, ad2.id], steps=[ad.stats["steps"], ad2.stats["steps"]],
               ad_attrs=sorted(a for a in dir(ad) if not a.startswith("__")))
    ad2.stats["steps"] = 999             # a copy: the supervisor's record of the adapter is unaffected
    print("PROBE " + json.dumps(rec), file=sys.stderr, flush=True)
    return ad2 if len(tool_names) % 2 else ad
'''


def _mini_data(root, n_tasks=3, n_worlds=2):
    """A 2-world copy of the validation split with few tasks and small battery (fast on CPU)."""
    pub = pub_worlds(f"{D}/validation")[:n_worlds]
    priv = js(f"{D}/validation/private/worlds.json")
    for sub in ("public/worlds", "public/twins", "private", "train"):
        os.makedirs(f"{root}/{sub}", exist_ok=True)
    json.dump([w["id"] for w in pub], open(f"{root}/public/order.json", "w"))
    for w in pub:
        json.dump(w, open(f"{root}/public/worlds/{w['id']}.json", "w"))
        shutil.copy(f"{D}/validation/public/twins/{w['id']}.json", f"{root}/public/twins/{w['id']}.json")
    json.dump({w["id"]: {**priv[w["id"]], "tasks": priv[w["id"]]["tasks"][:n_tasks]} for w in pub},
              open(f"{root}/private/worlds.json", "w"))
    g = js(f"{D}/validation/private/guard.json")
    json.dump({**g, "tasks": g["tasks"][:2]}, open(f"{root}/private/guard.json", "w"))
    json.dump(js(f"{D}/validation/private/gsm8k.json")[:2], open(f"{root}/private/gsm8k.json", "w"))
    np.save(f"{root}/private/text.npy", np.load(f"{D}/validation/private/text.npy")[:2, :65])
    shutil.copy(f"{D}/validation/private/fauxos.py", f"{root}/private/fauxos.py")
    shutil.copy(f"{D}/validation/private/wording.py", f"{root}/private/wording.py")
    shutil.copy(f"{D}/train/replay.npy", f"{root}/train/replay.npy")
    return root


def _harness(tiny, data, work, out, secs=120, gen_tokens=10**6, train_tokens=10**6, limit=0, grace=None):
    os.makedirs(out, exist_ok=True)
    return subprocess.run([sys.executable, "/frozen/harness.py", "--out", str(out), "--seed", "1", "--split", "validation",
                           "--adapt-seconds", str(secs), "--gen-tokens", str(gen_tokens), "--train-tokens", str(train_tokens),
                           "--device", "cpu", "--model", tiny, "--data", str(data), "--work", str(work),
                           "--limit-worlds", str(limit)] + (["--grace", str(grace)] if grace is not None else []),
                          capture_output=True, text=True, env=ENV, timeout=600)


def _evaluate(tiny, data, run, tmp, work):
    out = f"{tmp}/metrics.json"
    r = subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", out, "--data", str(data),
                        "--cache", f"{tmp}/cache", "--device", "cpu", "--model", tiny, "--max-new", "8", "--work", str(work)],
                       capture_output=True, text=True, env=ENV, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return js(out)


def _surface(tmp, name, src):
    (tmp / f"w_{name}").mkdir()
    (tmp / f"w_{name}" / "adapt.py").write_text(src)
    return tmp / f"w_{name}"


def _battery_sources() -> list:
    """The raw GSM8K / OASST2 files under /hf that PREPARE builds the private batteries and replay rows from (a campaign
    mounts the whole HF cache at /hf in RUN): the sandboxed surface must not be able to read them."""
    import ast
    tree = ast.parse(open("/pack/frozen/prepare/prepare.py").read())
    return [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", "") in ("GSM8K_FILE", "OASST_FILE")]


@pytest.fixture(scope="module")
def probe_run(tiny, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probe")
    data = _mini_data(tmp / "data")
    reads = [f"{data}/public/order.json", f"{data}/train/replay.npy", f"{MODEL_DIR}/config.json", f"{tiny}/config.json"] + \
        [f"{data}/public/{d}/{w}.json" for w in js(f"{data}/public/order.json") for d in ("worlds", "twins")] + \
        _battery_sources()
    src = PROBE_SURFACE.replace("@READS@", repr(reads)).replace("@DATA@", str(data)).replace("@OUT@", str(tmp / "out"))
    work = _surface(tmp, "probe", src)
    r = _harness(tiny, data, work, tmp / "out")
    assert r.returncode == 0, r.stderr[-3000:]
    log = [json.loads(x[len("PROBE "):]) for x in r.stderr.splitlines() if x.startswith("PROBE ")]
    return tmp, data, work, log, reads


def test_surface_runs_sandboxed_per_world_and_sees_only_its_transcript(probe_run):
    tmp, data, work, log, reads = probe_run
    assert len(log) == 4 and len(reads) == 10 and all(os.path.isfile(p) for p in reads)   # (readable from outside)
    # a candidate run adapts v0, its twin, v1, its twin: four fresh processes, each seeing one transcript
    worlds = [js(f"{data}/public/{d}/{w}.json") for w in ("v0", "v1") for d in ("worlds", "twins")]
    for r, w in zip(log, worlds):
        assert r["calls"] == 1, "a fresh process per world: no module state carries over"
        assert r["transcript"] == w["transcript"] and r["tools"] == w["tools"] and r["keys"] == ["call", "obs"]
        assert r["gen_attrs"] == ["_c", "_deadline", "tool_names", "transcript", "transcript_text"]
        assert not r["torch_loaded"] and 0 < r["seconds_left"] <= 120
        assert all(v != "OPEN" for v in r["reads"].values()), r["reads"]     # other worlds, replay, model, /hf datasets
        assert all(v != "OPEN" for v in r["listdir"].values()), r["listdir"]
        assert all(v != "OPEN" for v in r["writes"].values()), r["writes"]    # /work, /tmp, the run's /out
        assert r["procmem"] != "OPEN" and r["environ"] != "OPEN" and r["net"] != "OPEN"
        assert r["thread"] == "RuntimeError" and r["fork"] == "RuntimeError" and r["subprocess"] != "OPEN"
        assert r["metered"] > 0
        assert r["bad"] == {"unknown_fn": "ValueError", "prompts_str": "TypeError", "max_new_huge": "ValueError",
                            "teacher_forged": "ValueError", "init_forged": "ValueError", "replay_neg": "ValueError"}
        assert r["steps"] == [3, 1] and r["ad_attrs"] == ["id", "stats"]                # no weights in the child
    assert len({r["pid"] for r in log}) == 4 and all(r["scratch"]["empty"] for r in log)
    assert len({r["scratch"]["dir"] for r in log}) == 4 and not any(os.path.exists(r["scratch"]["dir"]) for r in log)
    assert log[0]["tools"] == log[1]["tools"] and log[0]["transcript"] != log[1]["transcript"]   # twin: same names
    assert not os.path.exists(work / "x") and not os.path.exists(tmp / "out" / "x")


def test_supervisor_records_and_saves_only_its_own_copies(probe_run):
    tmp, data, work, log, _ = probe_run
    stats = js(tmp / "out" / "stats.json")
    assert stats["arm"] == "adapter" and [w["id"] for w in stats["worlds"]] == ["v0", "v1"]   # ARM = "icl" ignored
    assert [w["donor"] for w in stats["worlds"]] == ["v0", "v1"] and stats["nan"] is False and stats["violations"] == 0
    assert [w["id"] for w in stats["twins"]] == [w["donor"] for w in stats["twins"]] == ["v0x", "v1x"]
    for w, r in zip(stats["worlds"] + stats["twins"], log[0::2] + log[1::2]):
        assert w["sandbox"]["ok"] is True and w["sandbox"]["landlock_abi"] >= 1 and w["violations"] == []
        assert w["gen_tokens"] > 0 and w["train_tokens"] > 0 and not w["killed"] and w["teacher_calls"] == 1
        assert w["train"]["steps"] == 1 and w["train_calls"] == 2         # 11 tools: ad2 returned; its "999" edit is local
    assert sorted(os.listdir(tmp / "out" / "adapters")) == ["v0", "v0x", "v1", "v1x"]
    (c0, a0), (c1, a1) = (lora.load(tmp / "out" / "adapters" / w) for w in ("v0", "v1"))
    assert any(not torch.equal(a0[k], a1[k]) for k in a0) and c0["rank"] == c1["rank"] == 2
    assert js(tmp / "out" / "budget.json")["adapt_s_max"] > 0


def test_evaluator_end_to_end_and_invalid_outputs(tiny, probe_run):
    from scoring import world_specific_share
    tmp, data, work, _, _ = probe_run
    m = _evaluate(tiny, data, tmp / "out", tmp, work)
    tasks = [t["id"] for p in js(data / "private" / "worlds.json").values() for t in p["tasks"]]
    assert m["valid"] and sorted(m["items"]) == sorted(tasks) and set(m["items"].values()) <= {0.0, 1.0}
    x = m["metrics"]
    assert m["primary"] == x["success"]
    for k in ("success", "twin_success", "world_specific_gain", "world_specific_share", "none_success",
              "battery_drop", "prefill_tokens", "initial_context_tokens", "retry_prompt_tokens", "decode_tokens",
              "inference_tokens", "text_nll_ratio"):
        assert np.isfinite(x[k]), k
    assert x["world_specific_gain"] == x["success"] - x["twin_success"]
    assert x["world_specific_share"] == world_specific_share(x["success"], x["twin_success"], x["none_success"])
    # token accounting: the retry call's context is counted (the random tiny model always needs retries)
    assert x["retry_rate"] > 0 and x["retry_prompt_tokens"] > x["retry_rate"] * x["initial_context_tokens"]
    assert abs(x["prefill_tokens"] - x["initial_context_tokens"] - x["retry_prompt_tokens"]) < 1e-6
    assert abs(x["inference_tokens"] - x["prefill_tokens"] - x["decode_tokens"]) < 1e-6 and x["decode_tokens"] > 0
    assert x["prefill_tokens"] < 1000 and x["n_adapters"] == 2 and x["killed_worlds"] == 0
    assert os.listdir(tmp / "cache")                                   # base results cached (surface-independent)

    def variant(name, edit):
        d = tmp / name
        shutil.copytree(tmp / "out", d)
        edit(d)
        return _evaluate(tiny, data, d, tmp, work)

    def set_stat(**kw):
        def f(d):
            s = js(d / "stats.json")
            s.update(kw)
            json.dump(s, open(d / "stats.json", "w"))
        return f

    def world_stat(i, **kw):
        def f(d):
            s = js(d / "stats.json")
            s["worlds"][i].update(kw)
            json.dump(s, open(d / "stats.json", "w"))
        return f

    assert not variant("nan", set_stat(nan=True))["valid"]
    assert not variant("nan_seen", world_stat(1, nan_seen=True))["valid"]
    assert not variant("thread", world_stat(1, violations=["thread 4242"]))["valid"]
    assert not variant("sandbox", world_stat(0, sandbox={"ok": False}))["valid"]
    assert not variant("arm", set_stat(arm="oracle"))["valid"]
    assert not variant("icl_with_adapters", set_stat(arm="icl"))["valid"]
    assert not variant("claims_placebo", set_stat(arm="placebo"))["valid"]      # the arm is the surface's hash, not a claim
    assert not variant("foreign_world", world_stat(0, donor="v1"))["valid"]     # a candidate adapts on its own world
    assert not variant("missing", lambda d: shutil.rmtree(d / "adapters" / "v1"))["valid"]
    assert not variant("missing_twin", lambda d: shutil.rmtree(d / "adapters" / "v1x"))["valid"]
    assert not variant("no_twins", set_stat(twins=[]))["valid"]                  # the control pass is mandatory

    def twin_stat(i, **kw):
        def f(d):
            s = js(d / "stats.json")
            s["twins"][i].update(kw)
            json.dump(s, open(d / "stats.json", "w"))
        return f
    assert not variant("twin_donor", twin_stat(0, donor="v0"))["valid"]         # the control must see the twin
    assert not variant("twin_thread", twin_stat(1, violations=["thread 1"]))["valid"]
    assert not variant("twin_nan", twin_stat(0, nan_seen=True))["valid"]

    def none_on_twin(d):              # astra r3: an adapter for the target but None for its twin fakes world-specificity
        twin_stat(1, adapter=False)(d)
        shutil.rmtree(d / "adapters" / "v1x")
    r = variant("none_on_twin", none_on_twin)
    assert not r["valid"] and "exactly one of target/twin" in r["message"], r["message"]
    assert not variant("nostats", lambda d: os.remove(d / "stats.json"))["valid"]

    def poison(d):
        from safetensors.torch import load_file, save_file
        t = load_file(d / "adapters" / "v0" / "adapter.safetensors")
        k = next(iter(t))
        t[k][0, 0] = float("inf")
        save_file(t, d / "adapters" / "v0" / "adapter.safetensors")
    assert not variant("inf", poison)["valid"]


def test_world_specific_share():
    from scoring import MIN_GAIN, world_specific_share
    assert MIN_GAIN == 0.025
    assert world_specific_share(0.30, 0.20, 0.10) == pytest.approx(0.5)     # half the gain needs the right world
    assert world_specific_share(0.30, 0.25, 0.10) == pytest.approx(0.25)    # mostly format learning: fails min 0.5
    assert world_specific_share(0.30, 0.10, 0.10) == pytest.approx(1.0)
    assert world_specific_share(0.30, 0.35, 0.10) < 0                        # another world's adapter does better
    assert world_specific_share(0.10, 0.30, 0.10) == 1.0                     # no gain: vacuous (do-nothing baseline)
    assert world_specific_share(0.12, 0.30, 0.10) == 1.0 and world_specific_share(0.05, 0.3, 0.10) == 1.0
    assert world_specific_share(0.126, 0.126, 0.10) == pytest.approx(0.0)    # gain 0.026: attributable, all format


def test_reference_arms(tiny, probe_run):
    tmp, data, _, _, _ = probe_run
    assert _code("/work/adapt.py") == _code("/frozen/ref_placebo/adapt.py"), \
        "ref_placebo must be the baseline surface's code (only the docstring differs)"
    import hashlib
    hashes = {hashlib.sha256(open(f).read().encode()).hexdigest() for f in
              ["/work/adapt.py"] + [f"/frozen/ref_{r}/adapt.py" for r in ("none", "icl", "placebo")]}
    assert len(hashes) == 4, "references must be byte-distinct from each other and from the baseline surface"
    assert arm_of("/work", "/frozen") == "adapter"
    for arm in ("none", "icl", "placebo"):
        assert arm_of(f"/frozen/ref_{arm}", "/frozen") == arm
        # (placebo trains the baseline LoRA: capped at 2 steps of 2 x 1024 positions per world to stay fast on CPU)
        r = _harness(tiny, data, f"/frozen/ref_{arm}", tmp / f"ref_{arm}", train_tokens=4100)
        assert r.returncode == 0, r.stderr[-2000:]
        s = js(tmp / f"ref_{arm}" / "stats.json")
        assert s["arm"] == arm and all(w["adapter"] == (arm == "placebo") for w in s["worlds"]) and s["twins"] == []
    s = js(tmp / "ref_placebo" / "stats.json")
    assert [w["donor"] for w in s["worlds"]] == ["v0x", "v1x"] and s["twins"] == []   # world i adapts on its twin
    icl = _evaluate(tiny, data, tmp / "ref_icl", tmp, "/frozen/ref_icl")
    none = _evaluate(tiny, data, tmp / "ref_none", tmp, "/frozen/ref_none")
    plc = _evaluate(tiny, data, tmp / "ref_placebo", tmp, "/frozen/ref_placebo")
    assert icl["valid"] and none["valid"] and plc["valid"], plc["message"]
    assert plc["primary"] == plc["metrics"]["success"] and "twin_success" not in plc["metrics"]
    # a run evaluated against a different surface than the one that produced it is invalid
    assert not _evaluate(tiny, data, tmp / "ref_placebo", tmp, "/work")["valid"]
    # the ICL arm carries the transcript in every prompt: far over the prefill guard; no-adaptation does not
    assert icl["metrics"]["prefill_tokens"] > 1000 and icl["metrics"]["initial_context_tokens"] > 1000
    assert none["metrics"]["prefill_tokens"] < 1000 and none["metrics"]["battery_drop"] == 0.0
    assert none["primary"] == none["metrics"]["none_success"]


LOOP_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    ad = None
    while True:        # never returns by itself: the deadline must stop it (BudgetExceeded escapes adapt())
        ad = train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64}, init=ad)
'''
HANG_SURFACE = '''
from surface_api import BudgetExceeded


def adapt(transcript, tool_names, gen, train):
    ad = train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64})
    while True:        # ignores the deadline and keeps calling: the supervisor must stop serving and kill it
        try:
            train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64}, init=ad)
        except BudgetExceeded:
            pass
'''
SPIN_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    while True:        # pure CPU, no calls: killed at deadline + grace
        pass
'''
FOREIGN_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    return {"weights": "from somewhere else"}
'''
FORGED_SURFACE = '''
from surface_api import AdapterRef


def adapt(transcript, tool_names, gen, train):
    return AdapterRef(3, {"steps": 1})      # an id the supervisor never issued in this world
'''
BIG_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    return train([{"text": "> a(1)\\nok\\n" * 8}] * 20, {"rank": 1, "alpha": 1, "epochs": 50, "max_len": 64, "batch_size": 4})
'''
GREEDY_SURFACE = '''
from surface_api import BudgetExceeded


def adapt(transcript, tool_names, gen, train):
    gen._c.gen_left = 10 ** 12          # local copies: the supervisor's counters and deadline are unaffected
    gen._deadline += 10 ** 6
    try:
        for _ in range(1000):
            gen.generate(["x" * 50], max_new_tokens=8, context=False)
    except BudgetExceeded:
        pass
    return None
'''
SLOW_IMPORT_SURFACE = '''
import time
time.sleep(3)          # module-level work is charged to every world's adapt budget (a fresh import per world)


def adapt(transcript, tool_names, gen, train):
    return None
'''
THREAD_SURFACE = '''
import ctypes


def adapt(transcript, tool_names, gen, train):
    libc = ctypes.CDLL(None)            # an OS thread that bypasses the Python-level block: seen from outside
    tid = ctypes.c_ulong()
    assert libc.pthread_create(ctypes.byref(tid), None, ctypes.cast(libc.sleep, ctypes.c_void_p), ctypes.c_void_p(100)) == 0
    return train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64})
'''
FORK_SURFACE = '''
import ctypes


def adapt(transcript, tool_names, gen, train):
    libc = ctypes.CDLL(None)            # a double-forked daemon in its own session: left running after adapt()
    if libc.fork() == 0:
        libc.setsid()
        if libc.fork() == 0:
            libc.sleep(100)
        libc._exit(0)
    return None
'''
TORCH_SURFACE = '''
import numpy as np
import torch


def adapt(transcript, tool_names, gen, train):
    x = torch.randn(64, 64) @ torch.randn(64, 64)       # CPU, single-threaded in the sandbox
    assert np.isfinite(x.numpy()).all()
    return None
'''


def test_budgets_and_contract_are_enforced(tiny, probe_run):
    tmp, data, _, _, _ = probe_run
    w = {n: _surface(tmp, n, s) for n, s in (("loop", LOOP_SURFACE), ("hang", HANG_SURFACE), ("spin", SPIN_SURFACE),
                                            ("foreign", FOREIGN_SURFACE), ("forged", FORGED_SURFACE), ("big", BIG_SURFACE),
                                            ("greedy", GREEDY_SURFACE), ("slow", SLOW_IMPORT_SURFACE))}
    r = _harness(tiny, data, w["loop"], tmp / "o_loop", secs=4)
    assert r.returncode == 0, r.stderr[-2000:]
    s = js(tmp / "o_loop" / "stats.json")
    assert all(x["adapter"] and x["budget_hit"] and not x["killed"] for x in s["worlds"] + s["twins"])   # last adapter kept
    assert js(tmp / "o_loop" / "budget.json")["adapt_s_max"] < 4 + 15
    for name in ("hang", "spin"):         # the supervisor stops serving at deadline + grace and kills the process group
        r = _harness(tiny, data, w[name], tmp / f"o_{name}", secs=3, grace=2, limit=1)
        assert r.returncode == 0, r.stderr[-2000:]
        st = js(tmp / f"o_{name}" / "stats.json")
        assert st["twins"][0]["killed"]
        x = st["worlds"][0]
        assert x["killed"] and x["budget_hit"].startswith("killed") and 5 <= x["adapt_s"] < 9 and x["violations"] == []
        assert x["adapter"] == (name == "hang")
    for name in ("foreign", "forged"):
        r = _harness(tiny, data, w[name], tmp / f"o_{name}", limit=1)
        assert r.returncode != 0 and "adapt() must return" in r.stderr, r.stderr[-2000:]
    r = _harness(tiny, data, w["big"], tmp / "o_big", gen_tokens=10, train_tokens=3_000)
    assert r.returncode == 0, r.stderr[-2000:]      # (this surface uses no gen tokens, so gen_tokens=10 is fine)
    x = js(tmp / "o_big" / "stats.json")["worlds"][0]
    assert x["train"]["stopped"] == "train_tokens" and x["train_tokens"] <= 3_000 and x["train"]["steps"] < x["train"]["planned_steps"]
    assert x["train_tokens"] % 4 == 0          # charged as batch_size x padded length
    r = _harness(tiny, data, w["greedy"], tmp / "o_greedy", gen_tokens=500, limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    x = js(tmp / "o_greedy" / "stats.json")["worlds"][0]
    assert 0 < x["gen_tokens"] <= 500 and not x["adapter"]
    r = _harness(tiny, data, w["slow"], tmp / "o_slow", secs=20)
    assert r.returncode == 0, r.stderr[-2000:]
    st = js(tmp / "o_slow" / "stats.json")
    assert len(st["twins"]) == 2 and all(x["adapt_s"] >= 3 for x in st["worlds"] + st["twins"])
    assert js(tmp / "o_slow" / "budget.json")["adapt_s_max"] >= 3


def test_threads_and_processes_left_by_the_surface_are_caught_and_killed(tiny, probe_run):
    tmp, data, _, _, _ = probe_run
    w = {n: _surface(tmp, n, s) for n, s in (("thread", THREAD_SURFACE), ("fork", FORK_SURFACE), ("torch", TORCH_SURFACE))}
    r = _harness(tiny, data, w["thread"], tmp / "o_thread", limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    x = js(tmp / "o_thread" / "stats.json")["worlds"][0]
    assert any(v.startswith("thread") for v in x["violations"]) and js(tmp / "o_thread" / "stats.json")["violations"] >= 1
    m = _evaluate(tiny, _mini_data(tmp / "data1", n_worlds=1), tmp / "o_thread", tmp, w["thread"])
    assert not m["valid"] and "isolation" in m["message"]
    r = _harness(tiny, data, w["fork"], tmp / "o_fork", limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    x = js(tmp / "o_fork" / "stats.json")["worlds"][0]
    procs = [int(v.split()[1]) for v in x["violations"] if v.startswith("process")]
    assert procs and x["stray_processes_killed"] >= 1
    for p in procs:                       # killed before the adapter was accepted (reaped or at most a zombie)
        assert not os.path.exists(f"/proc/{p}") or open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0] in "ZX"
    r = _harness(tiny, data, w["torch"], tmp / "o_torch", limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    assert js(tmp / "o_torch" / "stats.json")["worlds"][0]["violations"] == []   # numpy / torch on CPU are fine


def test_divergence_flag_is_sticky_and_adapter_ids_are_checked(tiny, tok):
    import trainer as T
    m = _load(tiny)
    tr = T.Trainer(m, Engine(m, tok, "cpu"), ["a_go"], np.load(f"{D}/train/replay.npy"), Budget(120, 0, 10**6), 0, "cpu")
    cfg = {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64}
    good = tr.train([{"text": "> a(1)\nok\n"}], cfg)
    real = tr._loss
    tr._loss = lambda *a: real(*a) * float("nan")       # a diverging second call
    tr.train([{"text": "> a(1)\nok\n"}], cfg)
    tr._loss = real
    assert tr.nan_seen and tr.saved_stats(good)["nan"] is False and tr.produced == [0, 1]
    for bad in (5, -1, True, "0", None, 1.0):
        with pytest.raises(ValueError):
            tr.saved(bad)
    with pytest.raises(ValueError):
        tr.train([{"text": "> a(1)\nok\n"}], cfg, init=7)
    with pytest.raises(ValueError):
        tr.train([{"text": "> a(1)\nok\n"}], {**cfg, "rank": 2}, init=good)   # init must match rank / targets / layers
    assert lora.n_wrapped(m) == 0                            # a refused call never leaves LoRA modules attached


def test_train_config_is_validated():
    import trainer
    base = {"rank": 2, "alpha": 4}
    assert trainer.check_config(base)["replay_rows"] == 0
    for bad in ({"replay_rows": -1}, {"replay_rows": 1.5}, {"replay_rows": True}, {"replay_rows": "3"}, {"replay_rows": 65},
                {"lr": float("nan")}, {"lr": 0}, {"lr": 1.0}, {"epochs": 0}, {"epochs": float("inf")}, {"batch_size": 2.0},
                {"batch_size": 64, "max_len": 1024}, {"kl_base": -1}, {"ce_weight": float("nan")}, {"grad_clip": 0},
                {"warmup": 2}, {"grad_ckpt": 1}, {"rank": True}, {"alpha": float("inf")}, {"targets": "q_proj"},
                {"layers": [-1]}, {"unknown": 1}):
        with pytest.raises(ValueError):
            trainer.check_config({**base, **bad})
    b = Budget(60, 100, 100)
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError):
            b.charge_gen(bad)
        with pytest.raises(ValueError):
            b.charge_train(bad)
    with pytest.raises(BudgetExceeded):
        b.charge_gen(101)
    assert b.gen_used == 0


def test_gen_budget_is_reserved_before_decoding_and_never_exceeded(tiny, tok):
    m = _load(tiny)
    E = Engine(m, tok, "cpu")
    tools = ["alpha_go", "beta_go"]
    b = Budget(120, 10**6, 0)
    g = Gen(E, tools, [{"call": "alpha_go(1)", "obs": "ok"}], b, 0)
    g.generate(["x"], max_new_tokens=4, context=False)
    pre = len(E.encode(chat_prefix(system_text(tools))))
    ls = len(E.encode(chat_user("x")))
    assert pre + ls <= b.gen_used <= pre + ls + 3          # prefix + prompt + decode steps actually run
    used = b.gen_used
    g.generate(["x", "a longer prompt here"], max_new_tokens=4, context=False)
    ls2 = len(E.encode(chat_user("a longer prompt here")))
    assert b.gen_used - used >= 2 * ls2                    # padded rows are charged
    cap = b.gen_used + 2 * (ls2 + 4) - 1                    # one position short of the worst case
    b2 = Budget(120, cap, 0)
    b2.gen_used = b.gen_used
    g._budget = b2
    with pytest.raises(BudgetExceeded):                     # refused before decoding, not charged after the batch
        g.generate(["x", "a longer prompt here"], max_new_tokens=4, context=False)
    assert b2.gen_used == b.gen_used
    cap3 = pre + 100
    b3 = Budget(120, cap3, 0)
    g3 = Gen(E, tools, [], b3, 0)
    n_ok = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            g3.generate(["x", "y", "z"], max_new_tokens=8, context=False)
            n_ok += 1
            assert b3.gen_used <= cap3
    assert n_ok >= 1 and b3.gen_used <= cap3


def test_baseline_surface_contract(tiny):
    sys.path.insert(0, "/work")
    import adapt as surface
    assert callable(surface.adapt) and not hasattr(surface, "ARM") and arm_of("/work", "/frozen") == "adapter"
    calls = []

    class G:
        transcript_text = "> a(1)\nok\n"

    def fake_train(examples, config=None, init=None):
        calls.append((examples, config))
        return "adapter"

    assert surface.adapt([{"call": "a(1)", "obs": "ok"}], ["a"], G(), fake_train) == "adapter"
    (ex, cfg), = calls
    assert ex == [{"text": G.transcript_text}] and lora.check_config(cfg)["rank"] == 16
