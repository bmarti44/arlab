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
            assert p["spec"] == json.loads(json.dumps(fauxos.gen_world(seed))), "data was prepared with another generator"
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


# ================================================================ transcripts reveal only observable facts
def test_transcript_is_a_faithful_replay():
    """Every observation is exactly what the simulator prints for that call, in order, from the initial state; the
    task start state is where the replay ends. Nothing else is in the transcript."""
    for split in SPLITS:
        pub, priv = pub_worlds(f"{D}/{split}"), js(f"{D}/{split}/private/worlds.json")
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
            errs = {e["obs"].split()[1] for e in w["transcript"] if e["obs"].startswith("error ")}
            assert set(spec["errors"].values()) <= errs, "every error code must be observed at least once"
            called = {e["call"].split("(")[0] for e in w["transcript"]}
            assert called == set(w["tools"]), "every tool must appear in the transcript"


PRIVATE_MARKERS = ('"op"', '"perm"', '"var"', '"verbose"', "skip_locked", "with_locked", "gold_state", '"reference"',
                   '"spec"', "find_tag", "badarg", "Goal:")


def test_public_and_train_data_reveal_no_hidden_state_or_labels():
    for split in SPLITS:
        order = js(f"{D}/{split}/public/order.json")
        raw = "".join(open(f"{D}/{split}/public/worlds/{w}.json").read() for w in order)
        for m in PRIVATE_MARKERS:
            assert m not in raw, f"{m} in public data"
        pub, priv = pub_worlds(f"{D}/{split}"), js(f"{D}/{split}/private/worlds.json")
        assert all(set(w) == {"id", "tools", "transcript"} for w in pub)
        assert sorted(os.listdir(f"{D}/{split}/public")) == ["order.json", "worlds"]
        assert sorted(os.listdir(f"{D}/{split}/public/worlds")) == sorted(f"{w}.json" for w in order)
        for w in pub:
            text = render_transcript(w["transcript"])
            for t in priv[w["id"]]["tasks"]:
                assert t["goal"] not in raw
                if t["answer"] is None:      # a mutation task's full reference program never appears in the transcript
                    assert not all(f"> {line}\n" in text for line in t["reference"]), t
            assert str(priv[w["id"]]["spec"]["seed"]) not in raw
    assert sorted(os.listdir(f"{D}/train")) == ["replay.npy"]


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
    assert src.count("a.data") == 3 and "f'{a.data}/public/order.json'" in src and "f'{a.data}/train/replay.npy'" in src
    assert "f'{a.data}/public/worlds/{wid}.json'" in src          # one world file at a time


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

    if locked:
        assert fauxos.call(spec, st, move["name"], args(locked, place)) == (f"error {E['locked']}", False)
    assert fauxos.call(spec, st, move["name"], args(99, place)) == (f"error {E['missing']}", False)
    assert fauxos.call(spec, st, move["name"], args(free, place)[::-1]) == (f"error {E['badarg']}", False)
    assert fauxos.call(spec, st, move["name"], [int(free)]) == (f"error {E['badarg']}", False)
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
sys.path.insert(0, "/frozen")
import lora
ARM = "icl"          # ignored: the arm is decided by frozen code from the file hash
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.jsonl")


def adapt(transcript, tool_names, gen, train):
    m = gen._e.model
    main = vars(sys.modules["__main__"])
    worlds_seen = [k for k, v in main.items() if (isinstance(v, dict) and "transcript" in v)
                   or (isinstance(v, list) and any(isinstance(x, dict) and "transcript" in x for x in v))]
    rec = {"wrapped": lora.n_wrapped(m), "fp": lora.fingerprint(m), "n_events": len(transcript), "worlds_seen": worlds_seen,
           "same": main["w"]["transcript"] == transcript,
           "keys": sorted({k for e in transcript for k in e}), "gen_attrs": sorted(vars(gen)), "tools": tool_names}
    out = gen.generate([gen.task_prompt("Archive item 100.")], max_new_tokens=3, context=False)
    p, c = gen.task_prompt("Lock item 7."), tool_names[0] + "(7)"
    t = gen.teacher([p], [c], k=4, context=True)[0]
    ad = train([{"prompt": p, "completion": c, "teacher": t}, {"text": "> x(1)\\nok\\n", "weight": 0.5}],
               {"rank": 2, "alpha": 4, "epochs": 3, "batch_size": 2, "max_len": 256, "kl_base": 0.1, "replay_rows": 1})
    ad2 = train([{"prompt": p, "completion": c}], {"rank": 2, "alpha": 4, "epochs": 1, "max_len": 256}, init=ad)
    ad.tensors[next(iter(ad.tensors))].fill_(7.0)           # editing the returned copy changes nothing saved
    ad.cfg["rank"] = 64
    rec.update(out=out, gen_used=gen.tokens_left, stats=ad.stats, stats2=ad2.stats, wrapped_after=lora.n_wrapped(m))
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\\n")
    return ad2 if len(tool_names) % 2 else ad
'''


def _mini_data(root, n_tasks=3, n_worlds=2):
    """A 2-world copy of the validation split with few tasks and small battery (fast on CPU)."""
    pub = pub_worlds(f"{D}/validation")[:n_worlds]
    priv = js(f"{D}/validation/private/worlds.json")
    for sub in ("public/worlds", "private", "train"):
        os.makedirs(f"{root}/{sub}", exist_ok=True)
    json.dump([w["id"] for w in pub], open(f"{root}/public/order.json", "w"))
    for w in pub:
        json.dump(w, open(f"{root}/public/worlds/{w['id']}.json", "w"))
    json.dump({w["id"]: {**priv[w["id"]], "tasks": priv[w["id"]]["tasks"][:n_tasks]} for w in pub},
              open(f"{root}/private/worlds.json", "w"))
    g = js(f"{D}/validation/private/guard.json")
    json.dump({**g, "tasks": g["tasks"][:2]}, open(f"{root}/private/guard.json", "w"))
    json.dump(js(f"{D}/validation/private/gsm8k.json")[:2], open(f"{root}/private/gsm8k.json", "w"))
    np.save(f"{root}/private/text.npy", np.load(f"{D}/validation/private/text.npy")[:2, :65])
    shutil.copy(f"{D}/validation/private/fauxos.py", f"{root}/private/fauxos.py")
    shutil.copy(f"{D}/train/replay.npy", f"{root}/train/replay.npy")
    return root


def _harness(tiny, data, work, out, secs=120, gen_tokens=10**6, train_tokens=10**6, limit=0):
    os.makedirs(out, exist_ok=True)
    return subprocess.run([sys.executable, "/frozen/harness.py", "--out", str(out), "--seed", "1", "--split", "validation",
                           "--adapt-seconds", str(secs), "--gen-tokens", str(gen_tokens), "--train-tokens", str(train_tokens),
                           "--device", "cpu", "--model", tiny, "--data", str(data), "--work", str(work),
                           "--limit-worlds", str(limit)],
                          capture_output=True, text=True, env=ENV, timeout=600)


def _evaluate(tiny, data, run, tmp, work):
    out = f"{tmp}/metrics.json"
    r = subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", out, "--data", str(data),
                        "--cache", f"{tmp}/cache", "--device", "cpu", "--model", tiny, "--max-new", "8", "--work", str(work)],
                       capture_output=True, text=True, env=ENV, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return js(out)


@pytest.fixture(scope="module")
def probe_run(tiny, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probe")
    data = _mini_data(tmp / "data")
    work = tmp / "work"
    work.mkdir()
    (work / "adapt.py").write_text(PROBE_SURFACE)
    r = _harness(tiny, data, work, tmp / "out")
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp, data, [json.loads(x) for x in open(work / "log.jsonl")]


def test_harness_resets_adapters_between_worlds_and_exposes_only_the_transcript(probe_run):
    tmp, data, log = probe_run
    assert len(log) == 2
    assert all(r["wrapped"] == 0 for r in log) and log[0]["fp"] == log[1]["fp"]      # fresh base model per world
    assert all(r["wrapped_after"] == 0 for r in log)
    assert all(r["keys"] == ["call", "obs"] for r in log)
    assert all(r["worlds_seen"] == ["w"] and r["same"] for r in log), "adapt() may see only the world being adapted"
    assert log[0]["gen_attrs"] == ["_budget", "_calls", "_e", "_prefix", "_seed", "tool_names", "transcript", "transcript_text"]
    for r in log:
        assert r["stats"]["steps"] == 3 and r["stats"]["nan"] is False and r["stats2"]["steps"] == 1
    stats = js(tmp / "out" / "stats.json")
    assert stats["arm"] == "adapter" and [w["id"] for w in stats["worlds"]] == ["v0", "v1"]   # ARM = "icl" ignored
    assert [w["donor"] for w in stats["worlds"]] == ["v0", "v1"] and stats["nan"] is False
    assert all(w["gen_tokens"] > 0 and w["train_tokens"] > 0 for w in stats["worlds"])
    assert sorted(os.listdir(tmp / "out" / "adapters")) == ["v0", "v1"]
    (c0, a0), (c1, a1) = (lora.load(tmp / "out" / "adapters" / w) for w in ("v0", "v1"))
    assert any(not torch.equal(a0[k], a1[k]) for k in a0)
    assert c0["rank"] == c1["rank"] == 2 and not any(bool((t == 7.0).all()) for t in list(a0.values()) + list(a1.values()))
    assert js(tmp / "out" / "budget.json")["adapt_s_max"] > 0


def test_evaluator_end_to_end_and_invalid_outputs(tiny, probe_run):
    tmp, data, _ = probe_run
    work = tmp / "work"
    m = _evaluate(tiny, data, tmp / "out", tmp, work)
    tasks = [t["id"] for p in js(data / "private" / "worlds.json").values() for t in p["tasks"]]
    assert m["valid"] and sorted(m["items"]) == sorted(tasks) and set(m["items"].values()) <= {0.0, 1.0}
    assert m["primary"] == m["metrics"]["success"]
    for k in ("success", "cross_world_success", "none_success", "battery_drop", "prefill_tokens", "text_nll_ratio"):
        assert np.isfinite(m["metrics"][k])
    assert m["metrics"]["prefill_tokens"] < 1500 and m["metrics"]["n_adapters"] == 2
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
    assert not variant("arm", set_stat(arm="oracle"))["valid"]
    assert not variant("icl_with_adapters", set_stat(arm="icl"))["valid"]
    assert not variant("claims_placebo", set_stat(arm="placebo"))["valid"]      # the arm is the surface's hash, not a claim
    assert not variant("foreign_world", world_stat(0, donor="v1"))["valid"]     # a candidate adapts on its own world
    assert not variant("missing", lambda d: shutil.rmtree(d / "adapters" / "v1"))["valid"]
    assert not variant("nostats", lambda d: os.remove(d / "stats.json"))["valid"]

    def poison(d):
        from safetensors.torch import load_file, save_file
        t = load_file(d / "adapters" / "v0" / "adapter.safetensors")
        k = next(iter(t))
        t[k][0, 0] = float("inf")
        save_file(t, d / "adapters" / "v0" / "adapter.safetensors")
    assert not variant("inf", poison)["valid"]


def test_reference_arms(tiny, probe_run):
    tmp, data, _ = probe_run
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
        assert s["arm"] == arm and all(w["adapter"] == (arm == "placebo") for w in s["worlds"])
    s = js(tmp / "ref_placebo" / "stats.json")
    assert [w["donor"] for w in s["worlds"]] == ["v1", "v0"]        # world i adapts on world i+1's transcript
    icl = _evaluate(tiny, data, tmp / "ref_icl", tmp, "/frozen/ref_icl")
    none = _evaluate(tiny, data, tmp / "ref_none", tmp, "/frozen/ref_none")
    plc = _evaluate(tiny, data, tmp / "ref_placebo", tmp, "/frozen/ref_placebo")
    assert icl["valid"] and none["valid"] and plc["valid"], plc["message"]
    assert plc["primary"] == plc["metrics"]["success"] and "cross_world_success" not in plc["metrics"]
    # a run evaluated against a different surface than the one that produced it is invalid
    assert not _evaluate(tiny, data, tmp / "ref_placebo", tmp, "/work")["valid"]
    # the ICL arm carries the transcript in every prompt: far over the prefill guard; no-adaptation does not
    assert icl["metrics"]["prefill_tokens"] > 1000
    assert none["metrics"]["prefill_tokens"] < 1500 and none["metrics"]["battery_drop"] == 0.0
    assert none["primary"] == none["metrics"]["none_success"]


LOOP_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    ad = None
    while True:        # never returns by itself: the harness deadline must stop it
        ad = train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64}, init=ad)
'''
FOREIGN_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    return {"weights": "from somewhere else"}
'''
TAMPER_SURFACE = '''
import torch


def adapt(transcript, tool_names, gen, train):
    with torch.no_grad():
        gen._e.model.lm_head.weight[0, 0] += 1.0     # edits the base model instead of returning an adapter
    return None
'''
BIG_SURFACE = '''

def adapt(transcript, tool_names, gen, train):
    return train([{"text": "> a(1)\\nok\\n" * 8}] * 20, {"rank": 1, "alpha": 1, "epochs": 50, "max_len": 64, "batch_size": 4})
'''


SLOW_IMPORT_SURFACE = '''
import time
time.sleep(3)          # module-level work is charged to world 0's adapt budget


def adapt(transcript, tool_names, gen, train):
    return None
'''
NAN_SURFACE = '''
import sys
import torch


def adapt(transcript, tool_names, gen, train):
    good = train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64})
    T = sys.modules["trainer"].Trainer
    real = T._loss
    T._loss = lambda self, *a: real(self, *a) * float("nan")       # simulate a diverging second call
    try:
        train([{"text": "> a(1)\\nok\\n"}], {"rank": 1, "alpha": 1, "epochs": 1, "max_len": 64})
    finally:
        T._loss = real
    return good                                                     # a finite adapter from before the divergence
'''


def test_budgets_and_contract_are_enforced(tiny, probe_run):
    tmp, data, _ = probe_run
    for name, src in (("loop", LOOP_SURFACE), ("foreign", FOREIGN_SURFACE), ("tamper", TAMPER_SURFACE), ("big", BIG_SURFACE),
                      ("slow", SLOW_IMPORT_SURFACE), ("nan", NAN_SURFACE)):
        (tmp / f"w_{name}").mkdir()
        (tmp / f"w_{name}" / "adapt.py").write_text(src)
    r = _harness(tiny, data, tmp / "w_loop", tmp / "o_loop", secs=4)
    assert r.returncode == 0, r.stderr[-2000:]
    s = js(tmp / "o_loop" / "stats.json")
    assert all(w["adapter"] and w["budget_hit"] for w in s["worlds"])            # the last adapter trained is kept
    assert js(tmp / "o_loop" / "budget.json")["adapt_s_max"] < 4 + 15
    r = _harness(tiny, data, tmp / "w_foreign", tmp / "o_foreign")
    assert r.returncode != 0 and "adapt() must return" in r.stderr
    r = _harness(tiny, data, tmp / "w_tamper", tmp / "o_tamper")
    assert r.returncode != 0 and "base model changed" in r.stderr
    r = _harness(tiny, data, tmp / "w_big", tmp / "o_big", gen_tokens=10, train_tokens=3_000)
    assert r.returncode == 0, r.stderr[-2000:]      # (this surface uses no gen tokens, so gen_tokens=10 is fine)
    w = js(tmp / "o_big" / "stats.json")["worlds"][0]
    assert w["train"]["stopped"] == "train_tokens" and w["train_tokens"] <= 3_000 and w["train"]["steps"] < w["train"]["planned_steps"]
    assert w["train_tokens"] % 4 == 0          # charged as batch_size x padded length
    r = _harness(tiny, data, tmp / "w_slow", tmp / "o_slow", secs=2, limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    w = js(tmp / "o_slow" / "stats.json")["worlds"][0]
    assert w["import_s"] >= 3 and w["adapt_s"] >= 3 and js(tmp / "o_slow" / "budget.json")["adapt_s_max"] >= 3
    r = _harness(tiny, data, tmp / "w_nan", tmp / "o_nan", limit=1)
    assert r.returncode == 0, r.stderr[-2000:]
    s = js(tmp / "o_nan" / "stats.json")
    assert s["nan"] is True and s["worlds"][0]["nan_seen"] and s["worlds"][0]["train"]["nan"] is False
    m = _evaluate(tiny, _mini_data(tmp / "data1", n_worlds=1), tmp / "o_nan", tmp, tmp / "w_nan")
    assert not m["valid"] and "diverged" in m["message"]


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
