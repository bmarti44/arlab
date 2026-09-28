"""Deterministic CPU checks for ttt-context (run by `arlab check` as root, no GPU; data at /data, surface at /work,
the whole sealed pack at /pack: the generator is read from /pack/frozen/prepare because RUN never sees it).
The model-level tests use a tiny random-init Qwen3 (2 layers, d=64, the real vocabulary), never the 1.7B weights."""
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch

sys.path[:0] = ["/work", "/eval", "/pack/frozen/prepare"]
import docgen  # noqa: E402
import ttt as surface  # noqa: E402  (the baseline surface at TESTS time)
import tttlib  # noqa: E402
from common import ANSWER_PREFIX, MAX_NEW, MODEL_DIR, QUESTION  # noqa: E402
from docgen import CONTEXT_TOKENS, N_ITEMS  # noqa: E402
from scoring import answer_text, normalize, score  # noqa: E402
from solver import solve  # noqa: E402

D = "/data"
SPLITS = ("validation", "holdout")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _count(tok):
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def _json(p):
    return json.load(open(p))


def _docs(split):
    flat, off = np.load(f"{D}/{split}/public/docs.npy"), np.load(f"{D}/{split}/public/offsets.npy")
    return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]


# ---------------------------------------------------------------- generator: determinism, answers follow the document
def test_generator_deterministic(tok):
    for kind in docgen.KINDS:
        a, b = docgen.build(77, kind, 1500, _count(tok)), docgen.build(77, kind, 1500, _count(tok))
        assert a == b
        c = docgen.build(78, kind, 1500, _count(tok))
        assert c["text"] != a["text"] and c["question"] != a["question"]
        assert 1500 - 64 < a["n_tokens"] <= 1500


def test_generator_answer_is_a_function_of_the_document():
    """For many seeds: the text solver reproduces the answer; removing the last critical event changes it;
    the answer never appears in the question."""
    fake = lambda s: len(s) // 4  # noqa: E731  (a stand-in tokenizer keeps this fast)
    for seed in range(120):
        kind = docgen.KINDS[seed % 4]
        doc = docgen.generate(seed, kind, 300)
        text = docgen.render(doc)
        assert solve(text, doc["question"]) == docgen.replay(doc) == doc["answer"], (seed, kind)
        crit = [i for i, e in enumerate(doc["events"]) if e[2] == docgen.CRITICAL]
        cf = dict(doc, events=[e for i, e in enumerate(doc["events"]) if i != crit[-1]])
        assert solve(docgen.render(cf), doc["question"]) != doc["answer"], (seed, kind)
        assert not re.search(rf"\b{re.escape(normalize(doc['answer']))}\b", normalize(doc["question"])), (seed, kind)
        it = docgen.build(seed, kind, 1200, fake)
        assert solve(it["text"], it["question"]) == it["answer"]


def test_distractors_are_hard():
    """Every question has near-duplicate entities and at least two events of its kind about other entities."""
    for seed in range(40):
        kind = docgen.KINDS[seed % 4]
        doc = docgen.generate(seed, kind, 300)
        prot = [e for e in doc["events"] if e[2] == docgen.PROTECTED]
        assert len(prot) >= 2, (seed, kind)
        names = {str(e[1][0]) for e in prot} - {doc["target"]}
        assert names or kind == "count", (seed, kind)


# ---------------------------------------------------------------- prepared data: sizes, splits, no leakage
def test_splits_sized_disjoint_and_constant_length(tok):
    tail = tok(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]   # ... assistant header + "Answer:"
    assert tail[0] == 151645 and tail[-1] != 151645
    sp, info = _json(f"{D}/splits.json"), _json(f"{D}/info.json")
    assert info["difficulty"] == json.loads(json.dumps(docgen.DIFFICULTY)), "data prepared with other generator knobs"
    ctx = info["context_tokens"]
    assert ctx == CONTEXT_TOKENS and sp == {"validation": N_ITEMS, "holdout": N_ITEMS} == {s: info["n_items"] for s in SPLITS}
    hashes, questions, seeds = {}, {}, {}
    for split in SPLITS:
        docs, items, gold = _docs(split), _json(f"{D}/{split}/public/items.json"), _json(f"{D}/{split}/private/answers.json")
        assert len(docs) == len(items) == len(gold) == sp[split]
        assert [it["id"] for it in items] == list(gold)
        assert all(ctx - 128 < len(d) <= ctx for d in docs)
        assert all(tuple(d[:3]) == tttlib.HEAD_IDS for d in docs)
        assert all(it["q"][-len(tail):] == tail for it in items)       # the prompt stops before the answer
        hashes[split] = {hashlib.sha256(d.tobytes()).hexdigest() for d in docs}
        questions[split] = {tuple(it["q"]) for it in items}
        seeds[split] = {g["seed"] for g in gold.values()}
        assert len(hashes[split]) == len(docs) and len(questions[split]) == len(items)
        kinds = [g["kind"] for g in gold.values()]
        assert all(kinds.count(k) == len(kinds) // 4 for k in docgen.KINDS)
    assert not hashes["validation"] & hashes["holdout"]
    assert not questions["validation"] & questions["holdout"]
    assert not seeds["validation"] & seeds["holdout"]


def test_prepared_answers_follow_the_document(tok):
    for split in SPLITS:
        docs, items, gold = _docs(split), _json(f"{D}/{split}/public/items.json"), _json(f"{D}/{split}/private/answers.json")
        for d, it in zip(docs, items):
            g = gold[it["id"]]
            q = tok.decode(it["q"])
            assert solve(tok.decode(d.tolist()), q) == g["answer"], (split, it["id"])
            assert score(g["answer"], g["aliases"]) == 1.0
            assert not re.search(rf"\b{re.escape(normalize(g['answer']))}\b", normalize(q.split("<|im_end|>")[0]))
        # no answer is guessable from the question alone: the best constant guess per kind is rare
        answers = [g["answer"] for g in gold.values()]
        assert max(answers.count(x) for x in set(answers)) / len(answers) <= max(0.08, 2 / len(answers))
        for k in docgen.KINDS:
            ak = [g["answer"] for g in gold.values() if g["kind"] == k]
            assert max(ak.count(x) for x in set(ak)) / len(ak) <= max(0.3, 2 / len(ak)), k


def test_run_cannot_regenerate_the_data():
    """The generator, PREPARE and the split seeds are PREPARE-only; nothing RUN mounts (frozen/run) or the agent sees
    (program.md, IDEA.md) contains them."""
    run_files = [os.path.join(r, f) for r, _, fs in os.walk("/frozen") for f in fs if "__pycache__" not in r]
    assert not {"docgen.py", "prepare.py"} & {os.path.basename(f) for f in run_files}, run_files
    for f in run_files + ["/pack/program.md", "/pack/IDEA.md"]:
        src = open(f, errors="replace").read()
        for needle in ("SPLIT_SEEDS", "1_000_000", "2_000_000", "1,000,000", "2,000,000", "docgen", "_ONSETS",
                       "TEMPLATES", "def replay", "random.Random("):
            assert needle not in src, (f, needle)
    assert 'command: "python /prepare/prepare.py' in open("/pack/pack.yaml").read()


def test_public_data_has_no_answers():
    for split in SPLITS:
        items = _json(f"{D}/{split}/public/items.json")
        assert all(set(it) == {"id", "q"} for it in items)
        assert sorted(os.listdir(f"{D}/{split}/public")) == ["docs.npy", "items.json", "offsets.npy"]
    for f in ("/frozen/harness.py", "/frozen/tttlib.py"):
        src = open(f).read()
        assert "private" not in src and "/eval" not in src and "holdout" not in src, f
    ctx = tttlib.TTTContext(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long), None, 0, 0.0)
    public = {k for k in dir(ctx) if not k.startswith("_")}
    assert public == {"doc_ids", "question_ids", "doc_len", "device", "seed", "generator", "time_left", "prefix_cache"}


# ---------------------------------------------------------------- scorer and evaluator (frozen EVALUATE)
def test_normalize_and_score():
    assert score("Dock Velm.", ["Velm", "dock Velm"]) == 1.0 and score("velm", ["Velm", "dock Velm"]) == 1.0
    assert score("**Velm**", ["Velm"]) == 1.0 and score("Room 214", ["214", "room 214"]) == 1.0
    assert score("three", ["3"]) == 1.0 and score("The answer: 3", ["3"]) == 0.0
    assert score("Velm or Arro", ["Velm"]) == 0.0 and score("Velm, Velmi", ["Velm"]) == 0.0
    assert score("Crate K-482 is at dock Velm.", ["Velm", "dock Velm"]) == 0.0
    assert score("", ["3"]) == 0.0 and score("48213", ["48123"]) == 0.0 and score("4821", ["48213"]) == 0.0
    assert score("-3", ["3"]) == 0.0 and score("3%", ["3"]) == 0.0 and score("3.5", ["3"]) == 0.0
    assert score("(3)", ["3"]) == 1.0 and score("48213.", ["48213"]) == 1.0
    assert answer_text(" 48213<|im_end|>junk") == "48213" and answer_text("\n 8\n\nExplanation: 7") == "8"
    assert score(answer_text("Velm\nArro"), ["Arro"]) == 0.0      # only the first line counts


def _evaluate(tmp_path, preds, per=None, ttt_seconds=4.0, limit=0, **stats):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    gold = _json(f"{D}/validation/private/answers.json")
    ids = list(gold)[:limit] if limit else list(gold)
    per = per or {i: {"ttt_s": 1.0, "prefill_s": 1.0, "answer_s": 0.1, "reset_ok": True, "probe_diff": 0.0,
                      "nonfinite": 0, "changed_tensors": 28, "doc_in_context": True} for i in ids}
    st = {"ttt_seconds_limit": ttt_seconds, "limit": limit, "sha_start": "x", "sha_end": "x", "sha_end_work": "x"} | stats
    json.dump(st, open(run / "stats.json", "w"))
    json.dump(preds, open(run / "preds.json", "w"))
    json.dump(per, open(run / "items.json", "w"))
    out = tmp_path / "m.json"
    cmd = [sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", str(out), "--data", f"{D}/validation",
           "--ttt-seconds", str(ttt_seconds)] + (["--limit", str(limit)] if limit else [])
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(out))


def test_evaluator_scores_raw_tokens(tmp_path, tok):
    gold = _json(f"{D}/validation/private/answers.json")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"][:MAX_NEW]  # noqa: E731
    preds = {i: enc(g["aliases"][-1] + "<|im_end|>") for i, g in gold.items()}
    m = _evaluate(tmp_path, preds)
    assert m["valid"] and m["primary"] == 1.0 and set(m["items"].values()) == {1.0}
    assert {f"acc_{k}" for k in docgen.KINDS} <= set(m["metrics"])
    ids = list(gold)
    other = next(g["answer"] for i, g in gold.items() if g["kind"] == gold[ids[0]]["kind"] and g["answer"] != gold[ids[0]]["answer"])
    preds[ids[0]] = enc(f"{gold[ids[0]]['answer']} or {other}<|im_end|>")   # hedged
    preds[ids[1]] = enc("I don't know<|im_end|>")
    preds[ids[2]] = enc(" " + gold[ids[2]]["answer"] + ".\n<|im_end|>")     # whitespace + period: fine
    m = _evaluate(tmp_path, preds)
    assert m["items"][ids[0]] == 0.0 and m["items"][ids[1]] == 0.0 and m["items"][ids[2]] == 1.0
    assert abs(m["primary"] - (len(ids) - 2) / len(ids)) < 1e-12


def test_evaluator_enforces_budget_and_reset(tmp_path, tok):
    gold = _json(f"{D}/validation/private/answers.json")
    preds = {i: tok(g["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"][:MAX_NEW] for i, g in gold.items()}
    base = {i: {"ttt_s": 1.0, "prefill_s": 1.0, "answer_s": 0.1, "reset_ok": True, "probe_diff": 0.0, "nonfinite": 0}
            for i in gold}
    ids = list(gold)
    slow = {i: dict(v) for i, v in base.items()}
    slow[ids[0]]["ttt_s"] = 4.01                       # over the 4 s budget: correct answer, scored 0
    m = _evaluate(tmp_path, preds, per=slow)
    assert m["valid"] and m["items"][ids[0]] == 0.0 and m["metrics"]["over_budget_items"] == 1.0
    bad = {i: dict(v) for i, v in base.items()}
    bad[ids[3]]["reset_ok"] = False
    assert not _evaluate(tmp_path, preds, per=bad)["valid"]
    assert not _evaluate(tmp_path, preds, per=base, sha_end="y")["valid"]            # weights not restored
    assert not _evaluate(tmp_path, preds, per=base, ttt_seconds_limit=8.0)["valid"]  # run used another budget
    nf = {i: dict(v) for i, v in base.items()}
    nf[ids[1]]["nonfinite"] = 1
    assert not _evaluate(tmp_path, preds, per=nf)["valid"]
    missing = dict(list(preds.items())[1:])
    assert not _evaluate(tmp_path, missing, per=base)["valid"]
    long = {**preds, ids[2]: [15] * (MAX_NEW + 1)}
    assert not _evaluate(tmp_path, long, per=base)["valid"]
    assert not _evaluate(tmp_path, preds, per=base, sha_end_work="y")["valid"]       # the surface's copy not restored
    for bad_tok in (True, 1.0, -1, len(tok), 10**9):                                  # exact ints inside the vocabulary
        assert not _evaluate(tmp_path, {**preds, ids[2]: [15, bad_tok]}, per=base)["valid"], bad_tok
    for bad_nf in (0.5, False, "0", None):
        nf = {i: dict(v) for i, v in base.items()}
        nf[ids[1]]["nonfinite"] = bad_nf
        assert not _evaluate(tmp_path, preds, per=nf)["valid"], bad_nf


# ---------------------------------------------------------------- the frozen item loop on a tiny model
def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768,
                      tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():   # larger weights than the default init, so answers depend on the context
        for p in m.parameters():
            p.mul_(8 if p.dim() > 1 else 1)
        m.model.embed_tokens.weight.mul_(0.25)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def small(tok):
    """Four small real documents (~600 tokens) with their question prompts."""
    docs, items = [], []
    for i, kind in enumerate(docgen.KINDS):
        it = docgen.build(500 + i, kind, 600, _count(tok))
        docs.append(np.asarray(tok(it["text"], add_special_tokens=False)["input_ids"]))
        items.append({"id": f"t{i}", "q": tok(QUESTION.format(q=it["question"]), add_special_tokens=False)["input_ids"]})
    return docs, items


def _run(adapt, small, n=None, model=None, ttt_seconds=60, loader=None):
    """run_items on the small documents with a surface given as an adapt function (or a module loader)."""
    loader = loader or (lambda: types.SimpleNamespace(adapt=adapt))
    return tttlib.run_items(model or tiny_model(), loader, small[0][:n], small[1][:n], ttt_seconds=ttt_seconds, seed=1,
                            device="cpu", log=lambda s: None)


def _no_ttt(model, ctx):
    return {"weights": {}, "doc_in_context": True}


def test_prefix_cache_and_greedy_match_full_forward(small):
    model = tiny_model()
    doc = torch.as_tensor(small[0][0]).unsqueeze(0)
    q = torch.as_tensor(small[1][0]["q"]).unsqueeze(0)
    cache = tttlib.prefill(model, doc)
    ctx = tttlib.TTTContext(doc, q, [(lay.keys, lay.values) for lay in cache.layers], 0, time.perf_counter() + 60)
    with torch.no_grad():
        full = model(input_ids=doc, use_cache=False).logits
        for start in (1, 200, doc.shape[1] - 64):
            part = model(input_ids=doc[:, start:start + 64], past_key_values=ctx.prefix_cache(start), use_cache=True).logits
            assert (part - full[:, start:start + 64]).abs().max() < 1e-4, start
        toks, bad = tttlib.greedy(model, q, tttlib.cache_view(cache, doc.shape[1]))
        seq, ref = torch.cat([doc, q], 1), []
        for _ in range(len(toks)):
            nxt = int(model(input_ids=seq, use_cache=False).logits[0, -1].argmax())
            ref.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], 1)
    assert bad == 0 and toks == ref
    assert cache.get_seq_length() == doc.shape[1]  # the harness cache was not extended


def test_baseline_surface_updates_only_q_proj_and_everything_is_reset(small):
    model = tiny_model()
    ref = {n: p.clone() for n, p in model.named_parameters()}
    seen = {}

    def spy(work, ctx):
        assert work is not model                                   # the surface never gets the answering model
        out = surface.adapt(work, ctx)
        seen[ctx.seed] = sorted(n for n, p in work.named_parameters() if not torch.equal(p, ref[n]))
        assert sorted(out["weights"]) == seen[ctx.seed] and out["doc_in_context"] is True
        return out

    preds, per, stats = _run(spy, small, model=model)
    q = sorted(n for n in ref if n.endswith("self_attn.q_proj.weight"))
    assert all(v == q for v in seen.values()) and len(q) == 2
    assert all(r["reset_ok"] and r["changed_tensors"] == 2 and r["doc_in_context"] for r in per.values())
    assert stats["sha_start"] == stats["sha_end"] == stats["sha_end_work"] and stats["all_reset_ok"]
    assert all(torch.equal(p, ref[n]) and not p.requires_grad for n, p in model.named_parameters())
    assert all(1 <= len(t) <= MAX_NEW for t in preds.values())


def test_returned_weights_are_what_answers(small):
    """Only the returned weights reach the answer model: zeroing the final norm changes every answer."""
    base, _, _ = _run(_no_ttt, small, n=2)

    def zero_norm(work, ctx):
        return {"weights": {"model.norm.weight": torch.zeros_like(work.model.norm.weight)}, "doc_in_context": True}

    preds, per, stats = _run(zero_norm, small, n=2)
    assert all(t[0] == 0 for t in preds.values()) and preds != base          # all-zero logits -> token 0
    assert all(r["reset_ok"] for r in per.values()) and stats["sha_start"] == stats["sha_end"]


def test_surface_code_cannot_touch_answering(small):
    """Hooks (module and global), class patches, patched harness functions, attention registry entries, instance
    attributes and config changes made during adapt() are all gone before the frozen answer: answers equal no_ttt's."""
    base, _, _ = _run(_no_ttt, small)
    lin_forward = torch.nn.Linear.forward

    def vandal(work, ctx):
        with torch.no_grad():
            for p in work.parameters():
                p.add_(0.5)
                p.requires_grad_(True)
        forced = lambda m, i, o: torch.full_like(o, -1e4).index_fill_(-1, torch.tensor([42]), 1e4)  # noqa: E731
        work.lm_head.register_forward_hook(forced)
        torch.nn.modules.module.register_module_forward_hook(lambda m, i, o: forced(m, i, o) if m.__class__.__name__ ==
                                                             "Linear" and o.shape[-1] > 1000 else o)
        torch.nn.Linear.forward = lambda self, x: torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype)
        tttlib.greedy = lambda *a, **k: ([42], 0)
        import transformers.modeling_utils as mu
        mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("patched"))
        work.stash = "x"
        work.model.layers[0].self_attn.scaling = 99.0
        work.config.rms_norm_eps = 1.0
        work.train()
        return {"weights": {}, "doc_in_context": True}

    try:
        preds, per, stats = _run(vandal, small)
    finally:
        torch.nn.Linear.forward = lin_forward
    assert torch.nn.Linear.forward is lin_forward and tttlib.greedy.__name__ == "greedy"
    assert preds == base
    assert all(r["reset_ok"] for r in per.values()) and stats["sha_start"] == stats["sha_end"] == stats["sha_end_work"]
    assert not torch.nn.modules.module._global_forward_hooks


def test_no_state_survives_between_items(small, tmp_path, monkeypatch):
    """Module globals, helper modules, model attributes, parameters and RNG state are fresh for every item."""
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "surf_helper.py").write_text("SEEN = []\n")
    (tmp_path / "surf_probe.py").write_text(
        "import torch\nimport surf_helper\nCOUNT = []\n"
        "def adapt(model, ctx):\n"
        "    COUNT.append(1)\n    surf_helper.SEEN.append(ctx.seed)\n"
        "    assert len(COUNT) == 1 and len(surf_helper.SEEN) == 1\n"
        "    assert not hasattr(model, 'stash') and model.model.layers[0].self_attn.scaling == 0.25\n"
        "    assert not model.training and model.config.rms_norm_eps == 1e-6\n"
        "    g = torch.Generator().manual_seed(ctx.seed)\n"
        "    assert torch.equal(torch.rand(3), torch.rand(3, generator=g))   # global RNG reseeded per item\n"
        "    model.stash = 1\n    model.model.layers[0].self_attn.scaling = 5.0\n    model.config.rms_norm_eps = 1.0\n"
        "    model.train()\n"
        "    return {'weights': {}, 'doc_in_context': False}\n")
    _, per, _ = _run(None, small, loader=lambda: tttlib.fresh_import("surf_probe"))
    assert "surf_helper" not in sys.modules and "surf_probe" not in sys.modules
    assert all(r["reset_ok"] and not r["doc_in_context"] and r["changed_tensors"] == 0 for r in per.values())


@pytest.mark.parametrize("ret", [None, {"weights": {}}, {"weights": {}, "doc_in_context": 1},
                                 {"weights": {}, "doc_in_context": np.bool_(True)},
                                 {"weights": {}, "doc_in_context": True, "cache": None},
                                 {"weights": [], "doc_in_context": True},
                                 {"weights": {"no.such.weight": torch.zeros(1)}, "doc_in_context": True},
                                 {"weights": {"model.norm.weight": torch.zeros(3)}, "doc_in_context": True},
                                 {"weights": {"model.norm.weight": torch.full((64,), float("nan"))}, "doc_in_context": True}])
def test_bad_return_values_are_refused(small, ret):
    with pytest.raises((TypeError, KeyError, ValueError)):
        _run(lambda work, ctx: ret, small, n=1)


@pytest.mark.parametrize("how", ["replace_module", "monkeypatch", "thread", "answer_model"])
def test_tampering_is_refused_or_flagged(small, how):
    model = tiny_model()

    def bad(work, ctx):
        attn = work.model.layers[0].self_attn
        if how == "replace_module":
            attn.q_proj = torch.nn.Sequential(attn.q_proj)   # a wrapper module left in place
        elif how == "monkeypatch":
            attn.forward = type(attn).forward.__get__(attn)   # same behaviour, but an instance attribute
        elif how == "thread":
            threading.Thread(target=time.sleep, args=(2,), daemon=True).start()
        else:  # reach the answering model some other way and change it
            victim = next(o for o in gc.get_objects() if o is model)
            with torch.no_grad():
                victim.model.norm.weight.add_(1.0)
        return {"weights": {}, "doc_in_context": True}

    if how == "answer_model":
        _, per, _ = _run(bad, small, n=1, model=model)
        assert not next(iter(per.values()))["reset_ok"]        # -> the evaluator marks the run invalid
    else:
        with pytest.raises(RuntimeError, match="surface"):
            _run(bad, small, n=1, model=model)


def test_budget_includes_import_and_return_validation(small):
    def slow_adapt(work, ctx):
        assert ctx.time_left() < 0.25
        time.sleep(0.2)
        return {"weights": {}, "doc_in_context": True}

    def slow_import():
        time.sleep(0.15)
        return types.SimpleNamespace(adapt=slow_adapt)

    _, per, _ = _run(None, small, n=2, ttt_seconds=0.3, loader=slow_import)
    assert all(r["ttt_s"] > 0.35 for r in per.values())


def test_reference_arms():
    for name, doc in (("ref_no_ttt", True), ("ref_no_doc", False)):
        mod = types.ModuleType(name)
        exec(compile(open(f"/frozen/{name}/ttt.py").read(), name, "exec"), mod.__dict__)
        assert mod.adapt(None, None) == {"weights": {}, "doc_in_context": doc}


def test_harness_and_evaluator_end_to_end(tmp_path):
    """harness.py + evaluate.py on 2 real validation items with a tiny saved model on the CPU."""
    model_dir = tmp_path / "tiny"
    tiny_model().save_pretrained(model_dir)
    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run([sys.executable, "/frozen/harness.py", "--out", str(out), "--seed", "1", "--split", "validation",
                        "--ttt-seconds", "60", "--device", "cpu", "--limit", "2", "--data", f"{D}/validation",
                        "--model", str(model_dir)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    per = _json(out / "items.json")
    assert len(per) == 2 and all(v["reset_ok"] and v["changed_tensors"] == 2 for v in per.values())
    assert _json(out / "budget.json")["ttt_seconds"] == pytest.approx(sum(v["ttt_s"] for v in per.values()))
    res = tmp_path / "m.json"
    subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(out), "--out", str(res), "--data", f"{D}/validation",
                    "--ttt-seconds", "60", "--limit", "2"], check=True, capture_output=True)
    m = _json(res)
    assert m["valid"] and len(m["items"]) == 2 and 0.0 <= m["primary"] <= 1.0
