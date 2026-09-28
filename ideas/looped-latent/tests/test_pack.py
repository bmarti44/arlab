"""Deterministic CPU checks for looped-latent (run by `arlab check` as root, no GPU, data at /data, surface at /work)."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "/work")
import loop as surface  # noqa: E402  (the baseline surface at TESTS time)
import progen  # noqa: E402
from arlab.lib.lm import causality_check  # noqa: E402
from common import IM_END, MAX_NEW, MODEL_DIR, LayerCounter, greedy_decode, token_nll  # noqa: E402

D = "/data"
ASSISTANT_HEADER = [151644, 77091, 198, 151667, 271, 151668, 271]  # <|im_start|>assistant\n<think>\n\n</think>\n\n


# ---------------------------------------------------------------- generator and splits
def test_generator_deterministic_and_correct():
    a, b = progen.generate(123, 300, (2, 8)), progen.generate(123, 300, (2, 8))
    assert a == b and a != progen.generate(124, 300, (2, 8))
    for p in a + progen.generate(5, 100, (9, 12)):
        assert p["answer"] == progen.run_program(p["code"]) and 0 <= int(p["answer"]) <= 99
        assert p["code"].splitlines()[-1].startswith("print(")
    assert {p["steps"] for p in a} == set(range(2, 9))


def _json(p):
    return json.load(open(p))


def test_splits_sized_and_disjoint():
    sp, info = _json(f"{D}/splits.json"), _json(f"{D}/info.json")
    assert info["difficulty"] == progen.DIFFICULTY, "data was prepared with other generator knobs"
    hashes, prompts = {}, {}
    for split in ("validation", "holdout"):
        pub, priv = _json(f"{D}/{split}/public/items.json"), _json(f"{D}/{split}/private/answers.json")
        assert len(pub["main"]) == len(priv["main"]) == sp[split]
        assert [it["id"] for it in pub["main"]] == list(priv["main"])
        for kind, (lo, hi) in (("main", progen.ID_STEPS), ("hard", progen.HARD_STEPS)):
            assert all(lo <= g["steps"] <= hi for g in priv[kind].values())
            hashes[split, kind] = {progen.code_hash(g["code"]) for g in priv[kind].values()}
            prompts[split, kind] = {tuple(it["prompt"]) for it in pub[kind]}
            assert len(hashes[split, kind]) == len(priv[kind])          # no duplicates inside a set
    keys = list(hashes)
    for i, k in enumerate(keys):
        for k2 in keys[i + 1:]:
            assert not hashes[k] & hashes[k2] and not prompts[k] & prompts[k2], (k, k2)
    tok, off, pl = (np.load(f"{D}/train/{f}.npy") for f in ("tokens", "offsets", "prompt_len"))
    train = {tuple(tok[off[i]:off[i] + pl[i]]) for i in range(len(pl))}
    assert len(train) == len(pl)
    for k in keys:
        assert not train & prompts[k], f"train overlaps {k}"
    t = np.load(f"{D}/validation/public/text.npy")
    assert not {r.tobytes() for r in t} & {r.tobytes() for r in np.load(f"{D}/holdout/public/text.npy")}


# ---------------------------------------------------------------- the surface cannot see private labels
def test_public_data_has_no_answers():
    for split in ("validation", "holdout"):
        pub = _json(f"{D}/{split}/public/items.json")
        for kind in ("main", "hard"):
            for it in pub[kind]:
                assert set(it) == {"id", "prompt"}
                assert it["prompt"][-len(ASSISTANT_HEADER):] == ASSISTANT_HEADER  # prompt stops before the answer
    src = open("/frozen/harness.py").read()
    assert "private" not in src and "/eval" not in src and "holdout" not in src
    # train targets are the answer tokens + <|im_end|> only
    tok, off, pl = (np.load(f"{D}/train/{f}.npy") for f in ("tokens", "offsets", "prompt_len"))
    for i in range(50):
        ans = tok[off[i] + pl[i]:off[i + 1]]
        assert ans[-1] == IM_END and 2 <= len(ans) <= 3


# ---------------------------------------------------------------- scorer and guard metrics (frozen EVALUATE)
@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _evaluate(tmp_path, preds, text_nll=None, base_nll=None, **stats):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    n = np.load(f"{D}/validation/public/text.npy").shape[0]
    base_nll = np.full((n, 16), 2.0, np.float32) if base_nll is None else base_nll
    np.save(run / "text_nll.npy", base_nll if text_nll is None else text_nll)
    np.save(run / "base_nll.npy", base_nll)
    st = {"train_steps": 10, "examples_seen": 320, "gen_s": 1.0, "depth_ratio": 1.0, "trainable_m": 10.0,
          "nan_at": None, "nonfinite_decode_calls": 0, "causal": True, "causal_max_diff": 0.0} | stats
    json.dump(st, open(run / "stats.json", "w"))
    json.dump(preds, open(run / "preds.json", "w"))
    out = tmp_path / "m.json"
    subprocess.run([sys.executable, "/eval/evaluate.py", "--run", str(run), "--out", str(out),
                    "--data", f"{D}/validation"], check=True, capture_output=True)
    return json.load(open(out))


def _gold():
    return _json(f"{D}/validation/private/answers.json")


def _enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def test_scorer_exact_match(tmp_path, tokenizer):
    g = _gold()
    preds = {k: {i: _enc(tokenizer, v["answer"] + "<|im_end|>") for i, v in g[k].items()} for k in ("main", "hard")}
    m = _evaluate(tmp_path, preds)
    assert m["valid"] and m["primary"] == 1.0 and m["metrics"]["hard_acc"] == 1.0
    assert len(m["items"]) == len(g["main"]) and set(m["items"].values()) == {1.0}
    ids = list(g["main"])
    ans = {i: g["main"][i]["answer"] for i in ids}
    other = lambda a: str((int(a) + 1) % 100)  # noqa: E731
    cases = {ids[0]: (" " + ans[ids[0]] + "\n<|im_end|>", 1.0),       # surrounding whitespace is fine
             ids[1]: (ans[ids[1]], 1.0),                              # no stop token within MAX_NEW: judged as is
             ids[2]: (f"{ans[ids[2]]} or {other(ans[ids[2]])}", 0.0),  # hedged
             ids[3]: (other(ans[ids[3]]) + "<|im_end|>", 0.0),         # wrong
             ids[4]: (ans[ids[4]] + ".<|im_end|>", 0.0),               # extra punctuation
             ids[5]: ("", 0.0)}                                       # nothing
    for i, (s, _) in cases.items():
        preds["main"][i] = _enc(tokenizer, s)[:MAX_NEW]
    m = _evaluate(tmp_path, preds)
    assert m["valid"]
    for i, (_, want) in cases.items():
        assert m["items"][i] == want, (i, cases[i])
    assert abs(m["primary"] - (len(ids) - 4) / len(ids)) < 1e-12


def test_scorer_rejects_bad_outputs(tmp_path, tokenizer):
    g = _gold()
    good = {k: {i: [15] for i in g[k]} for k in ("main", "hard")}
    assert _evaluate(tmp_path, good)["valid"]
    missing = {"main": dict(list(good["main"].items())[1:]), "hard": good["hard"]}
    assert not _evaluate(tmp_path, missing)["valid"]
    long = {"main": {**good["main"], next(iter(g["main"])): [15] * (MAX_NEW + 1)}, "hard": good["hard"]}
    assert not _evaluate(tmp_path, long)["valid"]
    n = np.load(f"{D}/validation/public/text.npy").shape[0]
    assert not _evaluate(tmp_path, good, text_nll=np.full((n, 16), np.nan, np.float32))["valid"]
    assert not _evaluate(tmp_path, good, causal=False)["valid"]
    assert not _evaluate(tmp_path, good, nan_at=7)["valid"]
    assert not _evaluate(tmp_path, good, nonfinite_decode_calls=1)["valid"]


def test_guard_metrics(tmp_path):
    g = _gold()
    n = np.load(f"{D}/validation/public/text.npy").shape[0]
    base = np.random.default_rng(0).uniform(1, 3, (n, 16)).astype(np.float32)
    preds = {k: {i: [15] for i in g[k]} for k in ("main", "hard")}
    m = _evaluate(tmp_path, preds, text_nll=base * 1.03, base_nll=base, depth_ratio=1.4286, trainable_m=11.2)["metrics"]
    assert abs(m["text_nll_vs_base"] - 1.03) < 1e-5 and abs(m["text_nll"] - 1.03 * base.mean()) < 1e-4
    assert m["depth_ratio"] == 1.4286 and m["trainable_m"] == 11.2
    assert {f"acc_s{k}" for k in range(2, 9)} <= set(m)


def test_token_nll_uniform_and_greedy_decode():
    V = 200_000
    rows = np.random.default_rng(0).integers(0, 1000, (3, 9))
    nll = token_nll(lambda x: torch.zeros(*x.shape, V), rows, 2, "cpu")
    assert nll.shape == (3, 8) and np.allclose(nll, math.log(V), atol=1e-4)

    def next_is_plus_one(x):  # next token = last token + 1; after 100 comes <|im_end|>; pads must never be read
        lg = torch.full((*x.shape, V), -1e4)
        nxt = torch.where(x == 100, torch.full_like(x, IM_END), x + 1)
        lg.scatter_(-1, nxt.unsqueeze(-1), 0.0)
        return lg

    out, bad = greedy_decode(next_is_plus_one, [[5, 6, 7], [98], [1, 2, 3, 4, 5, 6, 99]], batch=2, device="cpu")
    assert bad == 0
    assert out[0] == [8, 9, 10, 11, 12, 13][:MAX_NEW]
    assert out[1] == [99, 100, IM_END] and out[2] == [100, IM_END]


# ---------------------------------------------------------------- the surface: identity at init, loop trainable
def tiny_base(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=28, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def test_identity_at_init_any_k():
    base = tiny_base()
    ids = torch.randint(0, 512, (3, 20))
    with torch.no_grad():
        ref = base(input_ids=ids, use_cache=False).logits
    state = surface.build(base, {"device": "cpu", "seed": 0, "dtype": torch.float32})
    with torch.no_grad():
        for k in (1, 2, 4, surface.K_MAX):
            assert (state["model"](ids, k=k) - ref).abs().max().item() < 1e-5, k
    assert (surface.logits(state, ids) - ref).abs().max().item() < 1e-5


def test_baseline_is_the_no_loop_arm_and_depth_is_counted():
    assert tuple(surface.LOOPS_TRAIN) == (1,) and surface.LOOPS_EVAL == 1
    base = tiny_base()
    state = surface.build(base, {"device": "cpu", "seed": 0, "dtype": torch.float32})
    ids = torch.randint(0, 512, (2, 12))
    for k, want in ((1, 28), (4, 28 + 3 * (surface.LOOP_END - surface.LOOP_START))):
        c = LayerCounter(base.model.layers)
        with torch.no_grad():
            state["model"](ids, k=k)
        c.remove()
        assert c.calls == want, (k, c.calls)
    rows = np.random.default_rng(1).integers(0, 512, (4, 33))
    assert causality_check(lambda x: surface.logits(state, x), rows, 512, device="cpu")["causal"]


def test_loop_gates_learn_from_identity(monkeypatch):
    monkeypatch.setattr(surface, "LOOPS_TRAIN", (3,))
    base = tiny_base()
    state = surface.build(base, {"device": "cpu", "seed": 0, "dtype": torch.float32})
    ids = torch.randint(0, 512, (4, 16))
    tgt = torch.full_like(ids, -100)
    tgt[:, -3:] = ids[:, -3:]
    loss = surface.train_step(state, ids, tgt, 0.5)
    assert math.isfinite(loss)
    assert state["model"].gates[:2].abs().sum() > 0 and state["model"].gates[2:].abs().sum() == 0
    trainable = [n for n, p in state["model"].named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B", "gates", "inj.weight")) for n in trainable)  # base stays frozen
