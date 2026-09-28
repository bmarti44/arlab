"""Deterministic CPU checks for latent-arch (run by `arlab check` as root, no GPU; data at /data, surface at /work,
the sealed pack at /pack, frozen/run at /frozen, frozen/eval at /eval). The generator is read from
/pack/frozen/prepare because RUN never sees it."""
import json
import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

D = os.environ.get("LA_DATA", "/data")          # overrides only for running these tests outside the container
FROZEN = os.environ.get("LA_FROZEN", "/frozen")
EVAL = os.environ.get("LA_EVAL", "/eval")
WORK = os.environ.get("LA_WORK", "/work")
sys.path.insert(0, os.environ.get("LA_PREPARE", "/pack/frozen/prepare"))
sys.path.insert(0, EVAL)
import progen  # noqa: E402
import evalcore  # noqa: E402
from common import VOCAB, file_sha256, tensor_hash  # noqa: E402

L = 1 + progen.n_pieces()  # prompt tokens: BOS + statements + `q?`


def _npz(split):
    return dict(np.load(f"{D}/{split}/private/programs.npz"))


@pytest.fixture(scope="module")
def info():
    return json.load(open(f"{D}/info.json"))


@pytest.fixture(scope="module")
def codec(info):
    """Maps relabelled token ids back to progen pieces (via the secret permutation, audit-only)."""
    enc = pickle.load(open(f"{D}/audit/tokenizer.pkl", "rb"))
    perm = np.load(f"{D}/audit/vocab_perm.npy")
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    fb = info["piece_fallback"]
    orig = {p: enc.encode_single_token(fb[p]) if p in fb else enc.encode_ordinary(p)[0] for p in progen.PIECES}
    to_piece = {int(perm[t]): p for p, t in orig.items()}
    return {"enc": enc, "perm": perm, "inv": inv, "to_piece": to_piece, "bos": int(perm[enc.encode_single_token("<|reserved_0|>")])}


def _text(codec, row):
    return "".join(codec["to_piece"][int(t)] for t in row)


# ---------------------------------------------------------------- generator
def test_generator_deterministic_depth_and_answers():
    a, b = progen.generate(7, 600, (1, 12), balanced=True), progen.generate(7, 600, (1, 12), balanced=True)
    assert a == b and a != progen.generate(8, 600, (1, 12), balanced=True)
    assert [p["k"] for p in a[:12]] == list(range(1, 13))
    for p in a:
        q = progen.parse(progen.to_text(p))
        assert q["stmts"] == p["stmts"] and q["query"] == p["query"]
        assert progen.depths(q["stmts"])[q["query"]] == p["k"]
        assert progen.evaluate(q["stmts"])[q["query"]] == p["answer"] and 0 <= p["answer"] <= 99
        assert len(progen.pieces(p)) == progen.n_pieces() and len(p["stmts"]) == progen.DIFFICULTY["n_stmt"]
        assert sum(s[1] == "c" for s in p["stmts"]) == progen.DIFFICULTY["n_const"]
        assert len({s[0] for s in p["stmts"]}) == len(p["stmts"])          # single assignment


def test_depth_on_hand_made_programs():
    cases = {"a=5;a?": (0, 5), "a=5;b=a+3;c=b-9;c?": (2, 99), "a=1;b=a+1;c=b+a;c?": (2, 3),
             "x=7;a=5;b=a+x;c=b+x;d=x+1;c?": (2, 19), "x=7;a=5;d=x+1;b=a+x;c=b+x;d?": (1, 8),
             "a=50;b=a+1;c=b+1;d=c+1;e=a-d;e?": (4, 97), "a=2;z=9;y=z-1;x=y+a;w=x-a;w?": (3, 8)}
    for text, (k, ans) in cases.items():
        p = progen.parse(text)
        assert progen.depths(p["stmts"])[p["query"]] == k, text
        assert progen.evaluate(p["stmts"])[p["query"]] == ans, text


def test_normalized_hash_is_alpha_invariant():
    p = progen.parse("a=5;b=a+3;c=b-9;c?")
    assert progen.norm_hash(p) == progen.norm_hash(progen.parse("x=5;q=x+3;m=q-9;m?"))
    assert progen.norm_hash(p) != progen.norm_hash(progen.parse("a=5;b=a+3;c=b-8;c?"))


# ---------------------------------------------------------------- data: splits, depth labels, disjointness
def test_splits_sized_and_depth_ranges(info):
    sp = json.load(open(f"{D}/splits.json"))
    assert info["difficulty"] == progen.DIFFICULTY, "data prepared with other generator knobs"
    n = info["n_eval"]
    for split in ("validation", "holdout"):
        d = _npz(split)
        assert sp[split] == n["id"] + n["depth"] == 4000
        assert d["tokens"].shape == (len(d["group"]), L)
        for g, (lo, hi), cnt in ((0, progen.ID_K, n["id"]), (1, progen.DEPTH_K, n["depth"]), (2, progen.EXT_K, n["ext"])):
            k = d["k"][d["group"] == g]
            assert len(k) == cnt and k.min() == lo and k.max() == hi
            assert np.bincount(k)[lo:hi + 1].max() - np.bincount(k)[lo:hi + 1].min() <= 1   # balanced
        cf = d["group"] == 3
        assert cf.sum() == n["cf"] and (d["group"][d["cf_of"][cf]] < 2).all()
    tk = np.load(f"{D}/audit/train_k.npy")
    assert len(tk) == info["n_train_programs"] and tk.min() == progen.TRAIN_K[0] and tk.max() == progen.TRAIN_K[1]


def test_eval_programs_decode_to_their_labels(codec):
    for split in ("validation", "holdout"):
        d = _npz(split)
        for i in range(0, len(d["group"]), 7):
            row = d["tokens"][i]
            assert row[0] == codec["bos"]
            text = _text(codec, row[1:])
            assert text == str(d["text"][i])
            p = progen.parse(text)
            assert progen.depths(p["stmts"])[p["query"]] == d["k"][i]
            assert progen.evaluate(p["stmts"])[p["query"]] == d["value"][i]
            assert codec["to_piece"][int(d["answer"][i])] == str(d["value"][i])   # one-token answer
            assert progen.norm_hash(p) == int(d["hash"][i])


def test_splits_hash_disjoint_and_train_never_contains_eval(codec):
    hashes, prompts = [], set()
    for split in ("validation", "holdout"):
        d = _npz(split)
        hashes.append(set(d["hash"].tolist()))
        assert len(hashes[-1]) == len(d["hash"])                  # unique within the split
        prompts |= {r.tobytes() for r in d["tokens"].astype(np.uint16)}
    assert not hashes[0] & hashes[1]
    progs = np.memmap(f"{D}/train/programs.bin", dtype=np.uint16, mode="r")
    starts = np.load(f"{D}/train/program_starts.npy")
    stride = int(starts[1] - starts[0])
    assert stride == L + 1 and len(progs) == len(starts) * stride
    train = np.asarray(progs).reshape(-1, stride)
    assert (train[:, 0] == codec["bos"]).all()
    assert not {r.tobytes() for r in train[:, :L]} & prompts        # exact prompts: none shared
    tk = np.load(f"{D}/audit/train_k.npy")
    ev = hashes[0] | hashes[1]
    for j in np.random.default_rng(0).choice(len(train), 20_000, replace=False):   # normalized hash on a sample
        text = _text(codec, train[j, 1:L])
        p = progen.parse(text)
        assert progen.norm_hash(p) not in ev
        assert progen.depths(p["stmts"])[p["query"]] == tk[j] <= progen.TRAIN_K[1]
        assert codec["to_piece"][int(train[j, L])] == str(progen.evaluate(p["stmts"])[p["query"]])


def test_relabelling_is_a_consistent_permutation(codec):
    perm, enc = codec["perm"], codec["enc"]
    assert sorted(perm.tolist()) == list(range(VOCAB)) and (perm != np.arange(VOCAB)).mean() > 0.99
    tb0 = np.array([0 if enc.decode([i]).startswith("<|reserved_") else len(enc.decode_single_token_bytes(i))
                    for i in range(VOCAB)])
    for split in ("validation", "holdout"):
        tb = np.load(f"{D}/{split}/private/token_bytes.npy")
        assert (tb[perm] == tb0).all()
        rows = np.load(f"{D}/{split}/private/text_rows.npy")
        assert rows.shape == (2048, 1025)
        assert rows[0, 0] == codec["bos"] and (rows == codec["bos"]).any(axis=1).mean() > 0.2   # BOS-aligned docs
        txt = enc.decode(codec["inv"][rows[0, :200].astype(np.int64)].tolist())
        assert sum(c.isalpha() or c == " " for c in txt) > 0.6 * len(txt)   # un-relabelled it is ordinary text
    stream = np.memmap(f"{D}/train/tokens.bin", dtype=np.uint16, mode="r")
    assert len(stream) >= 200_000_000
    txt = enc.decode(codec["inv"][np.asarray(stream[1000:1300]).astype(np.int64)].tolist())
    assert sum(c.isalpha() or c == " " for c in txt) > 0.6 * len(txt)
    v, h = (np.load(f"{D}/{s}/private/text_rows.npy") for s in ("validation", "holdout"))
    assert not {r.tobytes() for r in v[:, :64]} & {r.tobytes() for r in h[:, :64]}


def test_heuristic_floors_are_low(info):
    for split in ("validation", "holdout"):
        d = _npz(split)
        main = d["group"] < 2
        val = d["value"][main]
        for h in ("h_last_const", "h_own_const", "h_root"):
            assert (d[h][main] == val).mean() <= 0.10, (split, h)
        assert np.bincount(val).max() / len(val) <= 0.10
        assert info["floors"][split]["modal"] <= 0.10


# ---------------------------------------------------------------- RUN cannot see eval programs or the generator
def test_run_cannot_see_private_data_or_generator():
    assert sorted(os.listdir(f"{D}/train")) == ["program_starts.npy", "programs.bin", "tokens.bin"]
    for split in ("validation", "holdout"):
        assert sorted(os.listdir(f"{D}/{split}")) == ["private"]      # no public/: RUN gets no eval inputs at all
    files = sorted(os.listdir(FROZEN))
    assert not {"progen.py", "prepare.py"} & set(files), files
    for root, _, fs in os.walk(FROZEN):
        for f in fs:
            if f.endswith(".py"):
                src = open(os.path.join(root, f)).read()
                assert "private" not in src and "progen" not in src and "audit" not in src, f
    h = open(f"{FROZEN}/harness.py").read()
    assert h.index("t0 = time.time()") < h.index("import train as surface")   # timer starts before the import


# ---------------------------------------------------------------- harness: budget and timer (CPU smoke config)
def _harness(tmp_path, work, seconds, name="out"):
    out = tmp_path / name
    out.mkdir()
    r = subprocess.run([sys.executable, f"{FROZEN}/harness.py", "--out", str(out), "--seed", "1", "--split", "validation",
                        "--train-seconds", str(seconds), "--data", f"{D}/train", "--work", str(work), "--smoke"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return out, json.load(open(out / "budget.json")), json.load(open(out / "stats.json"))


def _evaluate(tmp_path, run, work, limit=24, name="m.json"):
    out = tmp_path / name
    r = subprocess.run([sys.executable, f"{EVAL}/evaluate.py", "--run", str(run), "--out", str(out), "--data",
                        f"{D}/validation/private", "--work", str(work), "--device", "cpu", "--limit", str(limit)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(out))


def test_baseline_surface_smoke_run_and_evaluate(tmp_path):
    out, budget, stats = _harness(tmp_path, WORK, 8)
    assert 8 <= budget["train_seconds"] <= 8 + 30 and stats["train_steps"] >= 1 and stats["nan_at"] is None
    m = _evaluate(tmp_path, out, WORK)
    assert m["valid"], m["message"]
    mm = m["metrics"]
    assert len(m["items"]) == 48 and set(m["items"].values()) <= {0.0, 1.0}
    assert abs(m["primary"] - 0.5 * (mm["acc_id"] + mm["acc_depth"])) < 1e-12
    assert 25 < mm["params_m"] < 28 and mm["infer_flops_tok"] > 2e7 and mm["val_bpb"] > 0
    with open(out / "model.pt", "ab") as f:          # any change to the checkpoint after the deadline
        f.write(b"\0")
    assert not _evaluate(tmp_path, out, WORK, name="t.json")["valid"]


SLOW = """
import time
import torch
time.sleep({import_s})
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, idx):
        return torch.zeros(*idx.shape, 8192) + self.w
def build(config):
    return {{"model": M()}}
def train_step(state, batch, step, progress):
    time.sleep({step_s})
    return torch.tensor(1.0)
def save(state, path):
    torch.save(state["model"].state_dict(), path)
def load(path, device):
    m = M()
    m.load_state_dict(torch.load(path, weights_only=True))
    return m
"""


def test_timer_counts_import_and_stops_slow_steps(tmp_path):
    w = tmp_path / "w1"
    w.mkdir()
    (w / "train.py").write_text(SLOW.format(import_s=0, step_s=1.0))
    _, budget, stats = _harness(tmp_path, w, 3.5, "o1")
    assert stats["train_steps"] == 4 and 4.0 <= budget["train_seconds"] < 4.9   # steps start at ~0,1,2,3; none after 3.5
    w2 = tmp_path / "w2"
    w2.mkdir()
    (w2 / "train.py").write_text(SLOW.format(import_s=3, step_s=0.2))
    _, budget, stats = _harness(tmp_path, w2, 2, "o2")
    assert stats["train_steps"] == 0 and budget["train_seconds"] >= 3        # the import alone used the budget


def test_evaluator_rejects_overrun_budget(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    (w / "train.py").write_text(SLOW.format(import_s=0, step_s=0.0))
    out, _, _ = _harness(tmp_path, w, 1)
    assert _evaluate(tmp_path, out, w, limit=8)["valid"]
    json.dump({"train_seconds": 361.0}, open(out / "budget.json", "w"))
    m = _evaluate(tmp_path, out, w, limit=8, name="m2.json")
    assert not m["valid"] and "train_seconds" in m["message"]


# ---------------------------------------------------------------- evaluator on hand-made models
ORACLE = """
import numpy as np
import torch
d = np.load("{data}/validation/private/programs.npz")
L = d["tokens"].shape[1]
DEF = int(d["tokens"][0, 0])  # BOS: never an answer
TABLE = {{tuple(r[:L - {shift}].tolist()): int(a) for r, a in zip(d["tokens"], d["answer"])}}
class Oracle(torch.nn.Module):
    '''Causal lookup: when the last {n} tokens are an eval prompt (minus {shift}), put logit 10 on its answer
    (and {decoy} on BOS; everything else defaults to BOS).'''
    def forward(self, idx):
        out = torch.zeros(*idx.shape, 8192)
        out[..., DEF] = 1.0
        rows = idx.tolist()
        n = L - {shift}
        for b, r in enumerate(rows):
            for t in range(n - 1, len(r)):
                a = TABLE.get(tuple(r[t - n + 1:t + 1]))
                if a is not None:
                    out[b, t, a] = 10.0
                    out[b, t, DEF] = {decoy}
        return out
def load(path, device):
    return Oracle()
"""
PEEK = """
import torch
class Peek(torch.nn.Module):
    def forward(self, idx):
        z = torch.zeros(*idx.shape, 8192)
        z[..., 0] = idx.float().mean(1, keepdim=True)
        return z
def load(path, device):
    return Peek()
"""


def _fake_run(tmp_path, name, src, model):
    w = tmp_path / f"w_{name}"
    w.mkdir()
    (w / "train.py").write_text(src)
    run = tmp_path / f"run_{name}"
    run.mkdir()
    torch.save({}, run / "model.pt")
    json.dump({"train_seconds": 330.0}, open(run / "budget.json", "w"))
    json.dump({"nan_at": None, "model_sha256": file_sha256(str(run / "model.pt")), "tensor_hash": tensor_hash(model)},
              open(run / "stats.json", "w"))
    return _evaluate(tmp_path, run, w, limit=40, name=f"{name}.json")


def test_evaluator_scores_the_answer_position_only(tmp_path):
    m = _fake_run(tmp_path, "oracle", ORACLE.format(shift=0, n="L", decoy=0.0, data=D), torch.nn.Module())
    assert m["valid"] and m["primary"] == 1.0 and m["metrics"]["cf_both"] == 1.0 and m["metrics"]["acc_ext"] == 1.0
    assert m["metrics"]["infer_flops_tok"] == 0 and m["metrics"]["params_m"] == 0
    early = _fake_run(tmp_path, "early", ORACLE.format(shift=1, n="L-1", decoy=0.0, data=D), torch.nn.Module())
    assert early["valid"] and early["primary"] == 0.0        # answer one position early (after `q`, before `?`)
    decoy = _fake_run(tmp_path, "decoy", ORACLE.format(shift=0, n="L", decoy=20.0, data=D), torch.nn.Module())
    assert decoy["valid"] and decoy["primary"] == 0.0        # argmax over the FULL vocabulary
    peek = _fake_run(tmp_path, "peek", PEEK, torch.nn.Module())
    assert not peek["valid"] and "causal" in peek["message"]


class Looped(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        torch.manual_seed(0)
        self.k, self.emb, self.core = k, torch.nn.Embedding(VOCAB, 16), torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, VOCAB)

    def forward(self, idx):
        h = self.emb(idx)
        for _ in range(self.k):
            h = h + self.core(h)
        return self.head(h)


def test_flop_counter_counts_loops_and_flags_custom_ops():
    x = [np.random.default_rng(0).integers(0, VOCAB, (2, 32))]
    f1, f2, f3 = (evalcore.probe(Looped(k), x, "cpu")["flops_per_token"] for k in (1, 2, 3))
    core = 2 * 16 * 16
    assert f2 - f1 == core and f3 - f2 == core and f1 == core + 2 * 16 * VOCAB
    assert not evalcore.probe(Looped(1), x, "cpu")["bad_ops"]

    @torch.library.custom_op("latentarch_test::double", mutates_args=())
    def double(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    @double.register_fake
    def _(t):
        return torch.empty_like(t)

    class Custom(Looped):
        def forward(self, idx):
            return double(super().forward(idx))

    assert any("latentarch_test" in op for op in evalcore.probe(Custom(1), x, "cpu")["bad_ops"])
    q = torch.randn(1, 2, 16, 8)

    class Attn(torch.nn.Module):   # SDPA on CPU is counted like the CUDA kernels
        def forward(self, idx):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            return torch.zeros(*idx.shape, VOCAB)

    assert evalcore.probe(Attn(), x, "cpu")["flops"] == 4 * 1 * 2 * 16 * 16 * 8
