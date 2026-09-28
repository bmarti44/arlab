"""Deterministic CPU checks for latent-arch (run by `arlab check` as root, no GPU; data at /data, surface at /work,
the sealed pack at /pack, frozen/run at /frozen, frozen/eval at /eval). The generator is read from
/pack/frozen/prepare because RUN never sees it. Fake surfaces are written to tmp dirs and run through the real
supervisor (--smoke, CPU) and the real evaluator (--device cpu --limit N)."""
import json
import os
import pickle
import subprocess
import sys
import time

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
import meter  # noqa: E402
from common import VOCAB, count_bytes, count_elements, file_sha256, load_checkpoint, tensors_hash  # noqa: E402

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


def _chain(p):
    st = {s[0]: s for s in p["stmts"]}
    out = [p["query"]]
    while st[out[-1]][1] != "c":
        out.append(st[out[-1]][2])
    return set(out)


def test_training_programs_have_no_statement_deeper_than_six():
    import random
    rng = random.Random(0)
    for _ in range(50_000):
        p = progen.make_program(rng, rng.randint(*progen.TRAIN_K))
        assert max(progen.depths(p["stmts"]).values()) <= progen.TRAIN_K[1]
    for _ in range(10_000):                  # every split: only the queried chain may be deeper than 6
        p = progen.make_program(rng, rng.randint(1, 12))
        dep, ch = progen.depths(p["stmts"]), _chain(p)
        assert max(v for x, v in dep.items() if x not in ch) <= progen.DIFFICULTY["distractor_max_depth"]
        assert not any(p["query"] in (s[2], s[3]) for s in p["stmts"])      # nothing reads the queried variable


def _tv(a, b):
    ca, cb = np.bincount(a, minlength=32) / len(a), np.bincount(b, minlength=32) / len(b)
    return 0.5 * np.abs(ca - cb).sum()


def _cues(p):
    st = p["stmts"]
    pos = {s[0]: i for i, s in enumerate(st)}
    root = [x for x in _chain(p) if st[pos[x]][1] == "c"][0]
    return pos[p["query"]], pos[root], int(isinstance(st[pos[p["query"]]][3], str)), \
        sum(isinstance(s[3], str) for s in st if s[1] != "c")


def test_query_position_and_format_do_not_reveal_depth():
    """Per-k histograms (k 1..10, 3000 programs each) of the queried statement's index, the chain root's index,
    whether the query reads a variable, and the number of v=u±w statements: max total-variation distance over all
    pairs of k stays at sampling-noise level. (Inherent, not matched: for k=1 the query reads a constant statement;
    deeper chains leave fewer distractors in a fixed-length program.)"""
    feats = {k: np.array([_cues(p) for p in progen.generate(500 + k, 3000, (k, k))]) for k in range(1, 11)}
    for f, lim in ((0, 0.05), (1, 0.06), (2, 0.06), (3, 0.08)):
        worst = max(_tv(feats[a][:, f], feats[b][:, f]) for a in range(1, 11) for b in range(a + 1, 11))
        assert worst <= lim, (f, worst)
    assert {int(q) for k in feats for q in feats[k][:, 0]} == set(range(progen.DIFFICULTY["query_min"], 14))


def test_query_position_matched_in_prepared_data(codec):
    by = {}
    for split in ("validation", "holdout"):
        d = _npz(split)
        for i in range(len(d["group"])):
            if d["group"][i] < 2:
                p = progen.parse(str(d["text"][i]))
                by.setdefault(int(d["group"][i]), []).append([s[0] for s in p["stmts"]].index(p["query"]))
    assert _tv(np.array(by[0]), np.array(by[1])) <= 0.04          # ID vs DEPTH, 4000 items each


def test_counterfactual_twins_change_the_answer():
    import random
    rng = random.Random(0)
    n_none = 0
    for p in progen.generate(11, 1000, (1, 10), balanced=True):
        c = progen.counterfactual(p, rng)
        if c is None:
            n_none += 1
            continue
        assert c["answer"] != p["answer"] and c["k"] == p["k"] and c["query"] == p["query"]
        diff = [(x, y) for x, y in zip(p["stmts"], c["stmts"]) if x != y]
        assert len(diff) == 1 and diff[0][0][1] == "c"                 # exactly one constant changed
    p = progen.parse("a=5;b=a+3;c=b-a;c?")                           # the root cancels: no valid twin
    assert progen.counterfactual(progen.annotate({**p, "k": 2}), rng) is None
    assert n_none < 400


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
        orig = d["cf_of"][cf]
        assert cf.sum() == n["cf"] and (d["group"][orig] < 2).all()
        assert (d["value"][cf] != d["value"][orig]).all()                  # every twin changes the answer
        assert np.bincount(d["k"][orig], minlength=11)[1:11].tolist() == [n["cf"] // 10] * 10   # 50 per k 1..10
        assert len(set(orig.tolist())) == len(orig) and sorted(orig.tolist()) != list(orig)   # random, not strided
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
            if d["group"][i] == 0:                                           # ID items: the training distribution
                assert max(progen.depths(p["stmts"]).values()) <= progen.TRAIN_K[1]
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
        dep = progen.depths(p["stmts"])
        assert dep[p["query"]] == tk[j] and max(dep.values()) <= progen.TRAIN_K[1]     # no statement deeper than 6
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
                assert "private" not in src and "progen" not in src and "audit/" not in src, f
    h, tr = open(f"{FROZEN}/harness.py").read(), open(f"{FROZEN}/trainer.py").read()
    assert "import train" not in h                                       # the supervisor never imports the surface
    assert tr.index("go = json.loads") < tr.index("import train as surface")   # clock runs before the surface import


def test_checkpoint_loader_accepts_only_flat_tensor_dicts(tmp_path):
    good = {"w": torch.zeros(3), "idx": torch.arange(5), "flag": torch.ones(2, dtype=torch.bool)}
    torch.save(good, tmp_path / "g.pt")
    ck = load_checkpoint(str(tmp_path / "g.pt"))
    assert count_elements(ck) == 3 + 40 + 2 and count_bytes(ck) == 12 + 40 + 2   # non-float tensors count bytes
    assert tensors_hash(ck) == tensors_hash(good)
    for i, bad in enumerate([{"w": 1.0}, {"w": {"x": torch.zeros(1)}}, [torch.zeros(1)], {1: torch.zeros(1)},
                             {"w": torch.nn.Parameter(torch.zeros(1))}]):
        torch.save(bad, tmp_path / f"b{i}.pt")
        with pytest.raises(ValueError):
            load_checkpoint(str(tmp_path / f"b{i}.pt"))
    with open(tmp_path / "p.pt", "wb") as f:
        pickle.dump(Looped, f)                                           # arbitrary pickles never unpickle
    with pytest.raises(ValueError):
        load_checkpoint(str(tmp_path / "p.pt"))


# ---------------------------------------------------------------- RUN: trusted supervisor, budget, checkpoint
def _harness(tmp_path, work, seconds, name="out", grace=None, ok=True):
    out = tmp_path / name
    out.mkdir()
    cmd = [sys.executable, f"{FROZEN}/harness.py", "--out", str(out), "--seed", "1", "--split", "validation",
           "--train-seconds", str(seconds), "--data", f"{D}/train", "--work", str(work), "--smoke"]
    r = subprocess.run(cmd + (["--grace", str(grace)] if grace is not None else []), capture_output=True, text=True,
                       timeout=600)
    assert (r.returncode == 0) == ok, r.stderr[-3000:]
    return out, json.load(open(out / "budget.json")), json.load(open(out / "stats.json"))


def _evaluate(tmp_path, run, work, limit=24, name="m.json"):
    out = tmp_path / name
    r = subprocess.run([sys.executable, f"{EVAL}/evaluate.py", "--run", str(run), "--out", str(out), "--data",
                        f"{D}/validation/private", "--work", str(work), "--device", "cpu", "--limit", str(limit)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.load(open(out))


def test_baseline_surface_smoke_run_and_evaluate(tmp_path):
    out, budget, stats = _harness(tmp_path, WORK, 8)
    assert stats["status"] == "ok" and 8 <= budget["train_seconds"] <= 8 + 30
    assert stats["trainer"]["train_steps"] >= 1 and stats["trainer"]["nan_at"] is None
    ck = load_checkpoint(str(out / "model.pt"))
    assert "cos" in ck and "sin" in ck                                   # non-persistent buffers are state too
    m = _evaluate(tmp_path, out, WORK)
    assert m["valid"], m["message"]
    mm = m["metrics"]
    assert len(m["items"]) == 48 and set(m["items"].values()) <= {0.0, 1.0}
    assert abs(m["primary"] - 0.5 * (mm["acc_id"] + mm["acc_depth"])) < 1e-12
    assert mm["params_m"] == count_elements(ck) / 1e6 and 26 < mm["params_m"] < 27
    assert mm["params_bytes"] == count_bytes(ck) / 1e6 and mm["causal_cuts"] == 32 and mm["causal_max_diff"] < 1e-3
    assert mm["flops_tok_text"] > 2e7 and mm["flops_tok_prog"] > 2e7 and mm["val_bpb"] > 0
    with open(out / "model.pt", "ab") as f:                              # any change to the checkpoint after the deadline
        f.write(b"\0")
    assert not _evaluate(tmp_path, out, WORK, name="t.json")["valid"]


FAKE = """
import os, sys, time
import torch
time.sleep({import_s})
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("table", torch.zeros({n_int}, dtype=torch.int64))
    def forward(self, idx):
        return torch.zeros(*idx.shape, 8192) + self.w + 0 * self.table[:1].float()
def build(config):
    {build}
    return {{"model": M()}}
def train_step(state, batch, step, progress):
    time.sleep({step_s})
    return torch.tensor(1.0)
def make_model(config):
    return M()
"""


def _fake(tmp_path, name, import_s=0, step_s=0.0, n_int=1, build="pass"):
    w = tmp_path / f"w_{name}"
    w.mkdir()
    (w / "train.py").write_text(FAKE.format(import_s=import_s, step_s=step_s, n_int=n_int, build=build))
    return w


def test_timer_counts_import_and_stops_slow_steps(tmp_path):
    _, budget, stats = _harness(tmp_path, _fake(tmp_path, "slow", step_s=1.0), 3.5, "o1")
    assert stats["trainer"]["train_steps"] == 4 and 4.0 <= budget["train_seconds"] < 5.0   # steps at ~0,1,2,3
    _, budget, stats = _harness(tmp_path, _fake(tmp_path, "imp", import_s=3, step_s=0.2), 2, "o2")
    assert stats["trainer"]["train_steps"] == 0 and budget["train_seconds"] >= 3        # the import used the budget


def test_supervisor_kills_a_trainer_that_ignores_the_deadline(tmp_path):
    w = _fake(tmp_path, "clock", build="time.monotonic = lambda: 0.0; time.time = lambda: 0.0")
    t = time.monotonic()
    out, budget, stats = _harness(tmp_path, w, 2, "o", grace=1.5)
    assert stats["status"] == "killed" and stats["killed"] and not (out / "model.pt").exists()
    assert 3.4 <= budget["train_seconds"] <= 4.5 and time.monotonic() - t < 60
    m = _evaluate(tmp_path, out, w, limit=8)
    assert not m["valid"] and "killed" in m["message"]


DAEMON = ("import os\n    if os.fork() == 0:\n        os.setsid()\n        if os.fork() == 0:\n"
          "            out = os.path.dirname(sys.argv[sys.argv.index('--ckpt') + 1])\n"
          "            while True:\n                open(out + '/budget.json', 'w').write('{\"train_seconds\": 0.5}')\n"
          "                time.sleep(0.05)\n        os._exit(0)\n    os.wait()")


def test_supervisor_owns_budget_json_and_kills_escaped_helpers(tmp_path):
    out, budget, stats = _harness(tmp_path, _fake(tmp_path, "daemon", build=DAEMON), 2, "o")
    assert stats["status"] == "ok" and stats["stray_processes_killed"] >= 1 and budget["train_seconds"] >= 2
    time.sleep(0.5)
    assert json.load(open(out / "budget.json"))["train_seconds"] == budget["train_seconds"]   # nothing rewrites it


def test_evaluator_rejects_overrun_budget_and_counts_all_tensor_state(tmp_path):
    w = _fake(tmp_path, "int", n_int=1_000_000)
    out, _, _ = _harness(tmp_path, w, 1)
    m = _evaluate(tmp_path, out, w, limit=8)
    assert m["valid"], m["message"]
    assert m["metrics"]["params_m"] == 8.000001                          # int64 buffer: every byte counts
    assert m["metrics"]["params_bytes"] == 8.000004
    budget = json.load(open(out / "budget.json"))
    for bad in (361.0, budget["train_seconds"] - 0.5):                   # over the limit / not the supervisor's number
        json.dump({"train_seconds": bad}, open(out / "budget.json", "w"))
        assert not _evaluate(tmp_path, out, w, limit=8, name="m2.json")["valid"]


# ---------------------------------------------------------------- EVALUATE: sandboxed worker, scoring on raw logits
ORACLE = """
import os
import numpy as np
import torch
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.npz"))
L = d["tokens"].shape[1]
DEF = int(d["tokens"][0, 0])  # BOS: never an answer
TABLE = {{tuple(r[:L - {shift}].tolist()): int(a) for r, a in zip(d["tokens"], d["answer"])}}
class Oracle(torch.nn.Module):
    # causal lookup: when the last L-{shift} tokens are an eval prompt (minus {shift}), logit 10 on its answer
    def forward(self, idx):
        out = torch.zeros(*idx.shape, 8192)
        out[..., DEF] = 1.0
        n = L - {shift}
        for b, r in enumerate(idx.tolist()):
            for t in range(n - 1, len(r)):
                a = TABLE.get(tuple(r[t - n + 1:t + 1]))
                if a is not None:
                    out[b, t, a] = 10.0
                    out[b, t, DEF] = {decoy}
        return out
def make_model(config):
    return Oracle()
"""
PEEK = """
import torch
class Peek(torch.nn.Module):
    def forward(self, idx):
        z = torch.zeros(*idx.shape, 8192)
        z[..., 0] = idx.float().mean(1, keepdim=True)
        return z
def make_model(config):
    return Peek()
"""
EVIL = """
import os, sys, json
import torch
LOOT = {{}}
def attempt(name, fn):
    try:
        fn()
        LOOT[name] = True
    except Exception as e:
        print("blocked", name, repr(e)[:120], file=sys.stderr)
def steal_answers():
    import numpy as np
    d = np.load("{data}/validation/private/programs.npz")
    LOOT["table"] = {{tuple(r.tolist()): int(a) for r, a in zip(d["tokens"], d["answer"])}}
def patch_scorer():
    import evalcore
    evalcore.accuracy_report = lambda *a, **k: {{"metrics": {{"accuracy": 1.0}}, "items": {{}}}}
def forge_result():
    open("{out}", "w").write(json.dumps({{"valid": True, "primary": 1.0, "metrics": {{}}, "items": {{}}}}))
def parent_memory():
    open("/proc/%d/mem" % os.getppid(), "rb").read(1)
def parent_environ():
    open("/proc/%d/environ" % os.getppid(), "rb").read(1)
def network():
    import socket
    s = socket.socket()
    s.settimeout(1)
    s.connect(("127.0.0.1", 9))
for name, fn in [("steal", steal_answers), ("patch", patch_scorer), ("forge", forge_result),
                 ("mem", parent_memory), ("environ", parent_environ), ("net", network)]:
    attempt(name, fn)
sys.modules["__main__"].__dict__["accuracy_report"] = None   # reaches only the worker's own __main__
class Evil(torch.nn.Module):
    def forward(self, idx):
        out = torch.zeros(*idx.shape, 8192)
        for b, r in enumerate(idx.tolist()):
            out[b, :, r[0]] = 1.0                                   # default: BOS (never an answer)
            a = LOOT.get("table", {{}}).get(tuple(r))
            if a is not None:
                out[b, -1, a] = 10.0
        return out
def make_model(config):
    assert not LOOT, "an attack succeeded: " + ",".join(LOOT)
    return Evil()
"""


def _fake_run(tmp_path, name, src, files=None, n_limit=40, ckpt=None):
    w = tmp_path / f"w_{name}"
    w.mkdir()
    (w / "train.py").write_text(src)
    for f, obj in (files or {}).items():
        np.savez(w / f, **obj)
    run = tmp_path / f"run_{name}"
    run.mkdir()
    torch.save(ckpt or {}, run / "model.pt")
    json.dump({"train_seconds": 330.0}, open(run / "budget.json", "w"))
    json.dump({"status": "ok", "train_seconds": 330.0, "seq_len": 1024, "model_sha256": file_sha256(str(run / "model.pt")),
               "tensors_hash": tensors_hash(ckpt or {}), "trainer": {"nan_at": None}}, open(run / "stats.json", "w"))
    return _evaluate(tmp_path, run, w, limit=n_limit, name=f"{name}.json")


def test_evaluator_scores_the_answer_position_only(tmp_path):
    d = _npz("validation")
    table = {"table.npz": {"tokens": d["tokens"], "answer": d["answer"]}}
    m = _fake_run(tmp_path, "oracle", ORACLE.format(shift=0, decoy=0.0), table)
    assert m["valid"], m["message"]
    assert m["primary"] == 1.0 and m["metrics"]["cf_both"] == 1.0 and m["metrics"]["acc_ext"] == 1.0
    assert m["metrics"]["params_m"] == 0 and m["metrics"]["params_bytes"] == 0
    early = _fake_run(tmp_path, "early", ORACLE.format(shift=1, decoy=0.0), table)
    assert early["valid"] and early["primary"] == 0.0        # answer one position early (after `q`, before `?`)
    decoy = _fake_run(tmp_path, "decoy", ORACLE.format(shift=0, decoy=20.0), table)
    assert decoy["valid"] and decoy["primary"] == 0.0        # argmax over the FULL vocabulary
    peek = _fake_run(tmp_path, "peek", PEEK)
    assert not peek["valid"] and "causal" in peek["message"]


PEEK_LATE = """
import torch
class PeekLate(torch.nn.Module):
    # causal up to position 768; from 769 on, position p predicts x[p+1] (the true next token)
    def forward(self, idx):
        out = torch.zeros(*idx.shape, 8192)
        T = idx.shape[1]
        if T > 770:
            p = torch.arange(769, T - 1)
            out[:, p, :] = 0.0
            out[torch.arange(idx.shape[0])[:, None], p[None, :], idx[:, p + 1]] = 30.0
        return out
def make_model(config):
    return PeekLate()
"""


def test_causality_is_checked_over_the_whole_row(tmp_path):
    m = _fake_run(tmp_path, "peeklate", PEEK_LATE)
    assert not m["valid"] and "non-causal" in m["message"], m["message"]


STATEFUL = """
import os
import numpy as np
import torch
PROMPTS = {{tuple(r.tolist()) for r in np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.npz"))["tokens"]}}
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
    {extra}
    def forward(self, idx):
        if self.calls == 0:   # the first forward is the warm-up: it must not carry an evaluation prompt
            assert not any(tuple(r) in PROMPTS for r in idx.tolist()), "warm-up saw an evaluation prompt"
        self.calls += 1
        {fwd}
        return torch.zeros(*idx.shape, 8192) + self.w
def make_model(config):
    return M()
"""


def test_evaluator_rejects_hidden_or_changing_model_state(tmp_path):
    d = _npz("validation")
    table = {"table.npz": {"tokens": d["tokens"]}}
    ck = {"w": torch.zeros(1)}
    ok = _fake_run(tmp_path, "honest", STATEFUL.format(extra="", fwd="pass"), table, ckpt=ck)
    assert ok["valid"], ok["message"]                                     # also: the warm-up used random tokens
    trains_in_eval = "def train(self, mode=True):\n        super().train(mode)\n        with torch.no_grad():\n" \
                     "            self.w.add_(1.0)\n        return self"
    for name, extra, fwd, why in (
            ("evaltrain", trains_in_eval, "pass", "differ from the checkpoint"),
            ("fwdtrain", "", "self.w.data.add_(0.5)", "differ from the checkpoint"),
            ("hidden", "", "self.cache = torch.ones(100)", "unregistered tensor state"),
            ("hiddenattr", "table = torch.ones(1000)", "pass", "unregistered tensor state")):
        m = _fake_run(tmp_path, name, STATEFUL.format(extra=extra, fwd=fwd), table, ckpt=ck)
        assert not m["valid"] and why in m["message"], (name, m["message"])


def test_worker_cannot_read_labels_patch_the_scorer_or_forge_results(tmp_path):
    out = tmp_path / "evil.json"
    m = _fake_run(tmp_path, "evil", EVIL.format(data=os.path.abspath(D), out=out))
    assert m["valid"], m["message"]                            # every attack failed (make_model asserts that) ...
    assert m["primary"] == 0.0 and m["metrics"]["acc_ext"] == 0.0   # ... and the scores are the honest ones


class Broadcast(torch.nn.Module):
    """Looped's core matmul rewritten as broadcast multiply + sum (no op with a FLOP formula)."""

    def __init__(self, k):
        super().__init__()
        self.inner = Looped(k)

    def forward(self, idx):
        m = self.inner
        h = m.emb(idx)
        for _ in range(m.k):
            h = h + (h.unsqueeze(-2) * m.core.weight).sum(-1) + m.core.bias
        return m.head(h)


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
    x = torch.as_tensor(np.random.default_rng(0).integers(0, VOCAB, (2, 32)))
    f1, f2, f3 = (meter.metered(Looped(k), x)[1] / x.numel() for k in (1, 2, 3))
    core = 2 * 16 * 16
    assert f2 - f1 == f3 - f2 and core <= f2 - f1 <= 1.2 * core and f1 >= core + 2 * 16 * VOCAB
    g1, g2 = (meter.metered(Broadcast(k), x)[1] / x.numel() for k in (1, 2))
    assert g2 - g1 >= 0.9 * (f2 - f1)             # matmul rewritten as broadcast-multiply + sum costs the same
    assert not meter.metered(Looped(1), x)[2]

    @torch.library.custom_op("latentarch_test::double", mutates_args=())
    def double(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    @double.register_fake
    def _(t):
        return torch.empty_like(t)

    class Custom(Looped):
        def forward(self, idx):
            return double(super().forward(idx))

    assert any("latentarch_test" in op for op in meter.metered(Custom(1), x)[2])
    q = torch.randn(1, 2, 16, 8)

    class Attn(torch.nn.Module):   # SDPA on CPU is counted like the CUDA kernels
        def forward(self, idx):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            return torch.zeros(*idx.shape, VOCAB)

    formula = 4 * 1 * 2 * 16 * 16 * 8
    assert formula <= meter.metered(Attn(), x)[1] <= 1.5 * formula
