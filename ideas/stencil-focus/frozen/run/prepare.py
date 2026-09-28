"""PREPARE for stencil-focus: pinned stencil-llm snapshot, model/dataset downloads, the synthetic bank, splits.

Layout written under --out (construction details: frozen/prepare/BANK.md, not agent-visible):
  {validation,holdout,pilot}/public/items.jsonl     {id, mentor, mentee, sessions: [text, ...], query}; item = last session
  {validation,holdout,pilot}/public/stencil/src/stencil/{__init__,memorycode,focus3,focus2,stats}.py
  {validation,holdout,pilot}/private/labels.jsonl   live set, history_regex, required, structure, units, oracle, sources
  {validation,holdout,pilot}/private/vendor/memorycode/{code/,topics.json,LICENSE,README.md}   (the checker)
  holdout/public/real.jsonl + holdout/private/real_labels.jsonl   the free real MemoryCode dialogues (descriptive)
  splits.json, prepare_report.json
The pilot slice is in neither split (only build/pilot.sh reads it). stencil-llm: GPL-2.0; MemoryCode: Apache-2.0;
both for local use only.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import statistics
import sys
import tarfile

STENCIL_SHA = "87b72a88cd56712a865359afd794a34b66f3d05d"          # stencil-llm HEAD 2026-09-13
SNAPSHOT = f"/prepare/stencil-{STENCIL_SHA}.tar.gz"                 # build/snapshot.sh: the path-limited git archive
SNAPSHOT_SHA256 = "4f8b214868a7eab2a03837b0fb52e1f94d9e097f2bf479bb55e939f246d96052"
SNAPSHOT_PATHS = ["src/stencil/__init__.py", "src/stencil/memorycode.py", "src/stencil/focus3.py", "src/stencil/focus2.py",
                  "src/stencil/stats.py", "vendor/memorycode", "data/g0/chat.jsonl"]
MODEL, MODEL_REV = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
ULTRACHAT, ULTRACHAT_REV = "HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"
ULTRACHAT_FILE = "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"
OASST, OASST_REV = "OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75"
OASST_FILE = "2023-11-05_oasst2_ready.trees.jsonl.gz"
POOL_SHARE = {"validation": 0.45, "holdout": 0.45, "pilot": 0.10}
REAL_W = 1536
REAL_EXPECT = {"short_dialogues": 108, "derived_pool": 80, "long_dialogues": 212, "free": 28}
MAX_TURN_CHARS = 4000
CURRENT_MAX_EXACT = 2300          # tokens; a 1,024-token reminder still leaves the whole current session in W

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--snapshot", default=SNAPSHOT)
ap.add_argument("--prepare-dir", default="/prepare")
ap.add_argument("--small", type=int, default=0, help="local smoke test only: N dialogues per slice, no model download")
ap.add_argument("--tokenizer", default=None, help="local smoke test only: tokenizer.json to use")
a = ap.parse_args()
OUT = a.out


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def jlines(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- 1. pinned, path-limited stencil-llm snapshot
blob = open(a.snapshot, "rb").read()
assert hashlib.sha256(blob).hexdigest() == SNAPSHOT_SHA256, "stencil snapshot tarball changed (re-run build/snapshot.sh?)"
SNAP = "/tmp/stencil-snapshot"
shutil.rmtree(SNAP, ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
    assert tf.pax_headers.get("comment") == STENCIL_SHA, "tarball is not the pinned stencil-llm commit"
    names = tf.getnames()
    assert all(any(n == p or n.startswith(p + "/") or p.startswith(n + "/") for p in SNAPSHOT_PATHS) for n in names), \
        "snapshot holds paths outside the pinned list"
    tf.extractall(SNAP, filter="data")
sys.path.insert(0, f"{SNAP}/src")
sys.path.insert(0, a.prepare_dir)
from stencil import focus3, memorycode as mc  # noqa: E402
import bank  # noqa: E402  (frozen/prepare/bank.py)
import window  # noqa: E402  (frozen/run/window.py)

assert bank.sentences("A b. C d e.f g! H") == focus3.sentences("A b. C d e.f g! H")
topics = json.load(open(f"{SNAP}/vendor/memorycode/topics.json"))
para = json.load(open(f"{a.prepare_dir}/paraphrases.json"))

# ---------------------------------------------------------------- 2. downloads (pinned revisions) into /hf
if a.small:
    from tokenizers import Tokenizer
    window.tokenizer.cache_clear()
    _tok = Tokenizer.from_file(a.tokenizer)
    window.tokenizer = lambda: _tok
    ucpath = os.environ.get("ULTRACHAT_PARQUET")
    opath = os.environ.get("OASST_FILE")
else:
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download(MODEL, revision=MODEL_REV)
    ucpath = hf_hub_download(ULTRACHAT, ULTRACHAT_FILE, repo_type="dataset", revision=ULTRACHAT_REV)
    opath = hf_hub_download(OASST, OASST_FILE, repo_type="dataset", revision=OASST_REV)
tok = window.tokenizer()

# ---------------------------------------------------------------- 3. chatter pool (never scored; threads never reused)
g0_roots = sorted({json.loads(line)["id"].split(":")[0] for line in open(f"{SNAP}/data/g0/chat.jsonl")})
flat = lambda t: " ".join(str(t).split())  # noqa: E731


def usable(turns):
    if len(turns) < 2 or turns[0][0] != "user" or not any(r == "assistant" for r, _ in turns):
        return False
    return all(t and "```" not in t and "as an ai" not in t.lower() and len(t) <= MAX_TURN_CHARS for _, t in turns)


threads, dropped = [], {"oasst2_g0": 0, "oasst2_filtered": 0, "ultrachat_filtered": 0}
with gzip.open(opath, "rt") as f:
    for line in f:
        tree = json.loads(line)
        node = tree["prompt"]
        if tree["message_tree_id"] in g0_roots:
            dropped["oasst2_g0"] += 1
            continue
        if node.get("lang") != "en":
            continue
        turns = []
        while node is not None:
            turns.append(("user" if node["role"] == "prompter" else "assistant", flat(node["text"])))
            reps = [r for r in node.get("replies") or [] if not r.get("deleted") and r.get("lang") == "en"]
            node = min(reps, key=lambda r: (r.get("rank") if r.get("rank") is not None else 99, r["message_id"])) if reps else None
        while turns and turns[-1][0] != "assistant":
            turns.pop()
        if usable(turns):
            threads.append({"source": f"oasst2:{tree['message_tree_id']}", "turns": turns})
        else:
            dropped["oasst2_filtered"] += 1
import pyarrow.parquet as pq  # noqa: E402

for row in pq.read_table(ucpath, columns=["prompt_id", "messages"]).to_pylist():
    turns = [("user" if m["role"] == "user" else "assistant", flat(m["content"])) for m in row["messages"] if m["role"] in ("user", "assistant")]
    while turns and turns[-1][0] != "assistant":
        turns.pop()
    if usable(turns):
        threads.append({"source": f"ultrachat:{row['prompt_id']}", "turns": turns})
    else:
        dropped["ultrachat_filtered"] += 1
threads.sort(key=lambda t: t["source"])
random.Random("chatter-pool").shuffle(threads)
pools, k0 = {}, 0
for name, share in POOL_SHARE.items():
    k1 = len(threads) if name == "pilot" else k0 + int(share * len(threads))
    pools[name] = bank.Chatter(threads[k0:k1])
    k0 = k1

# ---------------------------------------------------------------- 4. the bank
by_id = {int(t["id"]): t for t in topics["instructions"]}
n_dialogues = {k: (a.small or v) for k, v in bank.N_DIALOGUES.items()}
report = {"stencil_sha": STENCIL_SHA, "snapshot_sha256": SNAPSHOT_SHA256, "model": [MODEL, MODEL_REV],
          "ultrachat": [ULTRACHAT, ULTRACHAT_REV, ULTRACHAT_FILE], "oasst2": [OASST, OASST_REV, OASST_FILE],
          "g0_root_ids": g0_roots, "chatter_threads": len(threads), "chatter_dropped": dropped,
          "pivots": {"A": list(bank.SIDE_A), "B": list(bank.SIDE_B)}, "slices": {}}
for sl, n in n_dialogues.items():
    texts = bank.Texts(para, sl)
    items, labels = [], []
    for k in range(n):
        side = bank.SIDE_OF_SLICE.get(sl) or ("A" if k % 2 == 0 else "B")
        d = bank.build_dialogue(topics, texts, pools[sl], side, f"{sl}:{k}")
        iid = hashlib.sha256(f"stencil-focus:{sl}:{k}".encode()).hexdigest()[:12]
        s = d["n_sessions"] - 1
        required = mc.required_families(d["query"], d["families"])
        structure = mc.required_structure(d["query"])
        assert required, (sl, k, d["query"], d["families"])
        # every inserted sentence must be exactly one span of the frozen splitter in a mentor line of its session
        dia = window.as_dialogue(d)
        spans = {}
        for i in range(d["n_sessions"]):
            spans[i] = [x for sp, line in mc.speaker_lines(dia, i) if sp == "mentor" for _, x in focus3.sentences(line)]
        for u in d["units"]:
            assert u["text"] in spans[u["session"]], ("unit is not one splitter span", sl, k, u["text"])
        assert mc.fraction_required([1.0] * len(d["history_regex"]), d["history_regex"], required, True) == 1.0
        items.append({"id": iid, "mentor": d["mentor"], "mentee": d["mentee"], "sessions": d["sessions"], "query": d["query"]})
        labels.append({"id": iid, "slice": sl, "side": d["side"], "query": d["query"], "live": d["live"],
                       "history_regex": d["history_regex"], "history_eval_query": d["history_eval_query"],
                       "families": d["families"], "required": required, "structure": structure,
                       "template": d["template"], "kinds": d["kinds"], "fillers": d["fillers"], "units": d["units"],
                       "oracle_units": d["oracle_units"], "n_sessions": d["n_sessions"]})
    # sources: which chatter threads each slice consumed (disjoint across slices by construction)
    report["slices"][sl] = {"sources": pools[sl].used}
    # exact lengths
    hist = tok.encode_batch([mc.thread_text(window.as_dialogue(it), list(range(len(it["sessions"]) - 1))) for it in items])
    cur = tok.encode_batch([it["sessions"][-1] for it in items])
    for lab, h, c in zip(labels, hist, cur):
        lab["history_tokens"], lab["current_tokens"] = len(h.ids), len(c.ids)
        assert len(c.ids) <= CURRENT_MAX_EXACT, ("current session too long", lab["id"], len(c.ids))
    ht = [x["history_tokens"] for x in labels]
    fam = {}
    for x in labels:
        for f in x["required"]:
            fam[f] = fam.get(f, 0) + 1
    report["slices"][sl].update({
        "n": len(items), "sessions_mean": statistics.mean(x["n_sessions"] for x in labels),
        "history_tokens": {"min": min(ht), "median": statistics.median(ht), "mean": statistics.mean(ht), "max": max(ht)},
        "current_tokens_max": max(x["current_tokens"] for x in labels),
        "live_mean": statistics.mean(len(x["live"]) for x in labels),
        "updates_mean": statistics.mean(sum(k == "instruction-update" for ks in x["kinds"] for k in ks) for x in labels),
        "oracle_units_mean": statistics.mean(len(x["oracle_units"]) for x in labels),
        "required_family_share": {f: round(v / len(labels), 3) for f, v in sorted(fam.items())},
        "structure_class_share": sum(x["structure"] == ["class"] for x in labels) / len(labels)})
    jlines(f"{OUT}/{sl}/public/items.jsonl", items)
    jlines(f"{OUT}/{sl}/private/labels.jsonl", labels)
    shutil.copytree(f"{SNAP}/src", f"{OUT}/{sl}/public/stencil/src")
    for part in ("code", "topics.json", "LICENSE", "README.md"):
        src, dst = f"{SNAP}/vendor/memorycode/{part}", f"{OUT}/{sl}/private/vendor/memorycode/{part}"
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)
used = [set(v["sources"]) for v in report["slices"].values()]
assert not (used[0] & used[1] or used[0] & used[2] or used[1] & used[2]), "chatter reused across slices"
assert not any(s.split(":", 1)[1] in g0_roots for u in used for s in u if s.startswith("oasst2:")), "g0 tree used"

# ---------------------------------------------------------------- 5. the free real MemoryCode dialogues (holdout, descriptive)


def ptoks(dlg, s):
    q = dlg["sessions"][s]["history_eval_query"][0]
    return len(tok.encode(mc.chat_prompt(mc.user_message(dlg, s, q, "history", ""))).ids)


short_first, long_d = {}, set()
for did in mc.dialogue_ids():
    dlg = mc.load_dialogue(did)
    for s, sess in enumerate(dlg["sessions"]):
        if s == 0 or not sess["history_regex"] or not any(mc.has_instruction(dlg["sessions"][i]) for i in range(s)):
            continue
        t = ptoks(dlg, s)
        if t <= mc.MAX_PROMPT_TOKENS:
            short_first.setdefault(did, s)
        else:
            long_d.add(did)
            if t > mc.MAX_PROMPT_TOKENS + 1000:   # later sessions only grow the history
                break
order = sorted(short_first)
random.Random(0).shuffle(order)                  # stencil.memorycode.split_items
derived = set(order[:80])
free = sorted(set(short_first) - derived - long_d)
counts = {"short_dialogues": len(short_first), "derived_pool": len(derived), "long_dialogues": len(long_d), "free": len(free),
          "short_long_overlap": len(set(short_first) & long_d)}
report["real"] = {"counts": counts, "W": REAL_W}
if {k: counts[k] for k in REAL_EXPECT} == REAL_EXPECT and not counts["short_long_overlap"]:
    pub, priv = [], []
    for did in free:
        dlg, s = mc.load_dialogue(did), short_first[did]
        q = dlg["sessions"][s]["history_eval_query"][0]
        fams = sorted({str(o) for o, _ in dlg["sessions"][s]["history_regex"]})
        rid = "r" + hashlib.sha256(f"stencil-focus:real:{did}".encode()).hexdigest()[:11]
        pub.append({"id": rid, "mentor": dlg["context"]["mentor"], "mentee": dlg["context"]["mentee"],
                    "sessions": [dlg["sessions"][i]["text"] for i in range(s + 1)], "query": q})
        priv.append({"id": rid, "dialogue": did, "session": s, "query": q, "history_regex": dlg["sessions"][s]["history_regex"],
                     "families": fams, "required": mc.required_families(q, fams), "structure": mc.required_structure(q)})
    jlines(f"{OUT}/holdout/public/real.jsonl", pub)
    jlines(f"{OUT}/holdout/private/real_labels.jsonl", priv)
    report["real"]["used"] = True
else:   # the recomputed pool accounting differs from the reviewed one: never risk a registered stencil item
    report["real"]["used"] = False

jdump(f"{OUT}/splits.json", {"validation": n_dialogues["validation"], "holdout": n_dialogues["holdout"]})
jdump(f"{OUT}/prepare_report.json", report)
print(json.dumps({k: v for k, v in report.items() if k not in ("g0_root_ids",)} | {"slices": {
    k: {kk: vv for kk, vv in v.items() if kk != "sources"} for k, v in report["slices"].items()}}, indent=1))
