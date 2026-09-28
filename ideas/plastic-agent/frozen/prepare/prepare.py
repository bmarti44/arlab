"""PREPARE for plastic-agent (CPU, offline, deterministic). Mounted only here: /prepare (this dir), so the FauxOS
generator, the world seeds and the explorer are never visible to RUN.

Layout written under --out:
  {validation,holdout}/public/worlds.json    [{id, tools (sorted names), transcript: [{call, obs}]}]  <- all RUN sees
  {validation,holdout}/private/worlds.json   {id: {spec, end, tasks: [{id, goal, template, n_calls, check_state,
                                              gold_state, answer, reference}]}}
  {validation,holdout}/private/guard.json    the guard world (its own seed): tools, transcript, spec, end, tasks
  {validation,holdout}/private/gsm8k.json    forgetting battery: GSM8K test subset [{id, q, a}]
  {validation,holdout}/private/text.npy      OASST2 rows for the text-NLL report
  {validation,holdout}/private/fauxos.py     the simulator, for the evaluator (a copy of /prepare/fauxos.py)
  train/replay.npy                           OASST2 rows for KL-to-base / replay (disjoint from text.npy)
  splits.json, info.json
World seeds: validation, holdout and the guard world come from disjoint seed ranges; each world's tasks start from
the state its exploration ended in; tasks and gold outcomes are private.
"""
import argparse
import gzip
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, "/prepare")
sys.path.insert(0, "/frozen")
import fauxos  # noqa: E402
from common import GSM8K_FILE, MODEL_DIR, OASST_FILE, chat_prefix, system_text  # noqa: E402

SEED_BASE = {"validation": 26_092_700, "holdout": 26_092_800, "guard": 26_092_900}   # disjoint ranges of 100
N_WORLDS = {"validation": 4, "holdout": 8}
N_TASKS = {"validation": 60, "holdout": 80}   # per world: 240 / 640 items; power arithmetic in pack.yaml / IDEA.md
N_EXPLORE = 500           # tool calls per exploration transcript (CALIBRATE: transcript length vs ICL gate)
GUARD_EXPLORE, GUARD_TASKS = 150, 100
N_GSM8K, N_TEXT, N_REPLAY, TEXT_LEN = 200, 64, 512, 256

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--small", action="store_true", help="tiny sizes for local smoke tests only")
a = ap.parse_args()
if a.small:
    N_WORLDS, N_TASKS, N_EXPLORE = {"validation": 2, "holdout": 2}, {"validation": 12, "holdout": 12}, 60
    GUARD_EXPLORE, GUARD_TASKS, N_GSM8K, N_TEXT, N_REPLAY = 40, 8, 8, 8, 16


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


def build_world(wid: str, seed: int, n_explore: int, n_tasks: int) -> tuple[dict, dict]:
    spec = fauxos.gen_world(seed)
    events, end = fauxos.explore(spec, n_explore)
    tasks = fauxos.make_tasks(spec, end, events, n_tasks, wid)
    for k, t in enumerate(tasks):
        t["id"] = f"{wid}-t{k:02d}"
    pub = {"id": wid, "tools": sorted(t["name"] for t in spec["tools"]), "transcript": events}
    return pub, {"spec": spec, "end": end, "tasks": tasks}


from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
info = {"seed_base": SEED_BASE, "n_explore": N_EXPLORE, "n_tasks": N_TASKS, "worlds": {}}

g_pub, g_priv = build_world("g0", SEED_BASE["guard"], GUARD_EXPLORE, GUARD_TASKS)
guard = {**g_pub, **g_priv}
splits = {}
for split in ("validation", "holdout"):
    pubs, privs = [], {}
    for i in range(N_WORLDS[split]):
        wid = f"{split[0]}{i}"
        pub, priv = build_world(wid, SEED_BASE[split] + i, N_EXPLORE, N_TASKS[split])
        pubs.append(pub)
        privs[wid] = priv
        n_icl = len(tok(chat_prefix(system_text(pub["tools"], pub["transcript"])), add_special_tokens=False)["input_ids"])
        n_sys = len(tok(chat_prefix(system_text(pub["tools"])), add_special_tokens=False)["input_ids"])
        info["worlds"][wid] = {"seed": SEED_BASE[split] + i, "tools": len(pub["tools"]), "objects_end": len(priv["end"]["objs"]),
                               "icl_prefix_tokens": n_icl, "system_tokens": n_sys,
                               "templates": {t: sum(x["template"] == t for x in priv["tasks"]) for t in fauxos.TEMPLATES}}
    write_json(f"{a.out}/{split}/public/worlds.json", pubs)
    write_json(f"{a.out}/{split}/private/worlds.json", privs)
    write_json(f"{a.out}/{split}/private/guard.json", guard)
    shutil.copyfile("/prepare/fauxos.py", f"{a.out}/{split}/private/fauxos.py")
    splits[split] = sum(len(p["tasks"]) for p in privs.values())

# ---- forgetting battery: GSM8K test subset (pinned snapshot, offline)
import pyarrow.parquet as pq  # noqa: E402

rows = pq.read_table(GSM8K_FILE).to_pylist()
idx = random.Random(11).sample(range(len(rows)), N_GSM8K)
gsm = [{"id": f"gsm{i:04d}", "q": rows[i]["question"], "a": int(rows[i]["answer"].split("####")[-1].strip().replace(",", ""))}
       for i in idx]
for split in ("validation", "holdout"):
    write_json(f"{a.out}/{split}/private/gsm8k.json", gsm)

# ---- OASST2 English rows: text-NLL report (private) and replay rows for KL-to-base (train); disjoint
msgs = []


def walk(node):
    if node.get("lang") == "en" and node.get("text"):
        msgs.append((node["message_id"], node["text"]))
    for r in node.get("replies") or []:
        walk(r)


with gzip.open(OASST_FILE, "rt") as f:
    for line in f:
        walk(json.loads(line)["prompt"])
msgs.sort()
random.Random(7).shuffle(msgs)
need, text_rows = N_TEXT + N_REPLAY, []
for i in range(0, len(msgs), 512):
    for ids in tok([t for _, t in msgs[i:i + 512]], add_special_tokens=False)["input_ids"]:
        if len(ids) >= TEXT_LEN + 1 and len(text_rows) < need:
            text_rows.append(ids[:TEXT_LEN + 1])
    if len(text_rows) >= need:
        break
text_rows = np.array(text_rows, dtype=np.int64)
assert text_rows.shape == (need, TEXT_LEN + 1), text_rows.shape
for split in ("validation", "holdout"):
    np.save(f"{a.out}/{split}/private/text.npy", text_rows[:N_TEXT])
os.makedirs(f"{a.out}/train", exist_ok=True)
np.save(f"{a.out}/train/replay.npy", text_rows[N_TEXT:])

write_json(f"{a.out}/splits.json", splits)
info["guard"] = {"seed": SEED_BASE["guard"], "tasks": len(guard["tasks"]),
                 "icl_prefix_tokens": len(tok(chat_prefix(system_text(guard["tools"], guard["transcript"])), add_special_tokens=False)["input_ids"])}
info["splits"] = splits
write_json(f"{a.out}/info.json", info)
print(json.dumps({k: v for k, v in info.items()}, indent=1))
