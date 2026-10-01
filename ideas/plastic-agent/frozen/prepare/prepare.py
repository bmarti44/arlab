"""PREPARE for plastic-agent (CPU, offline, deterministic). Mounted only here: /prepare (this dir), so the FauxOS
generator, the world seeds and the explorer are never visible to RUN.

Layout written under --out:
  {validation,holdout}/public/order.json     [world ids] in harness order                           <- all RUN sees
  {validation,holdout}/public/worlds/<id>.json  {id, tools (sorted names), transcript: [{call, obs}]}   (one file per
                                              world: the harness loads only the world being adapted)
  {validation,holdout}/public/twins/<id>.json   the target world <id>'s paired CONTROL world ("twin"): {id: <id>x,
                                              tools (the SAME sorted names), transcript of the twin (same explorer)}
  {validation,holdout}/private/twins.json    {target id: {id, seed, spec, end}} of the twins (tests only)
  {validation,holdout}/private/worlds.json   {id: {spec, end, tasks: [{id, goal, template, n_calls, check_state,
                                              gold_state, answer, reference}]}}
  {validation,holdout}/private/guard.json    the split's guard world (own seed): tools, transcript, spec, end, tasks
  {validation,holdout}/private/gsm8k.json    forgetting battery: GSM8K test items [{id, q, a}] (disjoint per split)
  {validation,holdout}/private/text.npy      OASST2 rows for the text-NLL report (disjoint per split)
  {validation,holdout}/private/fauxos.py     the simulator, for the evaluator (a copy of /prepare/fauxos.py)
  train/replay.npy                           OASST2 rows for KL-to-base / replay (disjoint from text.npy)
  splits.json, info.json
novel/ (v2 robustness split, 2026-10-01): the same layout; 8 worlds in fauxos style "novel" (4 operation types that
  never occur in validation/holdout, reworded observations and goals for every op), their twins, and the HOLDOUT's
  forgetting battery (guard world, GSM8K, text) so battery numbers stay comparable. Validation/holdout are unchanged.
World seeds: validation, holdout, the two guard worlds and the twins come from disjoint seed ranges. A twin
(fauxos.gen_twin) has the target's tool names, operation set and argument vocabulary but deranged name -> operation
semantics, fresh argument orders, variants and objects; the surface adapts to it like to any world and the evaluator
scores that control adapter on the TARGET world's tasks (world_specific_gain). each world's tasks start
from the state its exploration ended in; tasks and gold outcomes are private. The forgetting batteries of the two
splits (guard world, GSM8K items, text rows) are disjoint. The raw dataset files are read only here (their paths are
not in frozen/run/); the evaluator reads the private copies.
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
from common import MODEL_DIR, chat_prefix, system_text  # noqa: E402

OASST_FILE = ("/hf/hub/datasets--OpenAssistant--oasst2/snapshots/179dd21fc55192153d94adb0e0ce8f69e222bf75/"
              "2023-11-05_oasst2_ready.trees.jsonl.gz")
GSM8K_FILE = "/hf/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"

SEED_BASE = {"validation": 26_092_700, "holdout": 26_092_800, "guard": 26_092_900,        # disjoint ranges of 100
             "twin_validation": 26_093_000, "twin_holdout": 26_093_100, "novel": 26_093_200, "twin_novel": 26_093_300}
N_WORLDS = {"validation": 4, "holdout": 8, "novel": 8}
N_TASKS = {"validation": 60, "holdout": 80, "novel": 80}
STYLE = {"validation": "std", "holdout": "std", "novel": "novel"}   # per world: 240 / 640 items; power arithmetic in pack.yaml / IDEA.md
N_EXPLORE = 120           # tool calls per exploration transcript (~2.5k tokens; CALIBRATE with the gate)
GUARD_EXPLORE, GUARD_TASKS = 120, 100
N_GSM8K, N_TEXT, N_REPLAY, TEXT_LEN = 200, 64, 512, 256

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--small", action="store_true", help="tiny sizes for local smoke tests only")
a = ap.parse_args()
if a.small:
    N_WORLDS, N_TASKS, N_EXPLORE = {"validation": 2, "holdout": 2, "novel": 2}, {"validation": 12, "holdout": 12, "novel": 12}, 60
    GUARD_EXPLORE, GUARD_TASKS, N_GSM8K, N_TEXT, N_REPLAY = 60, 8, 8, 8, 16


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


def build_world(wid: str, seed: int, n_explore: int, n_tasks: int, style: str = "std") -> tuple[dict, dict]:
    spec = fauxos.gen_world(seed, style)
    events, end = fauxos.explore(spec, n_explore)
    tasks = fauxos.make_tasks(spec, end, events, n_tasks, wid)
    for k, t in enumerate(tasks):
        t["id"] = f"{wid}-t{k:02d}"
    pub = {"id": wid, "tools": sorted(t["name"] for t in spec["tools"]), "transcript": events}
    return pub, {"spec": spec, "end": end, "tasks": tasks}


from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
info = {"seed_base": SEED_BASE, "n_explore": N_EXPLORE, "n_tasks": N_TASKS, "worlds": {}, "twins": {}}

splits, guards = {}, {}
for k, split in enumerate(("validation", "holdout", "novel")):
    if split == "novel":
        guard = guards["holdout"]                     # the holdout's guard world (comparable battery)
    else:
        g_pub, g_priv = build_world(f"g{k}", SEED_BASE["guard"] + k, GUARD_EXPLORE, GUARD_TASKS)
        guards[split] = guard = {**g_pub, **g_priv}
    pubs, privs, twins = [], {}, {}
    for i in range(N_WORLDS[split]):
        wid = f"{split[0]}{i}"
        pub, priv = build_world(wid, SEED_BASE[split] + i, N_EXPLORE, N_TASKS[split], STYLE[split])
        pubs.append(pub)
        privs[wid] = priv
        tseed = SEED_BASE[f"twin_{split}"] + i
        tspec = fauxos.gen_twin(priv["spec"], tseed)
        tev, tend = fauxos.explore(tspec, N_EXPLORE)
        twins[wid] = {"pub": {"id": f"{wid}x", "tools": sorted(t["name"] for t in tspec["tools"]), "transcript": tev},
                      "priv": {"id": f"{wid}x", "seed": tseed, "spec": tspec, "end": tend}}
        assert twins[wid]["pub"]["tools"] == pub["tools"]
        info["twins"][f"{wid}x"] = {"seed": tseed, "twin_of": wid}
        n_icl = len(tok(chat_prefix(system_text(pub["tools"], pub["transcript"])), add_special_tokens=False)["input_ids"])
        n_sys = len(tok(chat_prefix(system_text(pub["tools"])), add_special_tokens=False)["input_ids"])
        info["worlds"][wid] = {"seed": SEED_BASE[split] + i, "tools": len(pub["tools"]), "objects_end": len(priv["end"]["objs"]),
                               "icl_prefix_tokens": n_icl, "system_tokens": n_sys,
                               "templates": {t: sum(x["template"] == t for x in priv["tasks"]) for t in fauxos.TEMPLATES}}
    write_json(f"{a.out}/{split}/public/order.json", [p["id"] for p in pubs])
    for p in pubs:
        write_json(f"{a.out}/{split}/public/worlds/{p['id']}.json", p)
        write_json(f"{a.out}/{split}/public/twins/{p['id']}.json", twins[p["id"]]["pub"])
    write_json(f"{a.out}/{split}/private/twins.json", {w: t["priv"] for w, t in twins.items()})
    write_json(f"{a.out}/{split}/private/worlds.json", privs)
    write_json(f"{a.out}/{split}/private/guard.json", guard)
    shutil.copyfile("/prepare/fauxos.py", f"{a.out}/{split}/private/fauxos.py")
    splits[split] = sum(len(p["tasks"]) for p in privs.values())

# ---- forgetting battery: GSM8K test subset (pinned snapshot, offline)
import pyarrow.parquet as pq  # noqa: E402

rows = pq.read_table(GSM8K_FILE).to_pylist()
idx = random.Random(11).sample(range(len(rows)), 2 * N_GSM8K)
gsm = [{"id": f"gsm{i:04d}", "q": rows[i]["question"], "a": int(rows[i]["answer"].split("####")[-1].strip().replace(",", ""))}
       for i in idx]
for k, split in enumerate(("validation", "holdout", "novel")):
    k = min(k, 1)                                     # novel reuses the holdout's GSM8K items
    write_json(f"{a.out}/{split}/private/gsm8k.json", gsm[k * N_GSM8K:(k + 1) * N_GSM8K])

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
need, text_rows = 2 * N_TEXT + N_REPLAY, []
for i in range(0, len(msgs), 512):
    for ids in tok([t for _, t in msgs[i:i + 512]], add_special_tokens=False)["input_ids"]:
        if len(ids) >= TEXT_LEN + 1 and len(text_rows) < need:
            text_rows.append(ids[:TEXT_LEN + 1])
    if len(text_rows) >= need:
        break
text_rows = np.array(text_rows, dtype=np.int64)
assert text_rows.shape == (need, TEXT_LEN + 1), text_rows.shape
for k, split in enumerate(("validation", "holdout", "novel")):
    k = min(k, 1)                                     # novel reuses the holdout's text rows
    np.save(f"{a.out}/{split}/private/text.npy", text_rows[k * N_TEXT:(k + 1) * N_TEXT])
os.makedirs(f"{a.out}/train", exist_ok=True)
np.save(f"{a.out}/train/replay.npy", text_rows[2 * N_TEXT:])

write_json(f"{a.out}/splits.json", splits)
info["guard"] = {s: {"id": g["id"], "seed": SEED_BASE["guard"] + k, "tasks": len(g["tasks"]),
                    "icl_prefix_tokens": len(tok(chat_prefix(system_text(g["tools"], g["transcript"])), add_special_tokens=False)["input_ids"])}
                 for k, (s, g) in enumerate(guards.items())}
info["splits"] = splits
write_json(f"{a.out}/info.json", info)
print(json.dumps({k: v for k, v in info.items()}, indent=1))
