"""Label-derived oracle plans for the pilot (build tooling; never part of a campaign).

Runs in the pack image with the PILOT slice mounted (/data/public, /data/private). For every item: the verbatim
sentence that last stated each live convention whose family the query REQUIRES (labels.oracle_units filtered by
labels.required; query-aware selection is allowed to the surface too), chronological, header 0, after the thread,
budget 1,024 (newest-first packing if they do not fit). A compound sentence can also carry a superseded
convention; such sentences are kept (the live statement exists only there) and counted as `stale`.
Writes /out/oracle_plans.json and prints clipping/stale counts.
"""
import json
import sys

sys.path.insert(0, "/frozen")
import window  # noqa: E402

topics = json.load(open("/data/private/vendor/memorycode/topics.json"))
family = {int(t["id"]): [r[0] for r in t["regex"]] for t in topics["instructions"]}
labels = {x["id"]: x for x in map(json.loads, open("/data/private/labels.jsonl"))}
plans, clipped, stale, empty = {}, 0, 0, 0
for it in map(json.loads, open("/data/public/items.jsonl")):
    lab = labels[it["id"]]
    live = [list(x) for x in lab["live"]]
    win = window.Window(it)
    ids = []
    for k in lab["oracle_units"]:
        u = lab["units"][k]
        if not any(pu in live and family[pu[0]][pu[1]] in lab["required"] for pu in u["conv"]):
            continue
        stale += any(pu not in live for pu in u["conv"])
        hits = [r["id"] for r in win.sentences if r["session"] == u["session"] and r["speaker"] == "mentor" and r["text"] == u["text"]]
        assert hits, (it["id"], u["text"])
        ids.append(hits[-1])
    ids = sorted(set(ids))
    empty += not ids
    kept = win.pack_newest_first(ids, window.MAX_BUDGET, 0)
    clipped += len(kept) < len(ids)
    plan = win.check(window.Plan(ids=kept, header=0, placement="after_thread", budget=window.MAX_BUDGET))
    plans[it["id"]] = {"ids": plan.ids, "header": plan.header, "placement": plan.placement, "budget": plan.budget}
json.dump(plans, open("/out/oracle_plans.json", "w"))
print(f"oracle plans for {len(plans)} items; {clipped} clipped to {window.MAX_BUDGET} tokens; {stale} sentences also carry "
      f"a superseded convention; {empty} items with no required convention outside the current session")
