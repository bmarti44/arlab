"""Reference `evicted_1024` (frozen): the baseline policy (evicted mentor sentences, newest-first, 4C header,
after the thread) with a 1,024-token budget instead of 256. Separates "restate more" from "select better"."""
from window import Plan

BUDGET = 1024


def plan(sentences, current, query, window):
    evicted = [r["id"] for r in sentences if r["speaker"] == "mentor" and r["evicted_at_base_window"]]
    return Plan(ids=window.pack_newest_first(evicted, BUDGET, 0), header=0, placement="after_thread", budget=BUDGET)
