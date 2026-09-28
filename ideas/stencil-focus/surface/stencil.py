"""Editable surface: the reminder policy (which earlier sentences are restated, how, where, how large).

plan(sentences, current, query, window) -> Plan. See frozen/run/window.py for the records and helpers.
Baseline = stencil-llm's Exp 4C shipping policy: every MENTOR sentence evicted at the base window (the thread cut
with no reminder), packed newest-first into 256 tokens under "Earlier instructions still in force:", placed after
the truncated thread and before the request.
"""
from window import Plan

BUDGET = 256          # reminder token cap (<= 1024)
HEADER = 0            # window.HEADERS[0] = "Earlier instructions still in force:"
PLACEMENT = "after_thread"


def plan(sentences, current, query, window):
    evicted = [r["id"] for r in sentences if r["speaker"] == "mentor" and r["evicted_at_base_window"]]
    return Plan(ids=window.pack_newest_first(evicted, BUDGET, HEADER), header=HEADER, placement=PLACEMENT, budget=BUDGET)
