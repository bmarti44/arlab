"""Reference `off` (frozen): no reminder; the thread fills the whole window."""
from window import Plan


def plan(sentences, current, query, window):
    return Plan(ids=[], header=0, placement="after_thread", budget=0)
