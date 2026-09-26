"""Deterministic CPU checks of memory-longmemeval: splits, scorer on ambiguous answers, BM25."""
import json
import sys

sys.path.insert(0, "/frozen")
from memtools import BM25  # noqa: E402
from normalize import aliases, normalize, score  # noqa: E402


def load(split, kind, name):
    return [json.loads(line) for line in open(f"/data/{split}/{kind}/{name}")]


def test_splits_disjoint_equal_and_short_span():
    v, h = load("validation", "private", "gold.jsonl"), load("holdout", "private", "gold.jsonl")
    sp = json.load(open("/data/splits.json"))
    assert len(v) == len(h) == sp["validation"] == sp["holdout"] >= 200
    assert not {g["id"] for g in v} & {g["id"] for g in h}
    assert all(g["abstention"] or (g["aliases"] and all(len(a.split()) <= 5 for a in g["aliases"])) for g in v + h)
    assert {g["type"] for g in v} == {g["type"] for g in h}
    assert not any("preference" in g["type"] for g in v + h)
    q = load("validation", "public", "questions.jsonl")
    assert [x["id"] for x in q] == [g["id"] for g in v] and "answer" not in q[0]


GOLD = {"aliases": aliases("25 minutes and 50 seconds (or 25:50)"), "abstention": False}


def test_scorer_exact_match_and_hedges():
    assert score("25:50", GOLD) == 1 and score("25 minutes and 50 seconds.", GOLD) == 1
    assert score("25:50 or 26:00", GOLD) == 0
    assert score("It was 25:50", GOLD) == 0
    g = {"aliases": aliases("Paris"), "abstention": False}
    assert score("paris", g) == 1 and score("The Paris.", g) == 1
    assert score("Paris or London", g) == 0 and score("not Paris", g) == 0 and score("Paris, London", g) == 0
    days = {"aliases": aliases("7 days. 8 days (including the last day) is also acceptable."), "abstention": False}
    assert score("seven days", days) == 1 and score("8 days", days) == 1 and score("7 or 8 days", days) == 0
    assert score("four", {"aliases": aliases("4"), "abstention": False}) == 1


def test_abstention():
    g = {"aliases": [], "abstention": True}
    assert score("I don't know", g) == 1 and score("I do not know.", g) == 1
    assert score("I don't know, maybe Paris", g) == 0 and score("Paris", g) == 0
    assert score("I don't know", {"aliases": aliases("Paris"), "abstention": False}) == 0


def test_normalize_numbers_and_punctuation():
    assert normalize("The $1,200 fee!") == "$ 1200 fee" and normalize("Twenty-one") == "21" and normalize("20 5") == "25"


def test_bm25_deterministic():
    idx = BM25(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    assert idx.search("cat", 5) == [0, 2] and idx.search("zebra", 5) == []
