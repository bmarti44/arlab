import pytest
from leaderboard import Leaderboard


def test_equal_scores_use_name_order():
    board = Leaderboard()
    for name in ["zoe", "amy", "bob"]:
        board.set_score(name, 10)
    assert board.ranking() == [(1, "amy", 10), (1, "bob", 10), (1, "zoe", 10)]


def test_competition_rank_skips_positions():
    board = Leaderboard()
    for name, score in [("d", 2), ("c", 5), ("b", 9), ("a", 9)]:
        board.set_score(name, score)
    assert board.ranking() == [(1, "a", 9), (1, "b", 9), (3, "c", 5), (4, "d", 2)]
    assert board.rank_of("b") == 1
    assert board.rank_of("missing") is None


def test_update_creates_and_breaks_ties():
    board = Leaderboard()
    board.set_score("a", 3)
    board.set_score("b", 8)
    board.set_score("a", 8)
    assert board.rank_of("a") == board.rank_of("b") == 1
    board.set_score("b", 1)
    assert board.ranking() == [(1, "a", 8), (2, "b", 1)]


def test_removal_recomputes_ranks():
    board = Leaderboard()
    for name, score in [("first", 10), ("a", 5), ("b", 5)]:
        board.set_score(name, score)
    assert board.remove("first") is True
    assert board.remove("absent") is False
    assert board.ranking() == [(1, "a", 5), (1, "b", 5)]
    assert board.remove("a") is True
    assert board.rank_of("b") == 1


def test_top_can_split_a_tie_and_is_detached():
    board = Leaderboard()
    board.set_score("b", 4)
    board.set_score("a", 4)
    assert board.top(1) == [(1, "a", 4)]
    assert board.top(0) == []
    with pytest.raises(ValueError):
        board.top(-1)
    result = board.top(20)
    result.clear()
    assert len(board.ranking()) == 2


def test_multiple_ties_case_order_and_negative_scores():
    board = Leaderboard()
    assert board.ranking() == [] and board.rank_of("x") is None
    for name, score in [("a", 0), ("A", 0), ("c", -2), ("b", -2), ("z", -7)]:
        board.set_score(name, score)
    assert board.ranking() == [(1, "A", 0), (1, "a", 0), (3, "b", -2), (3, "c", -2), (5, "z", -7)]
