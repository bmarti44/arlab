import pytest
from deadline_queue import DeadlineQueue


def test_priority_deadline_and_fifo_order():
    q = DeadlineQueue()
    for args in [("none", 1), ("late", 1, 20), ("z", 1, 10),
                 ("a", 1, 10), ("urgent", -1, 99)]:
        assert q.add(*args) is None
    assert q.pending() == ["urgent", "z", "a", "late", "none"]
    assert len(q) == 5


def test_expiry_boundary_and_global_purge():
    q = DeadlineQueue()
    q.add("expired", 99, 4)
    q.add("boundary", 0, 5)
    q.add("forever", 1)
    assert q.pop(5) == "boundary"
    assert q.pending() == ["forever"]
    assert len(q) == 1
    assert q.cancel("expired") is False


def test_empty_and_all_expired():
    q = DeadlineQueue()
    assert q.pop(0) is None
    q.add("old", 0, -2)
    q.add("older", 1, -3)
    assert q.pop(-1) is None
    assert len(q) == 0
    assert q.pending() == []


def test_duplicate_add_is_atomic():
    q = DeadlineQueue()
    q.add("a", 2, 10)
    q.add("b", 2, 10)
    with pytest.raises(ValueError):
        q.add("a", -10, -10)
    assert q.pending() == ["a", "b"]
    assert q.pop(0) == "a"


def test_cancel_and_reinsert_order():
    q = DeadlineQueue()
    q.add("a", 0)
    q.add("b", 0)
    assert q.cancel("missing") is False
    assert q.cancel("a") is True
    q.add("a", 0)
    assert q.pending() == ["b", "a"]
    assert q.pop(100) == "b"
    assert q.pop(100) == "a"


def test_pending_is_independent_and_does_not_expire():
    q = DeadlineQueue()
    q.add("past", -2, -50)
    q.add("future", 0, 30)
    ids = q.pending()
    ids.clear()
    assert len(q) == 2
    assert q.pending() == ["past", "future"]
    assert q.cancel("past") is True
