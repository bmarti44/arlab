import pytest
from interval_set import IntervalSet


def test_empty_and_endpoints():
    s = IntervalSet()
    assert s.intervals() == []
    assert not s.contains(0)
    s.add(-2, 3)
    assert [s.contains(x) for x in (-3, -2, 2, 3)] == [False, True, True, False]


def test_sort_merge_and_bridge():
    s = IntervalSet()
    for pair in [(8, 10), (1, 3), (4, 6), (3, 8)]:
        s.add(*pair)
    assert s.intervals() == [(1, 10)]
    s.add(2, 4)
    s.add(1, 10)
    assert s.intervals() == [(1, 10)]


def test_remove_splits_and_trims():
    s = IntervalSet()
    s.add(0, 20)
    s.remove(5, 9)
    assert s.intervals() == [(0, 5), (9, 20)]
    s.remove(-4, 2)
    s.remove(18, 25)
    assert s.intervals() == [(2, 5), (9, 18)]


def test_remove_across_gaps():
    s = IntervalSet()
    for pair in [(0, 2), (4, 6), (8, 10)]:
        s.add(*pair)
    s.remove(1, 9)
    assert s.intervals() == [(0, 1), (9, 10)]
    s.remove(1, 9)
    s.remove(100, 200)
    assert s.intervals() == [(0, 1), (9, 10)]
    s.remove(-100, 100)
    assert s.intervals() == []


def test_empty_reversed_and_atomic():
    s = IntervalSet()
    s.add(2, 6)
    s.add(3, 3)
    s.remove(4, 4)
    for method in (s.add, s.remove):
        with pytest.raises(ValueError):
            method(9, 1)
    assert s.intervals() == [(2, 6)]


def test_snapshot_and_large_negative_values():
    s = IntervalSet()
    s.add(-10**30, -10)
    snapshot = s.intervals()
    snapshot.clear()
    assert s.contains(-10**30)
    assert not s.contains(-10)
    assert s.intervals() == [(-10**30, -10)]
