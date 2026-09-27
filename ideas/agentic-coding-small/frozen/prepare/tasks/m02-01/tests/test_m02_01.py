import pytest
from histogram import BoundedHistogram


def test_bin_geometry():
    h = BoundedHistogram(-2, 4, 3)
    assert h.bins() == [(-2, 0, 0), (1, 3, 0), (4, 4, 0)]
    assert BoundedHistogram(5, 5, 9).bins() == [(5, 5, 0)]


def test_accumulates_per_bin():
    h = BoundedHistogram(-2, 4, 3)
    assert h.add(-2, 2) is None
    h.add(0)
    h.add(4, 3)
    assert h.count(-1) == 3
    assert h.count(1) == 0
    assert h.count(4) == 3


def test_invalid_configuration():
    for args in [(2, 1, 1), (0, 2, 0), (0, 2, -1)]:
        with pytest.raises(ValueError):
            BoundedHistogram(*args)
    for args in [(False, 2, 1), (0, 2.0, 1), (0, 2, '1')]:
        with pytest.raises(TypeError):
            BoundedHistogram(*args)


def test_rejected_add_is_atomic():
    h = BoundedHistogram(0, 3, 2)
    h.add(1, 4)
    for args in [(-1, 0), (4, 2), (1, -1)]:
        with pytest.raises(ValueError):
            h.add(*args)
    for args in [(True, 2), (1, False), (1.5, 1)]:
        with pytest.raises(TypeError):
            h.add(*args)
    assert h.bins() == [(0, 1, 4), (2, 3, 0)]


def test_count_validation_and_zero_add():
    h = BoundedHistogram(2, 6, 2)
    h.add(6, 0)
    assert h.count(6) == 0
    for value in [1, 7]:
        with pytest.raises(ValueError):
            h.count(value)
    for value in [True, '2', 2.0]:
        with pytest.raises(TypeError):
            h.count(value)


def test_reset_and_snapshot():
    h = BoundedHistogram(1, 5, 3)
    h.add(2, 7)
    snapshot = h.bins()
    snapshot.clear()
    assert h.count(3) == 7
    assert h.reset() is None
    assert h.bins() == [(1, 3, 0), (4, 5, 0)]
    h.add(5)
    assert h.count(4) == 1
