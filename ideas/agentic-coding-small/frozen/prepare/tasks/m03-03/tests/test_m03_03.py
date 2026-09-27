import pytest
from moving_average import MovingAverage


def test_warmup_uses_actual_count():
    m = MovingAverage(4)
    assert m.average is None
    assert m.count == 0
    assert m.add(8) == 8.0
    assert m.add(4) == 6.0
    assert m.snapshot() == (8.0, 4.0)


def test_eviction_subtracts_oldest():
    m = MovingAverage(2)
    m.extend([2, 4])
    assert m.add(8) == 6.0
    assert m.add(-2) == 3.0
    assert m.count == 2
    assert m.snapshot() == (8.0, -2.0)


def test_single_sample_window():
    m = MovingAverage(1)
    assert m.extend([5, -4, 0, 2.5]) == [5.0, -4.0, 0.0, 2.5]
    assert m.average == 2.5


def test_reset_starts_new_warmup():
    m = MovingAverage(3)
    m.extend([3, 6, 9])
    old = m.snapshot()
    m.reset()
    assert m.average is None and m.count == 0
    assert m.window == 3 and m.snapshot() == ()
    assert m.add(-6) == -6.0
    assert old == (3.0, 6.0, 9.0)


def test_validation_preserves_state():
    for window in (0, -1, True, 2.5, '2'):
        with pytest.raises(ValueError):
            MovingAverage(window)
    m = MovingAverage(3)
    m.add(6)
    for sample in (True, None, '3', float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            m.add(sample)
    assert m.snapshot() == (6.0,)
    assert m.average == 6.0


def test_extend_partial_commit_and_generator():
    m = MovingAverage(3)
    stream = iter([3, 9, 'bad', 100])
    with pytest.raises(ValueError):
        m.extend(stream)
    assert next(stream) == 100
    assert m.average == 6.0
    assert m.extend(x for x in (6, 12)) == [6.0, 9.0]
    assert m.extend([]) == []
