import pytest
from seats import SeatMap, parse_seat, seat_label


def test_helpers_and_dimensions():
    assert seat_label(25, 12) == 'Z12'
    assert parse_seat('B10', 2, 12) == (1, 10)
    for args in [(26, 1), (-1, 1), (True, 1), (0, 0)]:
        with pytest.raises(ValueError):
            seat_label(*args)
    for args in [(0, 3), (27, 3), (2, 0), (True, 4), (2, 2.0)]:
        with pytest.raises(ValueError):
            SeatMap(*args)


def test_reserve_deduplicate_numeric_order():
    m = SeatMap(2, 12)
    assert m.reserve(iter(['B2', 'A10', 'A2', 'A2']), 'Lee') == ['A2', 'A10', 'B2']
    assert list(m.snapshot()) == ['A2', 'A10', 'B2']
    assert m.reserve([], 'Lee') == []
    assert 'A2' not in m.available('A')


def test_atomic_conflict_and_bad_labels():
    m = SeatMap(2, 4)
    m.reserve(['A2'], 'Lee')
    for labels in (['A1', 'A2'], ['B1', 'C1'], ['A3', 'a1'], ['A3', 'A01'], ['A3', 'A0'], ['A3', 'A5'], ['A3', ' A1'], ['A3', 'A1\n']):
        with pytest.raises(ValueError):
            m.reserve(labels, 'Lee')
        assert m.snapshot() == {'A2': 'Lee'}


def test_release_and_snapshot_copy():
    m = SeatMap(2, 3)
    m.reserve(['B1', 'A3'], 'x')
    m.reserve(['A1'], 'y')
    copy = m.snapshot()
    copy['A2'] = 'z'
    assert m.release('x') == ['A3', 'B1']
    assert m.release('x') == []
    assert m.snapshot() == {'A1': 'y'}
    assert m.available('B') == ['B1', 'B2', 'B3']


def test_leftmost_block_and_fragmentation():
    m = SeatMap(1, 7)
    m.reserve(['A2', 'A6'], 'x')
    assert m.reserve_block('A', 2, 'y') == ['A3', 'A4']
    assert m.reserve_block('A', 2, 'z') is None
    assert m.reserve_block('A', 8, 'z') is None
    assert m.reserve_block('A', 1, 'z') == ['A1']


def test_block_validation_is_atomic():
    m = SeatMap(1, 2)
    assert m.available('A') == ['A1', 'A2']
    for args in [('a', 1, 'x'), ('B', 1, 'x'), ('AA', 1, 'x'), ('A', 0, 'x'), ('A', True, 'x'), ('A', 3, ''), ('A', 1, None)]:
        with pytest.raises(ValueError):
            m.reserve_block(*args)
    assert m.snapshot() == {}


def test_owners_and_row_validation():
    m = SeatMap(1, 2)
    assert m.reserve(['A1'], ' ') == ['A1']
    for owner in ('', None, 12):
        with pytest.raises(ValueError):
            m.reserve([], owner)
    for row in ('', 'AA', 'B', 'a'):
        with pytest.raises(ValueError):
            m.available(row)
