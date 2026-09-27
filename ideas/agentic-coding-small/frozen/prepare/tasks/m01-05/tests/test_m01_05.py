import pytest
from scheduler import EventScheduler, next_occurrence

def test_next_occurrence_boundaries():
    assert next_occurrence(5, 3, 4) == 5
    assert next_occurrence(5, 3, 5) == 8
    assert next_occurrence(5, 3, 10) == 11
    assert next_occurrence(-4, 2, -1) == 0
    for interval in (0, -1):
        with pytest.raises(ValueError):
            next_occurrence(0, interval, -1)

def test_once_and_empty_advance():
    s = EventScheduler()
    assert s.now == 0 and s.pending() == []
    s.add('later', 4, 'b')
    s.add('first', 2)
    assert s.run_until(2) == [(2, 'first', None)]
    assert s.now == 2 and s.pending() == [(4, 'later')]
    assert s.run_until(8) == [(4, 'later', 'b')]
    assert s.run_until(9) == [] and s.now == 9

def test_recurrences_and_split_runs():
    s = EventScheduler()
    s.add('r', 0, 'x', interval=3)
    assert s.run_until(6) == [(0, 'r', 'x'), (3, 'r', 'x'), (6, 'r', 'x')]
    assert s.run_until(6) == []
    assert s.pending() == [(9, 'r')]
    assert s.run_until(10) == [(9, 'r', 'x')]

def test_stable_ties_with_recurring_events():
    s = EventScheduler()
    s.add('z', 1, interval=2)
    s.add('a', 3)
    s.add('m', 1, interval=2)
    assert s.pending() == [(1, 'z'), (1, 'm'), (3, 'a')]
    assert s.run_until(3) == [(1, 'z', None), (1, 'm', None), (3, 'z', None), (3, 'a', None), (3, 'm', None)]

def test_cancel_and_reuse():
    s = EventScheduler()
    s.add('x', 2, interval=1)
    s.add('y', 2)
    assert s.cancel('x') is True and s.cancel('x') is False
    s.add('x', 2)
    assert s.run_until(2) == [(2, 'y', None), (2, 'x', None)]
    s.add('x', 2, 'again')
    assert s.run_until(2) == [(2, 'x', 'again')]

def test_invalid_operations_preserve_state():
    s = EventScheduler()
    s.add('x', 5)
    s.run_until(3)
    for action in (lambda: s.add('x', 6), lambda: s.add('past', 2), lambda: s.add('bad', 4, interval=0), lambda: s.add('bad', 4, interval=-2), lambda: s.run_until(2)):
        with pytest.raises(ValueError):
            action()
    assert s.now == 3 and s.pending() == [(5, 'x')]

def test_payload_identity_and_pending_copy():
    s = EventScheduler()
    payload = {'job': 1}
    s.add('x', 0, payload)
    pending = s.pending()
    pending.clear()
    emitted = s.run_until(0)
    assert emitted == [(0, 'x', payload)] and emitted[0][2] is payload
