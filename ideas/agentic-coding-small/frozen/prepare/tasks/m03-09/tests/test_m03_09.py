from copy import deepcopy
import pytest
from elevator_state import new_state, validate_floor
from elevator import step


def test_initial_state_and_validation():
    assert new_state(1) == {'floors': 1, 'floor': 0, 'direction': 0, 'doors_open': False, 'requests': []}
    assert validate_floor(2, 3) is None
    for floors, start in ((0, 0), (True, 0), (3.0, 0), (3, -1), (3, 3), (3, True)):
        with pytest.raises(ValueError):
            new_state(floors, start)
    for floor in (-1, 3, True, 1.5):
        with pytest.raises(ValueError):
            validate_floor(floor, 3)


def test_idle_tie_goes_down_and_arrival_opens():
    state = new_state(5, 2)
    nxt = step(state, [3, 1, 3])
    assert nxt == {'floors': 5, 'floor': 1, 'direction': -1, 'doors_open': True, 'requests': [3]}
    assert state == new_state(5, 2)


def test_current_floor_and_door_cycle():
    state = step(new_state(4, 1), [1, 3])
    assert state['floor'] == 1 and state['doors_open'] is True
    assert state['direction'] == 0 and state['requests'] == [3]
    state = step(state, [1])
    assert state['floor'] == 1 and state['doors_open'] is False
    assert state['requests'] == [1, 3]
    state = step(state)
    assert state['doors_open'] is True and state['requests'] == [3]
    state = step(step(state))
    assert state['floor'] == 2 and state['direction'] == 1
    assert state['doors_open'] is False


def test_continue_sweep_then_reverse():
    state = step(new_state(6), [4])
    state = step(state, [0, 2])
    assert (state['floor'], state['direction'], state['doors_open'], state['requests']) == (2, 1, True, [0, 4])
    state = step(state)
    assert state['floor'] == 2 and not state['doors_open']
    state = step(step(state))
    assert (state['floor'], state['direction'], state['requests']) == (4, 1, [0])
    state = step(step(state))
    assert (state['floor'], state['direction'], state['doors_open']) == (3, -1, False)
    for _ in range(3):
        state = step(state)
    assert state['floor'] == 0 and state['doors_open'] is True
    assert state['direction'] == 0 and state['requests'] == []


def test_finish_closes_then_stays_idle():
    state = step(new_state(3), [1])
    assert state['floor'] == 1 and state['doors_open'] is True and state['direction'] == 0
    state = step(state)
    assert state == {'floors': 3, 'floor': 1, 'direction': 0, 'doors_open': False, 'requests': []}
    assert step(state) == state


def test_invalid_requests_do_not_mutate():
    state = step(new_state(5), [4])
    saved = deepcopy(state)
    for bad in (-1, 5, True, 2.5, '2'):
        with pytest.raises(ValueError):
            step(state, iter([2, bad]))
        assert state == saved


def test_request_copies_and_downward_sweep():
    state = step(new_state(6, 5), [1])
    nxt = step(state, (x for x in [5, 3, 3]))
    assert (nxt['floor'], nxt['direction'], nxt['doors_open'], nxt['requests']) == (3, -1, True, [1, 5])
    nxt['requests'].append(0)
    assert state['requests'] == [1]
    still = step(new_state(1))
    still['requests'].append(0)
    assert new_state(1)['requests'] == []
