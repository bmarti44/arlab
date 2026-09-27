import pytest
from circular_buffer import CircularBuffer

def test_empty_and_capacity():
    b = CircularBuffer(2)
    assert len(b) == 0 and b.snapshot() == []
    with pytest.raises(IndexError):
        b.pop()
    for capacity in (0, -3):
        with pytest.raises(ValueError):
            CircularBuffer(capacity)

def test_fifo_and_overwrite():
    b = CircularBuffer(2)
    assert b.append('a') is None
    assert b.append('b') is None
    assert b.append('c') == 'a'
    assert len(b) == 2 and b.snapshot() == ['b', 'c']
    assert b.pop() == 'b'
    assert b.pop() == 'c'

def test_extend_generator():
    b = CircularBuffer(3)
    assert b.extend(i for i in range(7)) == [0, 1, 2, 3]
    assert b.snapshot() == [4, 5, 6]
    assert b.extend([]) == []

def test_evicted_none():
    b = CircularBuffer(1)
    assert b.extend([None, 'x', None]) == [None, 'x']
    assert b.snapshot() == [None]
    assert b.pop() is None

def test_reuse_and_wraparound():
    b = CircularBuffer(3)
    b.extend([1, 2, 3])
    assert b.pop() == 1
    assert b.append(4) is None
    assert b.append(5) == 2
    assert [b.pop(), b.pop(), b.pop()] == [3, 4, 5]
    b.extend([6, 7])
    assert b.snapshot() == [6, 7]

def test_snapshot_independence():
    b = CircularBuffer(2)
    obj = {'x': 1}
    b.append(obj)
    snapshot = b.snapshot()
    assert snapshot[0] is obj
    snapshot.clear()
    assert len(b) == 1 and b.pop() is obj
