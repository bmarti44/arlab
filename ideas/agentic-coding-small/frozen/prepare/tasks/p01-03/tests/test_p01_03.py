import pytest
from cache_entry import Entry
from lru_cache import LRUCache

def test_entry_expiration():
    assert Entry('x', None).expired(1000) is False
    assert Entry('x', 2).expired(1.999) is False
    assert Entry('x', 2).expired(2) is True
    assert Entry('x', 2).expired(3) is True

def test_lru_hits_and_eviction():
    c = LRUCache(2, lambda: 0)
    c.put('a', 1)
    c.put('b', 2)
    assert c.get('a') == 1
    c.put('c', 3)
    assert c.keys() == ['a', 'c']
    marker = object()
    assert c.get('b', marker) is marker
    assert c.get('missing') is None
    assert c.keys() == ['a', 'c']

def test_expiry_boundary_and_purge_before_eviction():
    now = [10.0]
    c = LRUCache(2, lambda: now[0])
    c.put('live', 1)
    c.put('short', 2, ttl=2)
    now[0] = 12
    c.put('new', 3)
    assert c.keys() == ['live', 'new']
    assert c.get('short', 'gone') == 'gone'
    assert len(c) == 2

def test_replacement_resets_ttl_and_recency():
    now = [0]
    c = LRUCache(2, lambda: now[0])
    c.put('a', 1, 1)
    c.put('b', 2)
    c.put('a', 3)
    assert c.keys() == ['b', 'a']
    now[0] = 2
    assert c.get('a') == 3
    c.put('a', 4, 2)
    now[0] = 4
    assert c.get('a') is None
    assert c.keys() == ['b']

def test_zero_ttl_delete_and_key_copy():
    now = [0]
    c = LRUCache(3, lambda: now[0])
    c.put('a', 1)
    c.put('b', 2, 1)
    result = c.keys()
    result.clear()
    assert c.keys() == ['a', 'b']
    c.put('a', 9, 0)
    c.put('new', 9, 0)
    assert c.keys() == ['b']
    now[0] = 1
    assert c.delete('b') is False
    assert c.delete('absent') is False
    c.put('x', None)
    assert c.delete('x') is True
    assert len(c) == 0

def test_validation_does_not_read_clock_or_mutate():
    calls = []
    def clock():
        calls.append(1)
        return 0
    for capacity in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            LRUCache(capacity, clock)
    c = LRUCache(1, clock)
    c.put('a', 1)
    calls.clear()
    for ttl in (-1, float('inf'), float('nan'), True, '1'):
        with pytest.raises(ValueError):
            c.put('a', 2, ttl)
    assert calls == []
    assert c.get('a') == 1
    assert calls == [1]

def test_reads_purge_all_and_instances_are_independent():
    now = [0]
    calls = []
    def clock():
        calls.append(1)
        return now[0]
    a, b = LRUCache(3, clock), LRUCache(1, clock)
    a.put('x', 1, 1)
    a.put('y', 2, 1)
    b.put('x', 9)
    now[0] = 1
    for operation in (lambda: a.get('absent'), lambda: a.delete('absent'), a.keys, lambda: len(a)):
        before = len(calls)
        operation()
        assert len(calls) == before + 1
    assert a.keys() == []
    assert b.get('x') == 9
