import pytest
from transaction_store import TransactionStore, diff

def test_base_operations_and_none():
    s = TransactionStore()
    assert s.snapshot() == {}
    assert s.set('a', None) is None and s.get('a') is None
    assert s.delete('a') is None
    for action in (lambda: s.get('a'), lambda: s.delete('a')):
        with pytest.raises(KeyError):
            action()
    assert s.snapshot() == {}

def test_copies_are_shallow():
    value = []
    original = {'x': value}
    s = TransactionStore(original)
    original['y'] = 2
    snap = s.snapshot()
    assert snap == {'x': []} and snap['x'] is value
    snap.clear()
    assert s.get('x') is value

def test_commit_and_rollback():
    s = TransactionStore({'x': 1})
    assert s.begin() == 1
    s.set('x', 2)
    s.set('y', 3)
    assert s.rollback() == 0 and s.snapshot() == {'x': 1}
    assert s.begin() == 1
    assert s.delete('x') == 1
    s.set('z', 4)
    assert s.commit() == 0 and s.snapshot() == {'z': 4}

def test_inner_commit_outer_rollback():
    s = TransactionStore({'x': 1})
    s.begin()
    s.set('x', 2)
    assert s.begin() == 2
    s.delete('x')
    s.set('y', 9)
    assert s.commit() == 1 and s.snapshot() == {'y': 9}
    assert s.rollback() == 0 and s.snapshot() == {'x': 1}

def test_inner_rollback_outer_commit():
    s = TransactionStore()
    s.begin()
    s.set('a', 2)
    s.begin()
    s.set('a', 7)
    assert s.begin() == 3
    s.set('b', 4)
    assert s.rollback() == 2
    assert s.rollback() == 1 and s.get('a') == 2
    assert s.commit() == 0 and s.snapshot() == {'a': 2}

def test_invalid_boundaries_and_missing_delete():
    s = TransactionStore({'x': 1})
    for action in (s.commit, s.rollback):
        with pytest.raises(RuntimeError):
            action()
    s.begin()
    with pytest.raises(KeyError):
        s.delete('missing')
    assert s.commit() == 0 and s.snapshot() == {'x': 1}

def test_diff_categories_and_inputs():
    before = {'same': [1], 'edit': 1, 'gone': None}
    after = {'same': [1], 'edit': None, 'new': 3}
    assert diff(before, after) == {'added': {'new': 3}, 'updated': {'edit': (1, None)}, 'removed': {'gone': None}}
    assert before == {'same': [1], 'edit': 1, 'gone': None}
    assert after == {'same': [1], 'edit': None, 'new': 3}
    assert diff({}, {}) == {'added': {}, 'updated': {}, 'removed': {}}
