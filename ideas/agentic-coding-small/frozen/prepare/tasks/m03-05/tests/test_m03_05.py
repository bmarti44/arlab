import pytest
from paths import PathStore, parse_path


def test_path_parser_escapes():
    assert parse_path(r'user.first\.name') == ['user', 'first.name']
    assert parse_path(r'a\\b.c') == ['a\\b', 'c']
    assert parse_path(' a . b ') == [' a ', ' b ']


def test_set_creates_and_replaces():
    s = PathStore()
    s.set('a.b.c', 4)
    assert s.get('a.b.c') == 4
    s.set('a.b', [1, None])
    assert s.snapshot() == {'a': {'b': [1, None]}}
    s.set(r'a.dot\.key', False)
    assert s.get(r'a.dot\.key') is False


def test_missing_default_and_none():
    s = PathStore({'x': None})
    assert s.get('x', 7) is None
    fallback = {'items': []}
    result = s.get('absent.deeper', fallback)
    result['items'].append(1)
    assert fallback == {'items': []}
    assert s.get('absent') is None


def test_deep_copy_boundaries():
    data = {'a': {'items': [1]}}
    s = PathStore(data)
    data['a']['items'].append(2)
    value = {'list': [3]}
    s.set('b', value)
    value['list'].append(4)
    s.get('a')['items'].clear()
    s.snapshot()['b']['list'].clear()
    assert s.snapshot() == {'a': {'items': [1]}, 'b': {'list': [3]}}


def test_delete_does_not_prune():
    s = PathStore({'a': {'b': 1}, 'other': 2})
    assert s.delete('a.b') is True
    assert s.delete('a.b') is False
    assert s.delete('missing.child') is False
    assert s.snapshot() == {'a': {}, 'other': 2}
    assert s.delete('a') is True


def test_non_dict_traversal_and_atomicity():
    s = PathStore({'a': {'b': [1]}, 'n': None})
    for path in ('a.b.c', 'n.x'):
        for op in (s.get, s.delete):
            with pytest.raises(TypeError):
                op(path)
        with pytest.raises(TypeError):
            s.set(path, 5)
    assert s.snapshot() == {'a': {'b': [1]}, 'n': None}


def test_invalid_syntax_before_mutation():
    s = PathStore({'a': 1})
    assert s.get('a') == 1
    for path in ('', '.a', 'a.', 'a..b', 'a\\', r'a\q'):
        with pytest.raises(ValueError):
            parse_path(path)
        for op in (s.get, s.delete):
            with pytest.raises(ValueError):
                op(path)
        with pytest.raises(ValueError):
            s.set(path, 5)
    assert s.snapshot() == {'a': 1}
