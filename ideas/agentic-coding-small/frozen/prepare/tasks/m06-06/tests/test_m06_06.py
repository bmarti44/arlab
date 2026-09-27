import pytest
from config_merge import Config, merge_config


def test_recursive_merge_and_replacement():
    base = {'db': {'host': 'local', 'port': 10}, 'mode': 'dev'}
    result = merge_config(base, {'db': {'port': 20}, 'mode': 'prod'})
    assert result == {'db': {'host': 'local', 'port': 20}, 'mode': 'prod'}
    assert base['db']['port'] == 10


def test_deletions_and_empty_parents():
    result = merge_config({'a': {'x': 1}, 'keep': None}, {'a': {'x': None}, 'absent': None})
    assert result == {'a': {}, 'keep': None}
    assert merge_config({'a': {'x': 1}}, {'a': {}}) == {'a': {'x': 1}}


def test_dict_over_scalar_and_list_replacement():
    result = merge_config({'a': 1, 'items': [1, 2]}, {'a': {'x': None, 'y': 2}, 'items': [None, {'z': None}]})
    assert result == {'a': {'y': 2}, 'items': [None, {'z': None}]}
    assert merge_config({'a': {'x': 1}}, {'a': False}) == {'a': False}


def test_merge_deep_independence():
    base = {'keep': [{'x': 1}]}
    override = {'new': [{'y': 2}]}
    result = merge_config(base, override)
    result['keep'][0]['x'] = 9
    result['new'][0]['y'] = 9
    assert base == {'keep': [{'x': 1}]}
    assert override == {'new': [{'y': 2}]}


def test_config_snapshot_and_apply():
    source = {'a': {'b': 1}, 'null': None}
    config = Config(source)
    source['a']['b'] = 8
    updated = config.apply({'a': {'b': 2}})
    assert updated is not config
    assert config.get('a.b') == 1
    assert updated.get('a.b') == 2
    assert config.get('null', 99) is None


def test_get_default_and_invalid_paths():
    config = Config({'a': 1, 'items': [8]})
    fallback = {'values': []}
    found = config.get('missing', fallback)
    found['values'].append(2)
    assert fallback == {'values': []}
    assert config.get('a.b', 'no') == 'no'
    assert config.get('items.0', 'no') == 'no'
    for path in ['', '.a', 'a.', 'missing..key']:
        with pytest.raises(ValueError):
            config.get(path)


def test_exports_and_found_values_are_copies():
    config = Config({'a': {'values': [1]}})
    config.get('a')['values'].append(2)
    exported = config.to_dict()
    exported['a']['values'].append(3)
    assert config.to_dict() == {'a': {'values': [1]}}
