import pytest
from ini_reader import loads
from ini_writer import dumps


def test_writer_exact_format_and_empty_sections():
    data = {'server': {'host': 'local', 'port': '80'}, 'empty': {}, 'user': {'name': 'Ada'}}
    assert dumps(data) == '[server]\nhost=local\nport=80\n\n[empty]\n\n[user]\nname=Ada\n'
    assert dumps({}) == ''
    assert dumps({'empty': {}}) == '[empty]\n'


def test_reader_whitespace_comments_and_crlf():
    text = ' # heading\r\n [ main ] \r\n key = hello world \r\n ; ignored\r\n empty=\r\n'
    assert loads(text) == {'main': {'key': 'hello world', 'empty': ''}}
    assert loads(' ; comment\n# another\n') == {}


def test_repeated_sections_and_key_order():
    data = loads('[b]\nx=1\ny=2\n[a]\nk=3\n[b]\nx=4\nz=5\n')
    assert list(data) == ['b', 'a']
    assert list(data['b']) == ['x', 'y', 'z']
    assert data['b'] == {'x': '4', 'y': '2', 'z': '5'}


def test_literal_values_and_round_trip():
    data = {'paths': {'eq': 'a=b=c', 'hash': '#tag ; note', 'quote': '"hello"', 'empty': '', 'path': r'C:\tmp'}, 'unused': {}}
    assert loads(dumps(data)) == data
    assert loads('[s]\nx=value # literal ; too\n')['s']['x'] == 'value # literal ; too'


def test_reader_invalid_syntax():
    for text in ['x=1', '[]', '[a', '[a] trailing', '[[a]]', '[a]\nbad', '[a]\n=empty', '[a]\nx=a\rb']:
        with pytest.raises(ValueError):
            loads(text)


def test_writer_invalid_names_and_values():
    bad = [{'': {}}, {' a': {}}, {'a]': {}}, {'a\nb': {}}, {'s': {'': 'v'}}, {'s': {' x': 'v'}}, {'s': {'a=b': 'v'}}, {'s': {'#x': 'v'}}, {'s': {';x': 'v'}}, {'s': {'[x': 'v'}}, {'s': {'x': ' v'}}, {'s': {'x': 'v '}}, {'s': {'x': 'a\nb'}}, {'s': {'x': 'a\rb'}}]
    for data in bad:
        with pytest.raises(ValueError):
            dumps(data)


def test_round_trip_file_and_independent_reads(tmp_path):
    data = {'z': {'unicode': 'café', 'zero': '0'}, 'a': {}}
    path = tmp_path / 'settings.ini'
    path.write_text(dumps(data), encoding='utf-8')
    first = loads(path.read_text(encoding='utf-8'))
    first['z']['zero'] = 'changed'
    assert loads(path.read_text(encoding='utf-8')) == data
    assert data['z']['zero'] == '0'
