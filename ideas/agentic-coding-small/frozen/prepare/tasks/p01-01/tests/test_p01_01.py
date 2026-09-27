import pytest
from config import Config, parse

def test_parse_sections_and_literals():
    c = parse('  # heading\nroot = a=b # literal\n[ server ]\n host = local ; literal\nempty=\n; end')
    assert c.get('', 'root') == 'a=b # literal'
    assert c.get('server', 'host') == 'local ; literal'
    assert c.get('server', 'empty') == ''

def test_repeated_sections_and_case():
    c = parse('[A]\nx=1\n[B]\nx=2\n[A]\nx=3\nX=4')
    assert c.get('A', 'x') == '3'
    assert c.get('A', 'X') == '4'
    assert c.get('B', 'x') == '2'
    assert c.get('a', 'x') is None

def test_missing_defaults_and_empty_config():
    c = Config()
    marker = object()
    for method in (c.get, c.get_int, c.get_bool):
        assert method('missing', 'key', marker) is marker
        assert method('', 'key') is None
    c = parse('[known]\na=1')
    assert c.get_int('known', 'absent', 'unconverted') == 'unconverted'
    assert c.get_bool('known', 'absent', marker) is marker

def test_integer_getter():
    c = parse('a=+0012\nb=-9\nc=0\nd=1_000\ne=1.2\nf=\ng=１２')
    assert [c.get_int('', key) for key in 'abc'] == [12, -9, 0]
    for key in 'defg':
        with pytest.raises(ValueError):
            c.get_int('', key)

def test_boolean_getter():
    c = parse('a=TRUE\nb=Yes\nc=on\nd=1\ne=false\nf=NO\ng=Off\nh=0\ni=maybe\nj=')
    assert all(c.get_bool('', key) is True for key in 'abcd')
    assert all(c.get_bool('', key) is False for key in 'efgh')
    for key in 'ij':
        with pytest.raises(ValueError):
            c.get_bool('', key)

def test_reject_malformed_lines():
    for text in ('[broken', '[]', '[   ]', '[a[b]', '[a]junk', 'bare', '=value'):
        with pytest.raises(ValueError):
            parse(text)

def test_independent_parses_and_blank_input():
    a = parse('x=old')
    b = parse('x=new')
    assert a.get('', 'x') == 'old'
    assert b.get('', 'x') == 'new'
    assert parse(' \n# hi\n; bye').get('', 'x') is None
