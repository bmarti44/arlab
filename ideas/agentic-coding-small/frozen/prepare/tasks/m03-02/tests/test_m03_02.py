import pytest
from identifiers import IdentifierConverter


def test_acronym_boundaries():
    c = IdentifierConverter()
    assert c.split('HTTPServer2_ID') == ['http', 'server', '2', 'id']
    assert c.split('XMLHTTPParser') == ['xmlhttp', 'parser']
    assert c.split('ABc') == ['a', 'bc']


def test_digit_groups_and_chunks():
    c = IdentifierConverter()
    assert c.split('a12B') == ['a', '12', 'b']
    assert c.split('version_2_value') == ['version', '2', 'value']
    assert c.to_snake('XML2HTTPParser99') == 'xml_2_http_parser_99'


def test_snake_and_single_tokens():
    c = IdentifierConverter()
    assert c.to_snake('already_snake') == 'already_snake'
    assert c.to_snake('XML') == 'xml'
    assert c.to_snake('x') == 'x'
    assert c.to_snake('getURLValue') == 'get_url_value'


def test_camel_modes():
    c = IdentifierConverter()
    assert c.to_camel('HTTP_server_2_ID') == 'httpServer2Id'
    assert c.to_camel('HTTP_server_2_ID', True) == 'HttpServer2Id'
    assert c.to_camel('a_12_b') == 'a12B'
    assert c.to_camel('X', True) == 'X'


def test_dispatch():
    c = IdentifierConverter()
    assert [c.convert('SomeHTTPValue', s) for s in ('snake', 'camel', 'pascal')] == [
        'some_http_value', 'someHttpValue', 'SomeHttpValue']
    with pytest.raises(ValueError):
        c.convert('valid', 'Camel')


def test_invalid_identifiers():
    c = IdentifierConverter()
    assert c.split('ok') == ['ok']
    for name in ('', '_a', 'a_', 'a__b', '2a', 'a-b', 'a b', 'café', 'a\n'):
        for method in (c.split, c.to_snake, c.to_camel):
            with pytest.raises(ValueError):
                method(name)
        with pytest.raises(ValueError):
            c.convert(name, 'pascal')
