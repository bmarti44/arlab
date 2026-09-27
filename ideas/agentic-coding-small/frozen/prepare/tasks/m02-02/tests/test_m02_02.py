import pytest
from query import Query


def test_duplicates_and_blanks():
    q = Query('?a=1&&a=2&flag&=x&')
    assert q.getall('a') == ['1', '2']
    assert q.getall('flag') == ['']
    assert q.getall('') == ['x']
    assert q.encode() == 'a=1&a=2&flag=&=x'


def test_decoding_and_encoding():
    q = Query('q=caf%C3%A9+tea&x=%2B%2F&v=a=b;c')
    assert q.getall('q') == ['café tea']
    assert q.getall('x') == ['+/']
    assert q.encode() == 'q=caf%C3%A9+tea&x=%2B%2F&v=a%3Db%3Bc'


def test_set_preserves_first_position():
    q = Query('z=0&a=1&b=2&a=3&z=4')
    assert q.set('a', 'new value') is None
    assert q.encode() == 'z=0&a=new+value&b=2&z=4'
    q.set('c', '')
    assert q.encode().endswith('&c=')


def test_append_and_getall_copy():
    q = Query()
    assert q.append('a+b', '%20') is None
    q.append('a+b', '')
    values = q.getall('a+b')
    values.clear()
    assert q.getall('a+b') == ['%20', '']
    assert q.getall('missing') == []
    assert q.encode() == 'a%2Bb=%2520&a%2Bb='


def test_permissive_escapes_and_prefix():
    q = Query('??x=%ZZ&bad=%FF&empty=&')
    assert q.getall('?x') == ['%ZZ']
    assert q.getall('bad') == ['�']
    assert q.encode() == '%3Fx=%25ZZ&bad=%EF%BF%BD&empty='
    assert Query('?&&').encode() == ''


def test_types_and_atomic_mutation():
    with pytest.raises(TypeError):
        Query(None)
    q = Query('a=1')
    with pytest.raises(TypeError):
        q.getall(1)
    for method in [q.append, q.set]:
        for args in [(1, 'x'), ('a', None)]:
            with pytest.raises(TypeError):
                method(*args)
    assert q.encode() == 'a=1'
