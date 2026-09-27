import pytest
from csv_sniffer import sniff_column, sniff_csv

def test_missing_only():
    assert sniff_column(['', ' ', '	']) == 'unknown'
    assert sniff_column([]) == 'unknown'

def test_missing_boolean_and_integer():
    assert sniff_column([' TRUE ', '', 'false']) == 'bool'
    assert sniff_column(['', '+12', '-03', '0']) == 'int'
    assert sniff_column(['true', '1']) == 'str'

def test_numeric_widening():
    assert sniff_column(['1', '2.5', '-3']) == 'float'
    assert sniff_column(['1', '.5', '2.', '-1E+3']) == 'float'
    for value in ('NaN', 'inf', '1_000', 'hello'):
        assert sniff_column(['1', value]) == 'str'

def test_csv_blanks_and_header_only():
    assert sniff_csv('name,count\nAda,1\nBob,\n\nCy,2\n') == {'name': 'str', 'count': 'int'}
    assert sniff_csv('a,b\n') == {'a': 'unknown', 'b': 'unknown'}
    assert sniff_csv('') == {}

def test_csv_numeric_and_quoted_cells():
    assert sniff_csv('label,value\n"a,b",1\n"two\nlines",2.5\n') == {'label': 'str', 'value': 'float'}
    assert sniff_csv(' x ,flag\n3,true\n,false\n') == {' x ': 'int', 'flag': 'bool'}

def test_validation_with_missing_cells():
    assert sniff_csv('a,b\n,4\n') == {'a': 'unknown', 'b': 'int'}
    for text in ('a,a\n1,2\n', 'a, \n1,2\n', '\n', 'a,b\n1\n', 'a\n1,2\n'):
        with pytest.raises(ValueError):
            sniff_csv(text)
