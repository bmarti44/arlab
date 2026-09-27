import pytest
from query_parser import parse_query
from query_engine import run_query


def test_parser_structure_and_empty_query():
    assert parse_query(' \t ') == {'where': [], 'select': None, 'order': None}
    assert parse_query('where age>=-02 and name != "old friend" select name,age order age desc') == {
        'where': [('age', '>=', -2), ('name', '!=', 'old friend')],
        'select': ['name', 'age'], 'order': ('age', 'desc')}


def test_filter_conjunction_and_comparison_operators():
    rows = [{'x': i} for i in range(5)]
    cases = [('=', [2]), ('!=', [0, 1, 3, 4]), ('<', [0, 1]), ('<=', [0, 1, 2]), ('>', [3, 4]), ('>=', [2, 3, 4])]
    for op, values in cases:
        assert run_query(rows, f'where x {op} 2') == [{'x': x} for x in values]
    assert run_query(rows, 'where x >= 1 and x < 3') == [{'x': 1}, {'x': 2}]


def test_strings_quotes_and_type_sensitive_filter():
    rows = [{'v': 2}, {'v': '2'}, {}, {'v': 'a,b = c'}, {'v': r'a\b'}]
    assert run_query(rows, 'where v = "2"') == [{'v': '2'}]
    assert run_query(rows, 'where v != 2') == []
    assert run_query(rows, "where v = 'a,b = c'") == [{'v': 'a,b = c'}]
    assert run_query(rows, r"where v = 'a\b'") == [{'v': r'a\b'}]
    assert run_query([{'s': 'apple'}, {'s': 'pear'}], "where s < 'banana'") == [{'s': 'apple'}]


def test_projection_missing_fields_and_independence():
    rows = [{'a': 1, 'b': 2}, {'b': 3}]
    result = run_query(rows, 'select b,a')
    assert result == [{'b': 2, 'a': 1}, {'b': 3, 'a': None}]
    assert list(result[0]) == ['b', 'a']
    result[0]['b'] = 99
    full = run_query(rows, '')
    assert full == rows and full is not rows and full[0] is not rows[0]
    full[0]['a'] = 10
    assert rows == [{'a': 1, 'b': 2}, {'b': 3}]


def test_stable_sort_and_missing_values_last():
    rows = [{'id': 'a', 'n': 2}, {'id': 'b'}, {'id': 'c', 'n': 1}, {'id': 'd', 'n': 2}, {'id': 'e'}]
    assert [r['id'] for r in run_query(rows, 'order n asc')] == ['c', 'a', 'd', 'b', 'e']
    assert [r['id'] for r in run_query(rows, 'order n desc')] == ['a', 'd', 'c', 'b', 'e']
    assert [r['id'] for r in rows] == ['a', 'b', 'c', 'd', 'e']


def test_all_clauses_sort_before_projection():
    rows = [{'name': 'low', 'score': 1}, {'name': 'high', 'score': 9}, {'name': 'mid', 'score': 5}]
    assert run_query(rows, 'where score > 1 select name order score desc') == [{'name': 'high'}, {'name': 'mid'}]
    assert run_query(rows, 'where score > 99 select name') == []
    assert run_query([], 'order score asc') == []


def test_invalid_syntax_in_parser_and_engine():
    bad = ['where', 'where x == 1', 'where x = nope', 'where x = +1', "where x = 'bad", 'where x = 1 and', 'select', 'select a,', 'select a,a', 'select where', 'order a', 'order a up', 'order a asc select a', 'select a where x = 1', 'SELECT a', 'select a;', 'where x = 1 where y = 2', 'select a select b', 'where x=1and y=2', 'where x=1 select a,b,', 'where x=1 order x ASC']
    for query in bad:
        with pytest.raises(ValueError):
            parse_query(query)
        with pytest.raises(ValueError):
            run_query([], query)


def test_strings_containing_other_quote_and_keywords():
    query = "where text = \"it's where, and order\" select text"
    rows = [{'text': "it's where, and order"}, {'text': 'other'}]
    assert run_query(rows, query) == [rows[0]]
    assert parse_query('select _id,Field2')['select'] == ['_id', 'Field2']
