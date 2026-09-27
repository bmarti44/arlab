import pytest
from spreadsheet import Spreadsheet, normalize_cell, parse_formula


def test_names_and_formula_terms():
    assert normalize_cell(' aa12 ') == 'AA12'
    assert parse_formula(' = -a1 + 02 - B3 ') == [(-1, 'A1'), (1, 2), (-1, 'B3')]
    assert parse_formula('=+12') == [(1, 12)]
    assert parse_formula('=0') == [(1, 0)]
    for name in ['A0', 'A01', '1A', 'A 1', '', 'Ａ1']:
        with pytest.raises(ValueError):
            normalize_cell(name)


def test_formula_syntax_rejections():
    assert parse_formula('=1-2+3') == [(1, 1), (-1, 2), (1, 3)]
    for text in ['1', '=', '=A1+', '=A1 B2', '=1 2', '=A 1', '=A01', '=A1+-2',
                 '=--A1', '=(A1)', '=2*3', '=١', '=A1/2']:
        with pytest.raises(ValueError):
            parse_formula(text)


def test_dependencies_and_updates():
    s = Spreadsheet()
    assert s.set(' a1 ', -3) is None
    s.set('B2', '= A1 + 10')
    s.set('C3', '=B2 - A1 + 1')
    assert s.get('b2') == 7 and s.get('C3') == 11
    s.set('A1', 8)
    assert s.get('B2') == 18 and s.get('C3') == 11


def test_missing_references_and_delete():
    s = Spreadsheet()
    s.set('A1', '=B1+1')
    with pytest.raises(KeyError):
        s.get('A1')
    s.set('B1', 2)
    assert s.get('A1') == 3
    assert s.delete(' b1 ') is None
    with pytest.raises(KeyError):
        s.get('A1')
    with pytest.raises(KeyError):
        s.delete('B1')
    with pytest.raises(KeyError):
        s.get('C1')


def test_cycles_are_local_and_recoverable():
    s = Spreadsheet()
    s.set('A1', '=B1')
    s.set('B1', '=A1')
    s.set('Z1', 9)
    with pytest.raises(ValueError):
        s.get('A1')
    assert s.get('Z1') == 9
    s.set('B1', 4)
    assert s.get('A1') == 4
    s.set('Z1', '=Z1')
    with pytest.raises(ValueError):
        s.get('Z1')


def test_shared_dependency_is_not_cycle():
    s = Spreadsheet()
    s.set('A1', 5)
    s.set('B1', '=A1+A1')
    s.set('C1', '=A1+B1')
    s.set('D1', '=B1+C1-A1')
    assert s.get('D1') == 20
    assert s.get('D1') == 20


def test_failed_set_preserves_value_and_name_validation():
    s = Spreadsheet()
    s.set('A1', 7)
    with pytest.raises(ValueError):
        s.set('A1', '=2+')
    with pytest.raises(TypeError):
        s.set('A1', 2.5)
    with pytest.raises(TypeError):
        s.set('A1', True)
    assert s.get('A1') == 7
    for operation in [lambda: s.set('A0', 1), lambda: s.get('A0'), lambda: s.delete('A0')]:
        with pytest.raises(ValueError):
            operation()
