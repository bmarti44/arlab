import pytest
from markdown_table import escape_cell, format_table

def test_escape_order_and_newlines():
    assert escape_cell('a\\b|c\r\nd\re\nf') == 'a\\\\b\\|c<br>d<br>e<br>f'
    assert escape_cell(None) == ''
    assert escape_cell(42) == '42'
    assert escape_cell(' x ') == ' x '

def test_default_table():
    assert format_table(['Name', 'Qty'], [['tea', 2], ['coffee', 10]]) == (
        '| Name   | Qty |\n'
        '| :----- | :-- |\n'
        '| tea    | 2   |\n'
        '| coffee | 10  |'
    )

def test_alignment_and_center_odd_padding():
    assert format_table(['L', 'R', 'C'], [['long', 20, 'abcd'], ['x', 1, 'x']], ['left', 'right', 'center']) == (
        '| L    |   R |  C   |\n'
        '| :--- | --: | :--: |\n'
        '| long |  20 | abcd |\n'
        '| x    |   1 |  x   |'
    )

def test_empty_rows_and_minimum_width():
    assert format_table(['', None], [], ['center', 'right']) == '|     |     |\n| :-: | --: |'
    assert format_table(['é'], []) == '| é   |\n| :-- |'

def test_width_after_escaping():
    assert format_table(['A|B'], [['x\ny']]) == '| A\\|B   |\n| :----- |\n| x<br>y |'
    assert format_table(['h'], [[None]]) == '| h   |\n| :-- |\n|     |'

def test_reject_invalid_shapes_and_alignments():
    for args in (([], []), (['a'], [[1, 2]]), (['a', 'b'], [[]]), (['a'], [], []), (['a'], [], ['LEFT']), (['a'], [], ['left', 'right'])):
        with pytest.raises(ValueError):
            format_table(*args)

def test_input_lists_are_unchanged():
    headers, rows, alignments = ['a'], [['x|y'], [None]], ['right']
    before = (headers[:], [r[:] for r in rows], alignments[:])
    result = format_table(headers, rows, alignments)
    assert isinstance(result, str)
    assert (headers, rows, alignments) == before
