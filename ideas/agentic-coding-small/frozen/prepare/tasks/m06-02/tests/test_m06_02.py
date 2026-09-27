import pytest
from shell_tokens import ShellTokenizer


def test_whitespace_and_empty_input():
    assert ShellTokenizer('').tokens() == []
    assert ShellTokenizer(' \t\r\n ').tokens() == []
    assert ShellTokenizer('  one\ttwo\rthree\nfour ').tokens() == ['one', 'two', 'three', 'four']


def test_fragments_and_empty_quotes():
    assert ShellTokenizer("ab' cd'\"ef\" '' \"\" x''y").tokens() == ['ab cdef', '', '', 'xy']


def test_single_quotes_are_literal():
    text = "'a\\b \"c\"' '$HOME # |'"
    assert ShellTokenizer(text).tokens() == ['a\\b "c"', '$HOME # |']


def test_escapes_outside_and_inside_double_quotes():
    text = 'a\\ b "c\\\"d" x\\\\y '
    assert ShellTokenizer(text).tokens() == ['a b', 'c"d', 'x\\y']
    assert ShellTokenizer('a\\\nb "x\\qy"').tokens() == ['a\nb', 'xqy']


def test_errors_for_unfinished_fragments():
    for text in ["'abc", '"abc', 'abc\\', '"abc\\', "ok x'bad"]:
        with pytest.raises(ValueError):
            ShellTokenizer(text).tokens()


def test_literals_and_repeated_calls():
    parser = ShellTokenizer('echo #tag $HOME | > file')
    first = parser.tokens()
    assert first == ['echo', '#tag', '$HOME', '|', '>', 'file']
    first.append('changed')
    assert parser.tokens() == ['echo', '#tag', '$HOME', '|', '>', 'file']
    assert parser.tokens() is not parser.tokens()
