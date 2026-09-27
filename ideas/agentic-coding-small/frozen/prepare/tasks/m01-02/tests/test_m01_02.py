import pytest
from text_layout import TextWrapper

def test_words_and_empty():
    w = TextWrapper(8)
    assert w.words(' a\tb\n c  ') == ['a', 'b', 'c']
    assert w.wrap(' \t\n') == []
    assert w.justify('') == [] and w.format('') == ''

def test_greedy_boundary():
    w = TextWrapper(7)
    assert w.wrap('one two a bb c') == ['one two', 'a bb c']
    assert w.wrap('a  b\nc') == ['a b c']

def test_oversized_words():
    w = TextWrapper(3)
    assert w.wrap('a elephant b c') == ['a', 'elephant', 'b c']
    assert w.justify('a elephant b c') == ['a', 'elephant', 'b c']

def test_leftmost_extra_spaces():
    w = TextWrapper(10)
    assert w.justify('aa b cc dddd e') == ['aa   b  cc', 'dddd e']
    assert w.format('aa b cc dddd e', justify=True) == 'aa   b  cc\ndddd e'

def test_last_line_and_repeated_calls():
    w = TextWrapper(8)
    assert w.justify('a bb c dddd') == ['a  bb  c', 'dddd']
    assert w.justify('a b') == ['a b']
    assert w.format('one two three') == 'one two\nthree'

def test_width_one_unicode_and_invalid():
    w = TextWrapper(1)
    assert w.wrap('é x ab') == ['é', 'x', 'ab']
    for width in (0, -1):
        with pytest.raises(ValueError):
            TextWrapper(width)
