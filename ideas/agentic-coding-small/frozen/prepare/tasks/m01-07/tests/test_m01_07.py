import pytest
from glob_matcher import GlobPattern, glob_match

def test_literal_and_whole_string():
    p = GlobPattern('a.b[0]')
    assert p.matches('a.b[0]') is True
    assert p.matches('xa.b[0]') is False
    assert p.matches('aXb0') is False
    assert glob_match('Ab', 'ab') is False

def test_empty_and_star():
    assert glob_match('', '') is True
    assert glob_match('', 'x') is False
    assert glob_match('***', '') is True
    assert glob_match('*', 'a/b\nc') is True
    assert glob_match('a*', 'a') is True

def test_question_character_rules():
    p = GlobPattern('?')
    for text in ('é', '/', '\n'):
        assert p.matches(text) is True
    assert p.matches('') is False and p.matches('ab') is False
    assert glob_match('??', 'e\u0301') is True

def test_escaped_metacharacters():
    assert glob_match(r'a\*b\?c', 'a*b?c') is True
    assert glob_match(r'a\*b\?c', 'axybzc') is False
    assert glob_match(r'\\*', '\\path') is True
    assert glob_match(r'\a', 'a') is True

def test_invalid_escape():
    with pytest.raises(ValueError):
        GlobPattern('abc' + '\\')
    with pytest.raises(ValueError):
        glob_match('\\', '')

def test_filter_order_and_reuse():
    p = GlobPattern('a*?b')
    assert p.filter(iter(['ab', 'axb', 'a/b', 'axb', 'ac', 'axyb'])) == ['axb', 'a/b', 'axb', 'axyb']
    assert p.matches('ab') is False
    assert p.matches('a\nb') is True

def test_ambiguous_star_paths():
    assert glob_match('*ab*ab', 'zzabab') is True
    assert glob_match('*ab*ab', 'zzaba') is False
    pattern = '*a' * 100 + 'b'
    assert glob_match(pattern, 'a' * 150 + 'b') is True
    assert glob_match(pattern, 'a' * 150 + 'c') is False
