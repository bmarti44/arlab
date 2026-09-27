import pytest
from worddiff import WordDiff, diff_words, normalize_words


def test_normalization_rules():
    words = ['  A ', '', '  ', 'two words', 'Straße']
    assert normalize_words(words) == ['A', 'two words', 'Straße']
    assert normalize_words(iter(words), True) == ['a', 'two words', 'strasse']
    assert words[0] == '  A '


def test_multiplicity_and_stable_order():
    assert diff_words(['a', 'b', 'a', 'd'], ['b', 'a', 'c', 'b']) == {
        'added': ['c', 'b'], 'removed': ['a', 'd'], 'unchanged': ['a', 'b']}
    assert diff_words(['x'], ['x', 'z', 'x'])['added'] == ['z', 'x']


def test_casefold_and_case_sensitive_modes():
    assert diff_words(['Straße', ' A ', 'a'], ['STRASSE', 'a'], True) == {
        'added': [], 'removed': ['a'], 'unchanged': ['strasse', 'a']}
    assert diff_words(['A'], ['a']) == {'added': ['a'], 'removed': ['A'], 'unchanged': []}


def test_snapshot_and_generator_inputs():
    before = [' a ', 'b']
    diff = WordDiff(before, (x for x in ['b', 'c']))
    before.clear()
    result = diff.result()
    result['added'].append('bad')
    result['removed'].clear()
    assert diff.result() == {'added': ['c'], 'removed': ['a'], 'unchanged': ['b']}
    assert diff.has_changes() is True


def test_reverse_recomputes_order():
    diff = WordDiff(['a', 'b', 'a'], ['b', 'a', 'c'])
    reverse = diff.reversed()
    assert isinstance(reverse, WordDiff)
    assert reverse.result() == {'added': ['a'], 'removed': ['c'], 'unchanged': ['b', 'a']}
    assert reverse.reversed().result() == diff.result()


def test_format_and_empty_comparison():
    diff = WordDiff(['old', 'keep', 'old'], ['keep', 'new'])
    assert diff.format() == '- old\n- old\n+ new\n= keep'
    same = WordDiff([' b ', 'a'], ['a', 'b'])
    assert same.has_changes() is False
    assert same.format() == '= b\n= a'
    empty = WordDiff(['  '], [])
    assert empty.result() == {'added': [], 'removed': [], 'unchanged': []}
    assert empty.has_changes() is False
    assert empty.format() == ''


def test_nonstring_elements_rejected():
    with pytest.raises(TypeError):
        normalize_words(['ok', 2])
    with pytest.raises(TypeError):
        diff_words(['ok'], [None])
    with pytest.raises(TypeError):
        WordDiff([False], [])
