import pytest
from trie_nodes import normalize_word
from autocomplete import Trie


def test_normalization_and_empty_trie():
    assert normalize_word('HeLLo') == 'hello'
    assert normalize_word('', True) == ''
    t = Trie()
    assert t.prefix_search('') == [] and t.autocomplete('') == []
    assert not t.contains('x') and not t.remove('x')
    for word in ('', 'hello ', 'a-b', 'café', 'abc1', None):
        with pytest.raises(ValueError):
            normalize_word(word)


def test_insert_accumulates_and_checks_complete_words():
    t = Trie()
    assert t.add('Cart') == 1
    assert t.add('CART', 3) == 4
    assert t.contains('cart') is True
    assert t.contains('car') is False
    assert t.add('car', 2) == 2
    assert t.autocomplete('CAR') == [('cart', 4), ('car', 2)]


def test_prefix_order_and_empty_prefix():
    t = Trie()
    for word in ('dog', 'cart', 'car', 'cat', 'apple'):
        t.add(word)
    assert t.prefix_search('ca') == ['car', 'cart', 'cat']
    assert t.prefix_search('CAR') == ['car', 'cart']
    assert t.prefix_search('') == ['apple', 'car', 'cart', 'cat', 'dog']
    assert t.prefix_search('z') == []


def test_ranking_ties_limits_and_default():
    t = Trie()
    for word, score in [('bat', 3), ('bar', 3), ('bag', 5), ('banana', 1), ('bake', 2), ('ball', 1)]:
        t.add(word, score)
    assert t.autocomplete('b') == [('bag', 5), ('bar', 3), ('bat', 3), ('bake', 2), ('ball', 1)]
    assert t.autocomplete('', 2) == [('bag', 5), ('bar', 3)]
    assert t.autocomplete('b', 0) == []
    assert t.autocomplete('missing', 20) == []


def test_remove_prefix_preserves_descendants_and_readd():
    t = Trie()
    t.add('a', 9)
    t.add('an', 4)
    t.add('ant', 2)
    assert t.remove('AN') is True
    assert t.contains('an') is False and t.contains('ant') is True
    assert t.remove('an') is False
    assert t.autocomplete('a') == [('a', 9), ('ant', 2)]
    assert t.add('an', 1) == 1
    assert t.remove('ant') is True
    assert t.prefix_search('a') == ['a', 'an']


def test_invalid_inputs_are_atomic():
    t = Trie()
    t.add('valid', 2)
    for weight in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            t.add('valid', weight)
    for word in ('', 'a b', 'a1', 'é', None):
        for method in (t.add, t.remove, t.contains):
            with pytest.raises(ValueError):
                method(word)
    for prefix in ('a b', 'a1', None):
        with pytest.raises(ValueError):
            t.prefix_search(prefix)
        with pytest.raises(ValueError):
            t.autocomplete(prefix, 0)
    for limit in (-1, True, 1.5):
        with pytest.raises(ValueError):
            t.autocomplete('', limit)
    assert t.autocomplete('') == [('valid', 2)]


def test_result_lists_are_independent():
    t = Trie()
    t.add('one')
    t.add('only', 2)
    matches = t.prefix_search('on')
    ranked = t.autocomplete('on')
    matches.clear()
    ranked.append(('fake', 100))
    t.remove('one')
    assert t.prefix_search('on') == ['only']
    assert t.autocomplete('on') == [('only', 2)]
