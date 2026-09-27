import pytest
from t9 import T9Index


def test_collisions_and_normalization():
    index = T9Index(["Tree", "used", "TREE", "trie", "zoo"])
    assert index.lookup("8733") == ["tree", "used"]
    assert index.lookup("8743") == ["trie"]
    assert index.lookup("999") == []


def test_add_remove_and_return_values():
    index = T9Index()
    assert index.add("Home") is True
    assert index.add("HOME") is False
    assert index.add("good") is True
    assert index.lookup("4663") == ["good", "home"]
    assert index.remove("hOmE") is True
    assert index.remove("home") is False
    assert index.lookup("4663") == ["good"]


def test_prefix_lexical_order_and_limits():
    index = T9Index(["cat", "bat", "car", "apple", "dog", "cab"])
    assert index.complete("22") == ["bat", "cab", "car", "cat"]
    assert index.complete("2", 3) == ["apple", "bat", "cab"]
    assert index.complete("2", 0) == []
    assert index.complete("999", 20) == []


def test_empty_queries_and_independent_results():
    index = T9Index(["b", "a", "d"])
    assert index.lookup("") == []
    result = index.complete("")
    assert result == ["a", "b", "d"]
    result.clear()
    exact = index.lookup("2")
    exact.append("fake")
    assert index.lookup("2") == ["a", "b"]
    assert index.complete("", 1) == ["a"]


def test_invalid_words_leave_index_intact():
    index = T9Index(["ok"])
    for word in ("", "has space", "abc1", "é", "K", "a-b"):
        with pytest.raises(ValueError):
            index.add(word)
        with pytest.raises(ValueError):
            index.remove(word)
        with pytest.raises(ValueError):
            T9Index([word])
    assert index.complete("") == ["ok"]


def test_invalid_queries_and_limit():
    index = T9Index(["a"])
    for query in ("1", "20", "2 3", "２", "x"):
        with pytest.raises(ValueError):
            index.lookup(query)
        with pytest.raises(ValueError):
            index.complete(query)
    for limit in (-1, 1.5, "2"):
        with pytest.raises(ValueError):
            index.complete("", limit)


def test_full_keypad_and_default_limit():
    words = ["adgjmptw", "behknqux", "cfilorvy", "sz"] + ["a" * i for i in range(1, 13)]
    index = T9Index(iter(words))
    assert index.lookup("23456789") == ["adgjmptw", "behknqux", "cfilorvy"]
    assert index.lookup("79") == ["sz"]
    assert index.complete("2") == ["a" * i for i in range(1, 11)]
