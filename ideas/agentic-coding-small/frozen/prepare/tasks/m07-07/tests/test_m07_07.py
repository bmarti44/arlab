import pytest
from caches import Cache, lru_cache, lfu_cache


def test_lru_hits_control_eviction():
    cache = lru_cache(2)
    assert cache.put("a", 1) is None
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b", "absent") == "absent"
    assert cache.snapshot() == [("a", 1), ("c", 3)]


def test_lfu_frequency_overrides_recency():
    cache = lfu_cache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.get("a")
    cache.get("b")
    cache.put("c", 3)
    assert cache.snapshot() == [("a", 1), ("c", 3)]
    assert cache.get("b") is None


def test_lfu_tie_uses_recency_and_update_counts():
    cache = lfu_cache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.get("b")
    cache.put("c", 3)
    assert cache.snapshot() == [("b", 2), ("c", 3)]
    cache.put("c", 30)
    cache.put("d", 4)
    assert cache.snapshot() == [("c", 30), ("d", 4)]


def test_updates_and_none_values():
    for factory in (lru_cache, lfu_cache):
        cache = factory(2)
        cache.put("a", None)
        cache.put("b", 2)
        assert cache.get("a", "missing") is None
        assert cache.snapshot() == [("b", 2), ("a", None)]
        cache.put("b", 20)
        assert cache.snapshot() == [("a", None), ("b", 20)]
        assert cache.get("missing", 99) == 99
        assert cache.snapshot() == [("a", None), ("b", 20)]
        cache.put("c", 3)
        assert cache.snapshot() == [("b", 20), ("c", 3)]


def test_delete_reinsert_resets_frequency():
    cache = lfu_cache(2)
    cache.put("x", 1)
    cache.get("x")
    cache.get("x")
    assert cache.delete("x") is True
    assert cache.delete("x") is False
    cache.put("a", 2)
    cache.get("a")
    cache.put("x", 3)
    cache.put("b", 4)
    assert cache.snapshot() == [("a", 2), ("b", 4)]


def test_zero_capacity_and_validation():
    for factory in (lru_cache, lfu_cache):
        cache = factory(0)
        cache.put("x", 1)
        assert cache.get("x", 7) == 7
        assert cache.delete("x") is False
        assert cache.snapshot() == []
        for capacity in (-1, 1.5):
            with pytest.raises(ValueError):
                factory(capacity)
    with pytest.raises(ValueError):
        Cache(1, "FIFO")


def test_snapshot_does_not_touch_and_factories_are_independent():
    for factory in (lru_cache, lfu_cache):
        cache, other = factory(2), factory(2)
        cache.put("a", 1)
        cache.put("b", 2)
        old = cache.snapshot()
        old.clear()
        assert cache.snapshot() == [("a", 1), ("b", 2)]
        assert other.snapshot() == []
        cache.put("c", 3)
        assert cache.snapshot() == [("b", 2), ("c", 3)]


def test_one_slot_and_direct_constructor():
    for policy in ("lru", "lfu"):
        cache = Cache(1, policy)
        cache.put("x", 1)
        cache.put("x", 2)
        assert cache.get("x") == 2
        cache.put("y", 3)
        assert cache.snapshot() == [("y", 3)]
        assert cache.delete("y") is True
        assert cache.snapshot() == []
