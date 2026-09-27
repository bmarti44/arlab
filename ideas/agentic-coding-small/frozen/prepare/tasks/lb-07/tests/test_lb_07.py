import pytest
from todo_store import TodoStore
from todo_query import select


def test_add_normalizes_and_detaches():
    store = TodoStore()
    tags = [" Work ", "WORK", "urgent"]
    assert store.add("  Ship release  ", tags) == 1
    tags.append("new")
    assert store.get(1) == {"id": 1, "title": "Ship release", "done": False,
                            "tags": ["urgent", "work"]}
    snapshot = store.get(1)
    snapshot["tags"].clear()
    snapshot["title"] = "changed"
    assert store.get(1)["title"] == "Ship release"
    assert store.get(1)["tags"] == ["urgent", "work"]


def test_failed_add_does_not_consume_id():
    store = TodoStore()
    for title, tags in [("  ", []), ("ok", ["fine", " "])]:
        with pytest.raises(ValueError):
            store.add(title, tags)
    assert store.items() == []
    assert store.add("first") == 1
    assert store.remove(1) is True
    assert store.remove(1) is False
    assert store.add("second") == 2
    with pytest.raises(KeyError):
        store.get(1)


def test_completion_validation_and_items_copy():
    store = TodoStore()
    store.add("one", ["a"])
    store.add("two")
    assert store.set_done(1, True) is None
    for bad in [0, 1, "yes", None]:
        with pytest.raises(ValueError):
            store.set_done(1, bad)
    with pytest.raises(ValueError):
        store.set_done(99, 1)
    with pytest.raises(KeyError):
        store.set_done(99, False)
    rows = store.items()
    assert [r["id"] for r in rows] == [1, 2]
    rows[0]["tags"].append("b")
    rows[0]["done"] = False
    assert store.get(1)["done"] is True
    assert store.get(1)["tags"] == ["a"]


def test_query_combines_filters_with_unicode():
    store = TodoStore()
    store.add("Straße cleanup", ["WORK"])
    store.add("Street cleanup", ["work"])
    store.add("Straße review", ["home"])
    store.add("Straße archived", ["work"])
    store.set_done(4, True)
    rows = select(store, done=False, tag=" Work ", text=" STRASSE ")
    assert [row["id"] for row in rows] == [1]
    assert select(store, tag="wor") == []


def test_order_then_limit_and_empty_text():
    store = TodoStore()
    for title in ["one", "two", "three", "four"]:
        store.add(title)
    store.set_done(1, True)
    store.set_done(3, True)
    assert [r["id"] for r in select(store, text=" ")] == [2, 4, 1, 3]
    assert [r["id"] for r in select(store, limit=3)] == [2, 4, 1]
    assert [r["id"] for r in select(store, done=True)] == [1, 3]
    assert select(store, limit=0) == []


def test_query_validation_on_empty_store():
    store = TodoStore()
    for kwargs in [{"limit": -1}, {"limit": True}, {"limit": 1.5},
                   {"done": 0}, {"tag": "  "}]:
        with pytest.raises(ValueError):
            select(store, **kwargs)
    assert select(store) == []


def test_query_results_cannot_mutate_store():
    store = TodoStore()
    store.add("one", ["tag"])
    rows = select(store)
    rows[0]["tags"].clear()
    rows[0]["done"] = True
    rows.clear()
    assert store.get(1) == {"id": 1, "title": "one", "done": False, "tags": ["tag"]}
