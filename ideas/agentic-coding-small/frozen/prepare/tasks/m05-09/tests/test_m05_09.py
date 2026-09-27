import pytest
from snapshot_store import SnapshotStore, validate_path
from version_history import History


def test_snapshot_copy_and_id_allocation():
    store = SnapshotStore()
    source = {"docs/a.txt": "one"}
    assert store.save(source) == 1
    source["docs/a.txt"] = "changed"
    loaded = store.load(1)
    loaded.clear()
    assert store.load(1) == {"docs/a.txt": "one"}
    assert store.save({"docs/a.txt": "one"}) == 2


def test_path_and_store_validation_are_atomic():
    store = SnapshotStore()
    for path in ["", "/a", "a/", "a//b", "./a", "a/../b", "a\\b"]:
        with pytest.raises(ValueError):
            validate_path(path)
        with pytest.raises(ValueError):
            store.save({path: "bad"})
    with pytest.raises(TypeError):
        store.save({"ok": "yes", "bad": 3})
    assert validate_path("a folder/file.txt") is None
    assert store.save({}) == 1
    for commit_id in [0, 9, True, "1"]:
        with pytest.raises(KeyError):
            store.load(commit_id)


def test_working_edits_and_deletion():
    history = History()
    assert history.head is None and history.files() == {} and history.log() == []
    history.write("a", "one")
    history.write("a", "two")
    history.files().clear()
    assert history.files() == {"a": "two"}
    assert history.delete("absent") is False
    assert history.delete("a") is True
    assert history.files() == {}
    with pytest.raises(ValueError):
        history.write("../a", "x")
    with pytest.raises(ValueError):
        history.delete("/a")
    with pytest.raises(TypeError):
        history.write("a", 7)
    assert history.files() == {}


def test_commit_checkout_and_trimmed_messages():
    history = History()
    history.write("a", "one")
    assert history.commit("  first ") == 1
    history.write("a", "two")
    history.write("b", "new")
    assert history.commit("second") == 2
    history.write("uncommitted", "discard")
    history.checkout(1)
    assert history.head == 1 and history.files() == {"a": "one"}
    assert history.log() == [{"id": 1, "message": "first", "parent": None}]
    history.checkout(2)
    assert history.files() == {"a": "two", "b": "new"}


def test_branch_follows_parent_and_retains_old_commits():
    history = History()
    history.commit("root")
    history.write("old", "branch")
    history.commit("old branch")
    history.checkout(1)
    history.write("new", "branch")
    assert history.commit("new branch") == 3
    assert history.log() == [{"id": 3, "message": "new branch", "parent": 1}, {"id": 1, "message": "root", "parent": None}]
    history.checkout(2)
    assert history.files() == {"old": "branch"}
    assert [item["id"] for item in history.log()] == [2, 1]


def test_invalid_operations_preserve_head_files_and_ids():
    history = History()
    history.write("a", "kept")
    assert history.commit("valid") == 1
    history.write("b", "uncommitted")
    with pytest.raises(ValueError):
        history.commit(" \n\t")
    for commit_id in [0, 20, True, "1"]:
        with pytest.raises(KeyError):
            history.checkout(commit_id)
    assert history.head == 1 and history.files() == {"a": "kept", "b": "uncommitted"}
    assert history.commit("next") == 2


def test_log_limits_and_detached_records():
    history = History()
    history.commit("a")
    history.commit("b")
    assert history.log(0) == []
    assert len(history.log(1)) == 1
    assert len(history.log(50)) == 2
    with pytest.raises(ValueError):
        history.log(-1)
    result = history.log()
    result[0]["parent"] = None
    result[0]["message"] = "changed"
    result.clear()
    assert history.log() == [{"id": 2, "message": "b", "parent": 1}, {"id": 1, "message": "a", "parent": None}]


def test_empty_and_unchanged_snapshots_are_commits():
    history = History()
    assert history.commit("empty") == 1
    assert history.commit("same") == 2
    history.write("a", "")
    history.commit("blank file")
    history.delete("a")
    assert history.commit("deleted") == 4
    history.checkout(3)
    assert history.files() == {"a": ""}
    history.checkout(4)
    assert history.files() == {}
