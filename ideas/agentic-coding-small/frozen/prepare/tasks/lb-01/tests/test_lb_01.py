import pytest
from text_buffer import TextBuffer


def test_insert_positions_and_unicode():
    b = TextBuffer("é")
    assert b.insert(0, "A") is None
    b.insert(2, "猫")
    b.insert(1, "!")
    assert b.text == "A!é猫"


def test_delete_returns_slice():
    b = TextBuffer("abcde")
    assert b.delete(1, 4) == "bcd"
    assert b.text == "ae"
    assert b.delete(0, 2) == "ae"
    assert b.text == ""


def test_round_trip_history():
    b = TextBuffer("a")
    assert b.undo() is False
    b.insert(1, "bc")
    b.delete(0, 1)
    assert b.undo() is True
    assert b.text == "abc"
    assert b.undo() is True
    assert b.text == "a"
    assert b.undo() is False
    assert b.redo() is True
    assert b.text == "abc"
    assert b.redo() is True
    assert b.text == "bc"
    assert b.redo() is False


def test_edit_after_undo_clears_redo():
    b = TextBuffer()
    b.insert(0, "old")
    b.undo()
    b.insert(0, "new")
    assert b.redo() is False
    assert b.text == "new"
    b.undo()
    assert b.text == ""


def test_noops_preserve_redo_and_undo():
    b = TextBuffer("x")
    b.insert(1, "y")
    b.undo()
    assert b.insert(0, "") is None
    assert b.delete(1, 1) == ""
    assert b.redo() is True
    assert b.text == "xy"
    assert b.undo() is True
    assert b.undo() is False


def test_invalid_edits_are_atomic():
    b = TextBuffer("ab")
    b.insert(2, "c")
    b.undo()
    for pos in (-1, 3):
        with pytest.raises(IndexError):
            b.insert(pos, "")
    for start, end in [(-1, 1), (2, 1), (0, 3)]:
        with pytest.raises(IndexError):
            b.delete(start, end)
    assert b.text == "ab"
    assert b.redo() is True
    assert b.text == "abc"
