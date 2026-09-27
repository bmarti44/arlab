import pytest
from knight_board import KnightBoard, parse_square, format_square


def test_coordinate_helpers():
    assert parse_square("z99", 26, 99) == (25, 98)
    assert format_square(25, 98) == "z99"
    assert parse_square("a1", 1, 1) == (0, 0)


def test_corner_and_center_moves():
    board = KnightBoard()
    assert board.moves("a1") == ["c2", "b3"]
    assert board.moves("d4") == ["c2", "e2", "b3", "f3", "b5", "f5", "c6", "e6"]
    assert board.moves("h8") == ["g6", "f7"]


def test_blocked_generator_and_duplicates():
    board = KnightBoard()
    assert board.moves("a1", (s for s in ["c2", "c2", "a1", "a2"])) == ["b3"]
    assert board.moves("a1", ["b3", "c2"]) == []


def test_board_render_orientation_and_priority():
    board = KnightBoard(4, 4)
    assert board.render("a1", (s for s in ["a1", "c2", "d4"])) == ". . . #\n. * . .\n. . # .\nK . . ."
    assert KnightBoard(1, 1).render("a1", ["a1"]) == "K"


def test_narrow_and_tall_board():
    assert KnightBoard(1, 12).moves("a8") == []
    assert KnightBoard(3, 12).moves("b9") == ["a7", "c7", "a11", "c11"]


def test_invalid_squares_and_blocked_entries():
    board = KnightBoard()
    for square in ["A1", "a01", "a0", "a9", "i1", "aa1", "a١", " a1", "a1\n"]:
        with pytest.raises(ValueError):
            parse_square(square, 8, 8)
        with pytest.raises(ValueError):
            board.moves("a1", [square])
        with pytest.raises(ValueError):
            board.render(square)


def test_dimensions():
    for width, height in [(0, 8), (27, 8), (8, 0), (8, 100)]:
        with pytest.raises(ValueError):
            KnightBoard(width, height)
    for width, height in [(True, 8), (8, False), (3.0, 4)]:
        with pytest.raises(TypeError):
            KnightBoard(width, height)
