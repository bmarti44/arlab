import pytest
from maze_parser import Maze, parse_maze
from maze_solver import shortest_path, solve_maze, render_path


def test_parse_coordinates_and_optional_newline():
    maze = parse_maze("#.E\nS..\n")
    assert isinstance(maze, Maze)
    assert maze.rows == ("#.E", "S..")
    assert maze.start == (1, 0)
    assert maze.end == (0, 2)
    assert parse_maze("#.E\nS..").rows == maze.rows


def test_invalid_grids():
    for text in ("", "\n", "SE\n\n", "S.\n\n.E", "S..\n.E", "S E", "SE\r\n", "S.", ".E", "SS E", "SEE", "SS.E", "S?E"):
        with pytest.raises(ValueError):
            parse_maze(text)
        with pytest.raises(ValueError):
            solve_maze(text)


def test_adjacent_and_single_column():
    assert solve_maze("SE") == [(0, 0), (0, 1)]
    assert solve_maze("E\n.\nS\n") == [(2, 0), (1, 0), (0, 0)]


def test_tie_breaking_up_then_right():
    text = "...\n.S.\n..E"
    assert solve_maze(text) == [(1, 1), (1, 2), (2, 2)]
    assert solve_maze("E..\n.S.\n...") == [(1, 1), (0, 1), (0, 0)]


def test_detour_and_rendering_are_pure():
    maze = parse_maze("S#E\n.#.\n...")
    before = maze.rows
    path = shortest_path(maze)
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    assert path == expected
    assert render_path(maze, path) == "S#E\n*#*\n***"
    assert maze.rows == before
    assert path == expected
    assert render_path(maze, []) == "S#E\n.#.\n..."


def test_unreachable_without_wraparound():
    maze = parse_maze("S#E\n###\n...")
    assert shortest_path(maze) is None
    assert solve_maze("S#E") is None


def test_open_grid_has_shortest_deterministic_path():
    size = 60
    rows = ["S" + "." * (size - 1)] + ["." * size for _ in range(size - 2)] + ["." * (size - 1) + "E"]
    path = solve_maze("\n".join(rows))
    expected = [(0, column) for column in range(size)] + [(row, size - 1) for row in range(1, size)]
    assert path == expected


def test_render_preserves_unvisited_open_cells():
    maze = parse_maze("S.E\n...")
    assert render_path(maze, shortest_path(maze)) == "S*E\n..."
    assert shortest_path(maze) == [(0, 0), (0, 1), (0, 2)]
