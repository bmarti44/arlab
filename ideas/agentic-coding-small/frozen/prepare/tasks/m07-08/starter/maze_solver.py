from maze_parser import Maze, parse_maze


def shortest_path(maze: Maze) -> list[tuple[int, int]] | None:
    raise NotImplementedError


def solve_maze(text: str) -> list[tuple[int, int]] | None:
    raise NotImplementedError


def render_path(maze: Maze, path: list[tuple[int, int]]) -> str:
    raise NotImplementedError
