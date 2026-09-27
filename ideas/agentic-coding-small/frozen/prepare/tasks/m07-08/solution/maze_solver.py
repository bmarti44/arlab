"""Iterative shortest paths and a separate rendering step."""
from collections import deque
from maze_parser import Maze, parse_maze


def shortest_path(maze: Maze) -> list[tuple[int, int]] | None:
    height = len(maze.rows)
    width = len(maze.rows[0])
    queue = deque([maze.start])
    parents = {maze.start: None}
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while queue:
        current = queue.popleft()
        if current == maze.end:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()
            return path
        row, column = current
        for row_delta, column_delta in directions:
            next_row = row + row_delta
            next_column = column + column_delta
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            neighbor = (next_row, next_column)
            if maze.rows[next_row][next_column] == "#":
                continue
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    return None


def solve_maze(text: str) -> list[tuple[int, int]] | None:
    maze = parse_maze(text)
    return shortest_path(maze)


def render_path(maze: Maze, path: list[tuple[int, int]]) -> str:
    cells = set(path)
    output = []
    for row_index, row in enumerate(maze.rows):
        rendered = []
        for column_index, cell in enumerate(row):
            if cell == "." and (row_index, column_index) in cells:
                rendered.append("*")
            else:
                rendered.append(cell)
        output.append("".join(rendered))
    return "\n".join(output)
