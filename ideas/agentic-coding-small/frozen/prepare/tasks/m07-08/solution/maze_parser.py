"""Strict parser for a small ASCII maze format."""
from dataclasses import dataclass


@dataclass
class Maze:
    rows: tuple[str, ...]
    start: tuple[int, int]
    end: tuple[int, int]


def parse_maze(text: str) -> Maze:
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        raise ValueError("empty maze")
    rows = tuple(text.split("\n"))
    width = len(rows[0])
    starts = []
    ends = []
    for row_index, row in enumerate(rows):
        if not row or len(row) != width:
            raise ValueError("maze must be rectangular")
        for column_index, cell in enumerate(row):
            coordinate = (row_index, column_index)
            if cell not in "#.SE":
                raise ValueError("invalid maze character")
            if cell == "S":
                starts.append(coordinate)
            elif cell == "E":
                ends.append(coordinate)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("expected exactly one start and end")
    return Maze(rows, starts[0], ends[0])
