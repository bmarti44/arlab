from dataclasses import dataclass


@dataclass
class Maze:
    rows: tuple[str, ...]
    start: tuple[int, int]
    end: tuple[int, int]


def parse_maze(text: str) -> Maze:
    raise NotImplementedError
