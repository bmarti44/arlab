from dataclasses import dataclass


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


def parse_patch(patch):
    raise NotImplementedError
