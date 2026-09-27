import re

def parse_square(square: str, width: int, height: int) -> tuple[int, int]:
    """Read canonical lowercase algebraic notation within a rectangle."""
    raise NotImplementedError()

def format_square(x: int, y: int) -> str:
    """Format validated coordinates without lexicographic rank sorting."""
    raise NotImplementedError()

class KnightBoard:
    """Calculate single knight jumps on a bounded rectangular board."""

    def __init__(self, width: int=8, height: int=8):
        raise NotImplementedError()

    def _blocked_coordinates(self, blocked):
        """Consume the iterable once, validate all entries, and deduplicate."""
        raise NotImplementedError()

    def moves(self, start: str, blocked=()) -> list[str]:
        """Return unoccupied destinations ordered by rank and then file."""
        raise NotImplementedError()

    def render(self, start: str, blocked=()) -> str:
        """Draw a top-down board, marking the knight before blocked cells."""
        raise NotImplementedError()
