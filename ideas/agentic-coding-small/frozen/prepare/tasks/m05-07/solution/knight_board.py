import re


def parse_square(square: str, width: int, height: int) -> tuple[int, int]:
    """Read canonical lowercase algebraic notation within a rectangle."""
    if re.fullmatch(r"[a-z][1-9][0-9]*", square) is None:
        raise ValueError("invalid square notation")
    x = ord(square[0]) - ord("a")
    y = int(square[1:]) - 1
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("square is outside the board")
    return x, y


def format_square(x: int, y: int) -> str:
    """Format validated coordinates without lexicographic rank sorting."""
    return f"{chr(ord('a') + x)}{y + 1}"


class KnightBoard:
    """Calculate single knight jumps on a bounded rectangular board."""

    def __init__(self, width: int = 8, height: int = 8):
        if type(width) is not int or type(height) is not int:
            raise TypeError("dimensions must be integers")
        if not 1 <= width <= 26 or not 1 <= height <= 99:
            raise ValueError("dimensions are outside supported bounds")
        self.width = width
        self.height = height

    def _blocked_coordinates(self, blocked):
        """Consume the iterable once, validate all entries, and deduplicate."""
        return {parse_square(square, self.width, self.height)
                for square in blocked}

    def moves(self, start: str, blocked=()) -> list[str]:
        """Return unoccupied destinations ordered by rank and then file."""
        x, y = parse_square(start, self.width, self.height)
        occupied = self._blocked_coordinates(blocked)
        destinations = []
        offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2), (1, 2), (2, -1), (2, 1)]
        for dx, dy in offsets:
            target = (x + dx, y + dy)
            tx, ty = target
            inside = 0 <= tx < self.width and 0 <= ty < self.height
            if inside and target not in occupied:
                destinations.append(target)
        destinations.sort(key=lambda point: (point[1], point[0]))
        return [format_square(tx, ty) for tx, ty in destinations]

    def render(self, start: str, blocked=()) -> str:
        """Draw a top-down board, marking the knight before blocked cells."""
        origin = parse_square(start, self.width, self.height)
        occupied = self._blocked_coordinates(blocked)
        blocked_names = [format_square(x, y) for x, y in occupied]
        destinations = set(self.moves(start, blocked_names))
        lines = []
        for y in range(self.height - 1, -1, -1):
            cells = []
            for x in range(self.width):
                point = (x, y)
                if point == origin:
                    cell = "K"
                elif point in occupied:
                    cell = "#"
                elif format_square(x, y) in destinations:
                    cell = "*"
                else:
                    cell = "."
                cells.append(cell)
            lines.append(" ".join(cells))
        return "\n".join(lines)
