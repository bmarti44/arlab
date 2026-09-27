def path_counts(rows, cols, blocked):
    """Count suffix routes by visiting cells in reverse movement order."""
    if rows <= 0 or cols <= 0:
        raise ValueError('dimensions must be positive')
    walls = set(blocked)
    for row, col in walls:
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError('blocked cell outside grid')
    counts = [[0] * cols for _ in range(rows)]
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if (row, col) in walls:
                continue
            if row == rows - 1 and col == cols - 1:
                counts[row][col] = 1
                continue
            right = counts[row][col + 1] if col + 1 < cols else 0
            down = counts[row + 1][col] if row + 1 < rows else 0
            counts[row][col] = right + down
    return counts


class GridPaths:
    """A fixed rectangular grid with cached suffix route counts."""

    def __init__(self, rows, cols, blocked=()):
        walls = set(blocked)
        counts = path_counts(rows, cols, walls)
        self._rows = rows
        self._cols = cols
        self._blocked = frozenset(walls)
        self._counts = counts

    def count(self):
        return self._counts[0][0]

    def path(self):
        if self.count() == 0:
            return None
        row, col = 0, 0
        result = [(row, col)]
        while row != self._rows - 1 or col != self._cols - 1:
            can_go_right = (
                col + 1 < self._cols
                and self._counts[row][col + 1] > 0
            )
            if can_go_right:
                col += 1
            else:
                row += 1
            result.append((row, col))
        return result

    def render(self):
        route = set(self.path() or [])
        lines = []
        destination = (self._rows - 1, self._cols - 1)
        for row in range(self._rows):
            cells = []
            for col in range(self._cols):
                cell = (row, col)
                if cell in self._blocked:
                    symbol = '#'
                elif cell == (0, 0):
                    symbol = 'S'
                elif cell == destination:
                    symbol = 'E'
                elif cell in route:
                    symbol = '*'
                else:
                    symbol = '.'
                cells.append(symbol)
            lines.append(''.join(cells))
        return '\n'.join(lines)
