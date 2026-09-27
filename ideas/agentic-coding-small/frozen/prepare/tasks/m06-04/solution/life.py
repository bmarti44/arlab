def neighbor_cells(x, y, width, height, wrap=False):
    """Return physical neighbor cells, counting each at most once."""
    neighbors = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if wrap:
                nx %= width
                ny %= height
            elif not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) != (x, y):
                neighbors.add((nx, ny))
    return neighbors


class LifeGrid:
    """A snapshot of a finite Game of Life board."""

    def __init__(self, width, height, alive=(), wrap=False):
        if width <= 0 or height <= 0:
            raise ValueError('dimensions must be positive')
        cells = frozenset((x, y) for x, y in alive)
        for x, y in cells:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError('live cell outside grid')
        self.width = width
        self.height = height
        self.wrap = wrap
        self.alive = cells

    def step(self):
        """Compute a generation using only the current snapshot.

        Cells not adjacent to any live cell cannot be born, so only
        the live cells and their neighbors need to be considered.
        """
        candidates = set(self.alive)
        for x, y in self.alive:
            candidates.update(neighbor_cells(
                x, y, self.width, self.height, self.wrap
            ))

        next_alive = set()
        for x, y in candidates:
            neighbors = neighbor_cells(
                x, y, self.width, self.height, self.wrap
            )
            count = len(neighbors & self.alive)
            survives = (x, y) in self.alive and count == 2
            if count == 3 or survives:
                next_alive.add((x, y))
        return LifeGrid(
            self.width, self.height, next_alive, self.wrap
        )

    def advance(self, steps):
        """Return an independent snapshot after zero or more steps.

        Construct the initial snapshot even when no evolution is requested.
        """
        if steps < 0:
            raise ValueError('steps must be nonnegative')
        current = LifeGrid(
            self.width, self.height, self.alive, self.wrap
        )
        for _ in range(steps):
            current = current.step()
        return current
