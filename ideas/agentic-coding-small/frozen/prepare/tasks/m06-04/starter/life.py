def neighbor_cells(x, y, width, height, wrap=False):
    """Return physical neighbor cells, counting each at most once."""
    raise NotImplementedError()

class LifeGrid:
    """A snapshot of a finite Game of Life board."""

    def __init__(self, width, height, alive=(), wrap=False):
        raise NotImplementedError()

    def step(self):
        """Compute a generation using only the current snapshot.

        Cells not adjacent to any live cell cannot be born, so only
        the live cells and their neighbors need to be considered.
        """
        raise NotImplementedError()

    def advance(self, steps):
        """Return an independent snapshot after zero or more steps.

        Construct the initial snapshot even when no evolution is requested.
        """
        raise NotImplementedError()
