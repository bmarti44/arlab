class SparseVector:
    """A fixed-dimensional vector storing only nonzero coordinates."""

    def __init__(self, dimension, entries=None):
        raise NotImplementedError()

    def __getitem__(self, index):
        """Check bounds even when the coordinate is not stored."""
        raise NotImplementedError()

    def to_dict(self):
        """Return a detached representation of the nonzero coordinates."""
        raise NotImplementedError()

    def add(self, other):
        """Merge sparse maps, dropping coordinates that cancel exactly."""
        raise NotImplementedError()

    def scale(self, factor):
        """Construct a new vector so zero filtering remains centralized."""
        raise NotImplementedError()
