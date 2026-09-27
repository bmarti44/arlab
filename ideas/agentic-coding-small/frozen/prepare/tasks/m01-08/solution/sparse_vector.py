class SparseVector:
    """A fixed-dimensional vector storing only nonzero coordinates."""

    def __init__(self, dimension, entries=None):
        if dimension < 0:
            raise ValueError("negative dimension")
        self.dimension = dimension
        self._entries = {}
        if entries is None:
            entries = {}
        for index, value in entries.items():
            if index < 0 or index >= dimension:
                raise IndexError(index)
            numeric = float(value)
            if numeric != 0:
                self._entries[index] = numeric

    def __getitem__(self, index):
        """Check bounds even when the coordinate is not stored."""
        if index < 0 or index >= self.dimension:
            raise IndexError(index)
        return self._entries.get(index, 0.0)

    def to_dict(self):
        """Return a detached representation of the nonzero coordinates."""
        return dict(self._entries)

    def add(self, other):
        """Merge sparse maps, dropping coordinates that cancel exactly."""
        if self.dimension != other.dimension:
            raise ValueError("dimension mismatch")
        result = dict(self._entries)
        for index, value in other._entries.items():
            combined = result.get(index, 0.0) + value
            if combined == 0:
                result.pop(index, None)
            else:
                result[index] = combined
        return SparseVector(self.dimension, result)

    def scale(self, factor):
        """Construct a new vector so zero filtering remains centralized."""
        result = {}
        for index, value in self._entries.items():
            scaled = value * factor
            if scaled != 0:
                result[index] = scaled
        return SparseVector(self.dimension, result)
