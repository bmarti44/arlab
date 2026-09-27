class CircularBuffer:
    """A bounded queue whose newest writes replace its oldest items."""

    def __init__(self, capacity):
        """Allocate storage once; head always points to the oldest item."""
        raise NotImplementedError()

    def __len__(self):
        """Return the occupied size, not the storage capacity."""
        raise NotImplementedError()

    def append(self, item):
        """Insert an item and return the displaced item, if any."""
        raise NotImplementedError()

    def extend(self, items):
        """Collect evictions without confusing an evicted None with no eviction."""
        raise NotImplementedError()

    def pop(self):
        """Remove the logical front, releasing its storage slot."""
        raise NotImplementedError()

    def snapshot(self):
        """Return an independent view in FIFO order."""
        raise NotImplementedError()
