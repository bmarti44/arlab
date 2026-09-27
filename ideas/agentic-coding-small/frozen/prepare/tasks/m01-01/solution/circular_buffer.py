class CircularBuffer:
    """A bounded queue whose newest writes replace its oldest items."""

    def __init__(self, capacity):
        """Allocate storage once; head always points to the oldest item."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items = [None] * capacity
        self._head = 0
        self._size = 0

    def __len__(self):
        """Return the occupied size, not the storage capacity."""
        return self._size

    def append(self, item):
        """Insert an item and return the displaced item, if any."""
        if self._size == self._capacity:
            evicted = self._items[self._head]
            self._items[self._head] = item
            self._head = (self._head + 1) % self._capacity
            return evicted
        tail = (self._head + self._size) % self._capacity
        self._items[tail] = item
        self._size += 1
        return None

    def extend(self, items):
        """Collect evictions without confusing an evicted None with no eviction."""
        evicted = []
        for item in items:
            was_full = self._size == self._capacity
            old = self.append(item)
            if was_full:
                evicted.append(old)
        return evicted

    def pop(self):
        """Remove the logical front, releasing its storage slot."""
        if not self._size:
            raise IndexError("empty buffer")
        item = self._items[self._head]
        self._items[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return item

    def snapshot(self):
        """Return an independent view in FIFO order."""
        return [self._items[(self._head + i) % self._capacity]
                for i in range(self._size)]
