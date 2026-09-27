from collections import OrderedDict
import math
from cache_entry import Entry

class LRUCache:
    def __init__(self, capacity, clock):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('invalid capacity')
        self.capacity = capacity
        self.clock = clock
        self._items = OrderedDict()

    def _purge(self, now):
        for key, entry in list(self._items.items()):
            if entry.expired(now):
                del self._items[key]

    def put(self, key, value, ttl=None):
        if ttl is not None and (type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl < 0):
            raise ValueError('invalid ttl')
        now = self.clock()
        self._purge(now)
        if ttl == 0:
            self._items.pop(key, None)
            return
        self._items[key] = Entry(value, None if ttl is None else now + ttl)
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, key, default=None):
        self._purge(self.clock())
        if key not in self._items:
            return default
        self._items.move_to_end(key)
        return self._items[key].value

    def delete(self, key):
        self._purge(self.clock())
        if key not in self._items:
            return False
        del self._items[key]
        return True

    def keys(self):
        self._purge(self.clock())
        return list(self._items)

    def __len__(self):
        self._purge(self.clock())
        return len(self._items)
