"""Small deterministic caches with configurable replacement policy."""


class Cache:
    """Entries contain value, frequency, and a logical access counter."""

    def __init__(self, capacity: int, policy: str):
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError("invalid capacity")
        if policy not in ("lru", "lfu"):
            raise ValueError("invalid policy")
        self._capacity = capacity
        self._policy = policy
        self._entries = {}
        self._tick = 0

    def get(self, key: str, default=None):
        if key not in self._entries:
            return default
        self._tick += 1
        entry = self._entries[key]
        entry[1] += 1
        entry[2] = self._tick
        return entry[0]

    def put(self, key: str, value) -> None:
        if self._capacity == 0:
            return
        self._tick += 1
        if key in self._entries:
            entry = self._entries[key]
            entry[0] = value
            entry[1] += 1
            entry[2] = self._tick
            return

        if len(self._entries) == self._capacity:
            if self._policy == "lru":
                victim = min(
                    self._entries,
                    key=lambda name: self._entries[name][2],
                )
            else:
                victim = min(
                    self._entries,
                    key=lambda name: (
                        self._entries[name][1],
                        self._entries[name][2],
                    ),
                )
            del self._entries[victim]
        self._entries[key] = [value, 1, self._tick]

    def delete(self, key: str) -> bool:
        if key not in self._entries:
            return False
        del self._entries[key]
        return True

    def snapshot(self) -> list[tuple[str, object]]:
        ordered = sorted(
            self._entries,
            key=lambda name: self._entries[name][2],
        )
        return [(key, self._entries[key][0]) for key in ordered]


def lru_cache(capacity: int) -> Cache:
    """Construct an independent least-recently-used cache."""
    return Cache(capacity, "lru")


def lfu_cache(capacity: int) -> Cache:
    """Construct an independent frequency-first cache."""
    return Cache(capacity, "lfu")
