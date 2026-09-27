class IntervalSet:
    """A normalized union of half-open integer intervals."""

    def __init__(self):
        self._ranges: list[tuple[int, int]] = []

    def add(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError("reversed interval")
        if start == end:
            return
        merged = []
        placed = False
        for left, right in self._ranges:
            if right < start:
                merged.append((left, right))
            elif end < left:
                if not placed:
                    merged.append((start, end))
                    placed = True
                merged.append((left, right))
            else:
                start = min(start, left)
                end = max(end, right)
        if not placed:
            merged.append((start, end))
        self._ranges = merged

    def remove(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError("reversed interval")
        if start == end:
            return
        remaining = []
        for left, right in self._ranges:
            if right <= start or left >= end:
                remaining.append((left, right))
                continue
            if left < start:
                remaining.append((left, start))
            if right > end:
                remaining.append((end, right))
        self._ranges = remaining

    def contains(self, value: int) -> bool:
        return any(left <= value < right for left, right in self._ranges)

    def intervals(self) -> list[tuple[int, int]]:
        return list(self._ranges)
