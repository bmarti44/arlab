class IntervalSet:
    """A normalized union of half-open integer intervals."""

    def __init__(self):
        raise NotImplementedError()

    def add(self, start: int, end: int) -> None:
        raise NotImplementedError()

    def remove(self, start: int, end: int) -> None:
        raise NotImplementedError()

    def contains(self, value: int) -> bool:
        raise NotImplementedError()

    def intervals(self) -> list[tuple[int, int]]:
        raise NotImplementedError()
