class BoundedHistogram:
    """Inclusive integer bins with validated, atomic updates."""

    def __init__(self, low: int, high: int, width: int):
        if any(type(x) is not int for x in (low, high, width)):
            raise TypeError("bounds and width must be integers")
        if low > high or width <= 0:
            raise ValueError("invalid bin configuration")
        self._low = low
        self._high = high
        self._width = width
        size = (high - low) // width + 1
        self._counts = [0] * size

    def add(self, value: int, count: int = 1) -> None:
        if type(value) is not int or type(count) is not int:
            raise TypeError("value and count must be integers")
        if not self._low <= value <= self._high:
            raise ValueError("value outside bounds")
        if count < 0:
            raise ValueError("negative count")
        index = (value - self._low) // self._width
        self._counts[index] += count

    def count(self, value: int) -> int:
        if type(value) is not int:
            raise TypeError("value must be an integer")
        if not self._low <= value <= self._high:
            raise ValueError("value outside bounds")
        return self._counts[(value - self._low) // self._width]

    def bins(self) -> list[tuple[int, int, int]]:
        result = []
        for index, amount in enumerate(self._counts):
            start = self._low + index * self._width
            end = min(start + self._width - 1, self._high)
            result.append((start, end, amount))
        return result

    def reset(self) -> None:
        for index in range(len(self._counts)):
            self._counts[index] = 0
