from collections import deque
from math import isfinite


class MovingAverage:
    """A fixed-capacity streaming average with a batch convenience method."""

    def __init__(self, window: int):
        if type(window) is not int or window <= 0:
            raise ValueError("window must be a positive integer")
        self.window = window
        self._values = deque()
        self._total = 0.0

    @property
    def count(self) -> int:
        """Number of retained samples, at most the window size."""
        return len(self._values)

    @property
    def average(self) -> float | None:
        """Return None until at least one sample has arrived."""
        if not self._values:
            return None
        return self._total / self.window

    def add(self, value: int | float) -> float:
        """Validate before mutation and return the updated average."""
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("sample must be a finite number")
        if len(self._values) == self.window:
            oldest = self._values.popleft()
            self._total += oldest
        self._values.append(float(value))
        self._total += value
        return self.average

    def extend(self, values) -> list[float]:
        """Consume once; valid samples before an error remain committed."""
        result = []
        for value in values:
            result.append(self.add(value))
        return result

    def reset(self) -> None:
        """Forget all samples while preserving capacity."""
        self._values.clear()
        self._total = 0.0

    def snapshot(self) -> tuple[float, ...]:
        """Return retained samples in arrival order."""
        return tuple(self._values)
