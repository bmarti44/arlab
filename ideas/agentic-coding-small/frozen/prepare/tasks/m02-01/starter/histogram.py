class BoundedHistogram:
    def __init__(self, low: int, high: int, width: int):
        raise NotImplementedError

    def add(self, value: int, count: int = 1) -> None:
        raise NotImplementedError

    def count(self, value: int) -> int:
        raise NotImplementedError

    def bins(self) -> list[tuple[int, int, int]]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
