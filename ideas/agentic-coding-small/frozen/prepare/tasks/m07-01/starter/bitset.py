class BitSet:
    def __init__(self, size: int, value: int = 0):
        raise NotImplementedError

    def set(self, index: int, enabled: bool = True) -> None:
        raise NotImplementedError

    def get(self, index: int) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def combine(self, other: "BitSet", operation: str) -> "BitSet":
        raise NotImplementedError

    def to01(self) -> str:
        raise NotImplementedError
