class BitSet:
    """A bounded, mutable collection of bits."""

    def __init__(self, size: int, value: int = 0):
        if not isinstance(size, int) or size < 0:
            raise ValueError("invalid size")
        if not isinstance(value, int) or value < 0 or value >= (1 << size):
            raise ValueError("value does not fit")
        self._size = size
        self._value = value

    def set(self, index: int, enabled: bool = True) -> None:
        if not 0 <= index < self._size:
            raise IndexError(index)
        mask = 1 << index
        if enabled:
            self._value |= mask
        else:
            self._value &= ~mask

    def get(self, index: int) -> bool:
        if not 0 <= index < self._size:
            raise IndexError(index)
        return bool(self._value & (1 << index))

    def count(self) -> int:
        return self._value.bit_count()

    def combine(self, other: "BitSet", operation: str) -> "BitSet":
        if self._size != other._size:
            raise ValueError("width mismatch")
        if operation == "and":
            value = self._value & other._value
        elif operation == "or":
            value = self._value | other._value
        elif operation == "xor":
            value = self._value ^ other._value
        else:
            raise ValueError("unknown operation")
        return BitSet(self._size, value)

    def to01(self) -> str:
        if self._size == 0:
            return ""
        return format(self._value, f"0{self._size}b")
