import pytest
from bitset import BitSet


def test_initial_width_and_bit_order():
    bits = BitSet(8, 5)
    assert bits.to01() == "00000101"
    assert bits.get(0) is True
    assert bits.get(1) is False
    assert bits.get(2) is True
    assert bits.count() == 2


def test_mutation_and_idempotence():
    bits = BitSet(5)
    assert bits.set(4) is None
    bits.set(4)
    bits.set(1)
    assert bits.to01() == "10010"
    assert bits.count() == 2
    assert bits.set(4, False) is None
    bits.set(4, False)
    assert bits.to01() == "00010"


def test_bounds_preserve_state():
    bits = BitSet(3, 6)
    for index in (-1, 3, 100):
        with pytest.raises(IndexError):
            bits.set(index)
        with pytest.raises(IndexError):
            bits.get(index)
    assert bits.to01() == "110"


def test_combinations_are_independent():
    left, right = BitSet(4, 10), BitSet(4, 12)
    assert left.combine(right, "and").to01() == "1000"
    assert left.combine(right, "or").to01() == "1110"
    result = left.combine(right, "xor")
    assert result.to01() == "0110"
    result.set(0)
    assert left.to01() == "1010"
    assert right.to01() == "1100"
    with pytest.raises(ValueError):
        left.combine(BitSet(3), "or")
    with pytest.raises(ValueError):
        left.combine(right, "AND")


def test_empty_and_invalid_construction():
    empty = BitSet(0)
    assert empty.to01() == ""
    assert empty.count() == 0
    assert empty.combine(BitSet(0), "xor").to01() == ""
    with pytest.raises(IndexError):
        empty.get(0)
    for size, value in [(-1, 0), (2.5, 0), (3, -1), (3, 8), (0, 1), (2, 1.5)]:
        with pytest.raises(ValueError):
            BitSet(size, value)


def test_large_width():
    bits = BitSet(130, (1 << 129) | 1)
    assert bits.count() == 2
    assert bits.to01() == "1" + "0" * 128 + "1"
    bits.set(64)
    assert bits.get(64) is True
    bits.set(129, False)
    assert bits.count() == 2
