import pytest
from packing import BinPacker, first_fit, validate_weights


def test_first_fit_preserves_order():
    assert first_fit([6, 7, 3, 4], 10) == [[6, 3], [7], [4]]
    assert first_fit([8, 2, 10], 10) == [[8, 2], [10]]


def test_incremental_assignments_and_capacity():
    packer = BinPacker(10)
    assert packer.add_many([6, 7, 3]) == [0, 1, 0]
    assert packer.add(2) == 1
    assert packer.remaining() == [1, 1]
    assert packer.add(1) == 0
    assert packer.snapshot() == [[6, 3, 1], [7, 2]]


def test_generator_and_empty_input():
    assert first_fit((x for x in [2, 4, 2]), 6) == [[2, 4], [2]]
    packer = BinPacker(1)
    assert packer.add_many(iter([])) == []
    assert packer.snapshot() == packer.remaining() == []
    assert first_fit([], 7) == []


def test_capacity_validation():
    for capacity in [0, -1]:
        with pytest.raises(ValueError):
            BinPacker(capacity)
        with pytest.raises(ValueError):
            first_fit([], capacity)
    for capacity in [True, 3.0, "3"]:
        with pytest.raises(TypeError):
            validate_weights([], capacity)


def test_weight_validation_and_single_add_is_atomic():
    packer = BinPacker(5)
    packer.add(2)
    for value, error in [(0, ValueError), (-2, ValueError), (6, ValueError), (True, TypeError), (1.0, TypeError)]:
        with pytest.raises(error):
            packer.add(value)
    assert packer.snapshot() == [[2]]


def test_bad_batch_is_atomic():
    packer = BinPacker(5)
    packer.add(2)
    with pytest.raises(ValueError):
        packer.add_many(x for x in [3, 2, 6])
    assert packer.snapshot() == [[2]] and packer.remaining() == [3]
    with pytest.raises(TypeError):
        packer.add_many([1, "2"])
    assert packer.snapshot() == [[2]]


def test_returned_lists_do_not_alias():
    source = [2, 3]
    checked = validate_weights(source, 5)
    checked.append(1)
    assert source == [2, 3]
    packer = BinPacker(5)
    packer.add_many(source)
    view = packer.snapshot()
    view[0].clear()
    view.append([5])
    free = packer.remaining()
    free[0] = 100
    assert packer.snapshot() == [[2, 3]] and packer.remaining() == [0]
