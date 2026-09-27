from copy import deepcopy
import pytest
from inventory import calculate_reorders


def test_incoming_supply_avoids_unneeded_order():
    rows = [dict(sku="A", on_hand=2, reserved=1, incoming=9, daily_rate=2, pack_size=5)]
    assert calculate_reorders(rows, 5) == []
    assert calculate_reorders([], 5) == []
    assert calculate_reorders(rows, 0) == []


def test_shortage_smaller_than_one_pack():
    rows = [dict(sku="B", on_hand=9, reserved=0, incoming=0, daily_rate=2, pack_size=6)]
    assert calculate_reorders(rows, 5) == [dict(sku="B", available=9, target=10, order=6)]


def test_fractional_demand_and_reservations():
    rows = [dict(sku="C", on_hand=3, reserved=2, incoming=2, daily_rate=1.25, pack_size=4)]
    assert calculate_reorders(rows, 5) == [dict(sku="C", available=3, target=7, order=4)]


def test_excess_reservations_and_exact_pack():
    rows = [dict(sku="D", on_hand=2, reserved=7, incoming=3, daily_rate=11, pack_size=4)]
    assert calculate_reorders(rows, 1) == [dict(sku="D", available=3, target=11, order=8)]


def test_order_filtering_and_input_isolation():
    rows = [
        dict(sku="Z", on_hand=0, reserved=0, incoming=2, daily_rate=7, pack_size=4, note="keep"),
        dict(sku="A", on_hand=20, reserved=0, incoming=0, daily_rate=1, pack_size=3),
        dict(sku="M", on_hand=0, reserved=0, incoming=1, daily_rate=3, pack_size=1),
    ]
    before = deepcopy(rows)
    result = calculate_reorders(iter(rows), 1)
    assert result == [dict(sku="Z", available=2, target=7, order=8), dict(sku="M", available=1, target=3, order=2)]
    result[0]["sku"] = "changed"
    assert rows == before


def test_validation_remains_in_place():
    row = dict(sku="X", on_hand=0, reserved=0, incoming=1, daily_rate=5, pack_size=3)
    assert calculate_reorders([row], 1) == [dict(sku="X", available=1, target=5, order=6)]
    for horizon in (-1, 1.5):
        with pytest.raises(ValueError):
            calculate_reorders([row], horizon)
    with pytest.raises(ValueError):
        calculate_reorders([row, dict(row)], 1)
    for field, value in [("sku", ""), ("pack_size", 0), ("reserved", -1), ("on_hand", 1.5), ("daily_rate", float("inf"))]:
        with pytest.raises(ValueError):
            calculate_reorders([dict(row, **{field: value})], 1)
    missing = dict(row)
    del missing["incoming"]
    with pytest.raises(ValueError):
        calculate_reorders([missing], 1)
