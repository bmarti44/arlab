import pytest
from parking import ParkingLot


def test_best_fit_before_spot_id():
    lot = ParkingLot({"A": "large", "Z": "compact", "B": "standard"})
    assert lot.park("small", "compact") == "Z"
    assert lot.park("medium", "standard") == "B"
    assert lot.park("big", "large") == "A"
    assert lot.locate("small") == "Z"


def test_ties_and_upgrades():
    lot = ParkingLot({"C2": "compact", "L": "large", "C1": "compact", "S": "standard"})
    assert lot.park("one", "compact") == "C1"
    assert lot.park("two", "compact") == "C2"
    assert lot.park("three", "compact") == "S"
    assert lot.park("four", "standard") == "L"
    assert lot.park("five", "compact") is None
    assert lot.locate("five") is None


def test_incompatible_space_and_retry():
    lot = ParkingLot({"C": "compact", "L": "large"})
    assert lot.park("bus", "large") == "L"
    assert lot.park("van", "standard") is None
    assert lot.locate("van") is None
    assert lot.leave("bus") == "L"
    assert lot.park("van", "standard") == "L"
    assert lot.locate("bus") is None


def test_duplicate_and_invalid_park_are_atomic():
    lot = ParkingLot({"A": "compact"})
    assert lot.park("p", "compact") == "A"
    before = lot.snapshot()
    for plate, kind in [("p", "compact"), ("p", "large"), ("", "compact"), ("x", "tiny")]:
        with pytest.raises(ValueError):
            lot.park(plate, kind)
        assert lot.snapshot() == before
    with pytest.raises(KeyError):
        lot.leave("missing")
    assert lot.snapshot() == before


def test_snapshot_order_and_isolation():
    spots = {"b": "large", "a": "compact"}
    lot = ParkingLot(spots)
    spots.clear()
    before = lot.snapshot()
    assert before == [("a", "compact", None), ("b", "large", None)]
    lot.park("P", "compact")
    assert before == [("a", "compact", None), ("b", "large", None)]
    before.clear()
    assert lot.snapshot() == [("a", "compact", "P"), ("b", "large", None)]


def test_empty_lot_and_constructor_validation():
    lot = ParkingLot({})
    assert lot.snapshot() == []
    assert lot.park("x", "large") is None
    assert lot.locate("x") is None
    with pytest.raises(KeyError):
        lot.leave("x")
    for spots in ({"": "compact"}, {"x": "COMPACT"}):
        with pytest.raises(ValueError):
            ParkingLot(spots)


def test_case_sensitive_plates_and_reuse():
    lot = ParkingLot({" ": "standard", "x": "standard"})
    assert lot.park("P", "standard") == " "
    assert lot.park("p", "standard") == "x"
    assert lot.leave("P") == " "
    assert lot.park(" ", "compact") == " "
    assert lot.locate("p") == "x"
