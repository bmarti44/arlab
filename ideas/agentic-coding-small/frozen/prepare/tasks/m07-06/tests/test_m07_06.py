import pytest
from itinerary import Itinerary, parse_time


def test_time_parser_boundaries():
    assert parse_time("0@00:00") == 0
    assert parse_time("01@23:59") == 2879
    assert parse_time("12@04:05") == 17525
    for bad in ("1@24:00", "0@12:60", "-1@00:00", "0@1:00", "0@01:0", " 0@00:00", "0@00:00\n", "０@00:00", "00:00"):
        with pytest.raises(ValueError):
            parse_time(bad)


def test_empty_and_single_leg():
    trip = Itinerary()
    assert trip.validate() == []
    assert trip.elapsed() == 0
    assert trip.legs() == []
    assert trip.add("A", "B", "0@23:00", "1@01:00") is None
    assert trip.validate() == []
    assert trip.elapsed() == 120
    assert trip.legs() == [("A", "B", 1380, 1500)]


def test_connection_threshold_and_zero_minimum():
    trip = Itinerary()
    trip.add("A", "B", "0@08:00", "0@09:00")
    trip.add("B", "C", "0@09:30", "0@10:00")
    trip.add("C", "D", "0@10:29", "0@11:00")
    assert trip.validate() == [(2, "connection")]
    assert trip.elapsed() == 180
    immediate = Itinerary(0)
    immediate.add("A", "B", "0@00:00", "0@01:00")
    immediate.add("B", "C", "0@01:00", "0@02:00")
    assert immediate.validate() == []


def test_issue_order_and_overlap_precedence():
    trip = Itinerary(90)
    trip.add("A", "B", "0@08:00", "0@10:00")
    trip.add("X", "C", "0@09:00", "0@11:00")
    trip.add("Y", "D", "0@11:10", "0@12:00")
    assert trip.validate() == [(1, "location"), (1, "overlap"), (2, "location"), (2, "connection")]


def test_invalid_add_is_atomic():
    trip = Itinerary()
    trip.add("A", "B", "0@08:00", "0@09:00")
    before = trip.legs()
    for leg in [("", "B", "0@10:00", "0@11:00"), ("A", "", "0@10:00", "0@11:00"), ("A", "B", "bad", "0@11:00"), ("A", "B", "0@10:00", "bad"), ("A", "B", "0@10:00", "0@10:00"), ("A", "B", "1@10:00", "0@11:00")]:
        with pytest.raises(ValueError):
            trip.add(*leg)
        assert trip.legs() == before
    for minimum in (-1, 2.5):
        with pytest.raises(ValueError):
            Itinerary(minimum)


def test_insertion_order_and_snapshot():
    trip = Itinerary()
    trip.add("A", "B", "2@10:00", "2@11:00")
    before = trip.legs()
    trip.add("B", "C", "0@08:00", "0@09:00")
    assert trip.elapsed() == -2940
    assert trip.validate() == [(1, "overlap")]
    assert before == [("A", "B", 3480, 3540)]
    before.clear()
    assert len(trip.legs()) == 2


def test_day_boundaries_and_exact_location_strings():
    trip = Itinerary(15)
    trip.add("A", "b", "0@23:30", "1@00:00")
    trip.add("B", " C ", "1@00:15", "1@01:00")
    trip.add(" C ", "D", "1@01:15", "1@02:00")
    assert trip.validate() == [(1, "location")]
    assert trip.elapsed() == 150
