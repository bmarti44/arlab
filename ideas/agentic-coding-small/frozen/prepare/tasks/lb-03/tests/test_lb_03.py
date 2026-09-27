import pytest
from date_ranges import dates_between, overlap


def test_single_day():
    assert dates_between("2024-05-10", "2024-05-10") == ["2024-05-10"]


def test_leap_and_month_boundary():
    assert dates_between("2024-02-28", "2024-03-01") == [
        "2024-02-28", "2024-02-29", "2024-03-01"]
    assert dates_between("2023-02-28", "2023-03-01") == [
        "2023-02-28", "2023-03-01"]


def test_year_boundary_and_maximum_date():
    assert dates_between("2023-12-31", "2024-01-02") == [
        "2023-12-31", "2024-01-01", "2024-01-02"]
    assert dates_between("9999-12-31", "9999-12-31") == ["9999-12-31"]


def test_touching_and_disjoint_intersections():
    assert overlap("2024-01-01", "2024-01-03", "2024-01-03", "2024-01-05") == (
        "2024-01-03", "2024-01-03")
    assert overlap("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05") is None


def test_containment_and_singleton_intersection():
    assert overlap("2024-01-01", "2024-01-10", "2024-01-04", "2024-01-04") == (
        "2024-01-04", "2024-01-04")
    assert overlap("2024-01-03", "2024-01-08", "2024-01-01", "2024-01-10") == (
        "2024-01-03", "2024-01-08")


def test_validation_preserved_alongside_inclusive_behavior():
    for bad in ["2023-02-29", "2024-2-01", " 2024-02-01", "0000-01-01", None]:
        with pytest.raises(ValueError):
            dates_between(bad, "2024-03-01")
    with pytest.raises(ValueError):
        dates_between("2024-02-02", "2024-02-01")
    with pytest.raises(ValueError):
        overlap("2024-01-01", "2024-01-02", "2025-01-02", "2025-01-01")
    assert dates_between("0001-01-01", "0001-01-01") == ["0001-01-01"]
