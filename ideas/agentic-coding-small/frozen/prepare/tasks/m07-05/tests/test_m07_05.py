import pytest
from report import SalesReport


def test_empty_report():
    report = SalesReport()
    assert report.render() == "Group | Item | Qty | Amount\nTOTAL |      |   0 |   0.00"


def test_one_group_exact_alignment():
    report = SalesReport()
    assert report.add("Food", "Tea", 2, 125) is None
    report.add("Food", "Cake", 1, 300)
    assert report.render() == (
        "Group | Item     | Qty | Amount\n"
        "Food  | Tea      |   2 |   2.50\n"
        "Food  | Cake     |   1 |   3.00\n"
        "Food  | SUBTOTAL |   3 |   5.50\n"
        "TOTAL |          |   3 |   5.50"
    )


def test_interleaved_groups_and_repeated_items():
    report = SalesReport()
    report.add("B", "x", 1, 100)
    report.add("A", "y", 2, 50)
    report.add("B", "x", 3, 10)
    assert report.render() == (
        "Group | Item     | Qty | Amount\n"
        "B     | x        |   1 |   1.00\n"
        "B     | x        |   3 |   0.30\n"
        "B     | SUBTOTAL |   4 |   1.30\n"
        "A     | y        |   2 |   1.00\n"
        "A     | SUBTOTAL |   2 |   1.00\n"
        "TOTAL |          |   6 |   2.30"
    )


def test_negative_and_zero_amounts():
    report = SalesReport()
    report.add("G", "credit", 1, -5)
    report.add("G", "zero", 0, -999)
    report.add("G", "sale", 1, 5)
    assert report.render() == (
        "Group | Item     | Qty | Amount\n"
        "G     | credit   |   1 |  -0.05\n"
        "G     | zero     |   0 |   0.00\n"
        "G     | sale     |   1 |   0.05\n"
        "G     | SUBTOTAL |   2 |   0.00\n"
        "TOTAL |          |   2 |   0.00"
    )


def test_wide_labels_and_numbers():
    report = SalesReport()
    report.add("Long group", "Long product", 1000, 12345)
    assert report.render() == (
        "Group      | Item         |  Qty |    Amount\n"
        "Long group | Long product | 1000 | 123450.00\n"
        "Long group | SUBTOTAL     | 1000 | 123450.00\n"
        "TOTAL      |              | 1000 | 123450.00"
    )


def test_validation_does_not_append():
    report = SalesReport()
    report.add("G", "ok", 1, 10)
    before = report.render()
    for args in [("", "x", 1, 1), ("G", "a|b", 1, 1), ("a\nb", "x", 1, 1), ("G", "x\r", 1, 1), ("G", "x", -1, 1), ("G", "x", 1.5, 1), ("G", "x", 1, 0.5)]:
        with pytest.raises(ValueError):
            report.add(*args)
        assert report.render() == before


def test_render_repeatability_and_spaces():
    report = SalesReport()
    report.add(" G ", " x ", 1, -150)
    before = report.render()
    assert before == report.render()
    assert before.splitlines()[1] == " G    |  x       |   1 |  -1.50"
    report.add(" G ", " x ", 1, 50)
    assert report.render().splitlines()[-1] == "TOTAL |          |   2 |  -1.00"
    assert before.splitlines()[-1] == "TOTAL |          |   1 |  -1.50"
