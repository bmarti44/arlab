import pytest
from text_table import TextTable


def test_mixed_alignment():
    table = TextTable(["Name", "Qty"], ["left", "right"])
    table.add_row(["Al", 2])
    table.add_row(["Beatrice", 1200])
    assert table.render() == "Name     |  Qty\n---------+-----\nAl       |    2\nBeatrice | 1200"
    assert table.row_count() == 2


def test_empty_table_and_zero_width():
    assert TextTable(["A", "B"]).render() == "A | B\n--+--"
    assert TextTable([""]).render() == "\n"


def test_cells_are_text_and_padding_is_preserved():
    table = TextTable(["a", "wide"])
    table.add_row([None, "x"])
    table.add_row([False, ""])
    assert table.render() == "a     | wide\n------+-----\nNone  | x   \nFalse |     "


def test_clear_recomputes_widths():
    table = TextTable(["x"], ["right"])
    table.add_row(["abcdef"])
    table.clear()
    assert table.row_count() == 0
    assert table.render() == "x\n-"
    table.add_row([12])
    assert table.render() == " x\n--\n12"


def test_invalid_configuration():
    for headers, alignments in [([], None), (["a"], []), (["a"], ["center"]), (["a\nb"], None), (["x\r"], None)]:
        with pytest.raises(ValueError):
            TextTable(headers, alignments)


def test_invalid_rows_are_atomic_and_inputs_copied():
    headers, aligns, row = ["H"], ["left"], ["ok"]
    table = TextTable(headers, aligns)
    table.add_row(row)
    headers[0], aligns[0], row[0] = "changed", "right", "changed"
    for bad in [[], ["a", "b"], ["a\nb"], ["a\rb"]]:
        with pytest.raises(ValueError):
            table.add_row(bad)
    assert table.row_count() == 1
    assert table.render() == "H \n--\nok"
