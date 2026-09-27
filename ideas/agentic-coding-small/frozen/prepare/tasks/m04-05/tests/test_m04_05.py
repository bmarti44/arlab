import pytest
from pricing import RateTable, format_money, format_invoice


def test_graduated_boundaries():
    table = RateTable([(10, 100), (20, 80), (None, 50)])
    assert [table.quote(q) for q in [0, 1, 10, 11, 20, 23]] == [0, 100, 1000, 1080, 1800, 1950]
    assert table.breakdown(23) == [(10, 100, 1000), (10, 80, 800), (3, 50, 150)]
    assert table.breakdown(10) == [(10, 100, 1000)]
    assert table.breakdown(0) == []


def test_free_and_increasing_prices():
    table = RateTable([(2, 0), (5, 10), (None, 20)])
    assert table.breakdown(7) == [(2, 0, 0), (3, 10, 30), (2, 20, 40)]
    assert table.quote(7) == 70
    assert RateTable([(None, 3)]).quote(1000000) == 3000000


def test_invalid_tables():
    for tiers in [[], [(10, 2)], [(None, 2), (None, 3)], [(0, 1), (None, 1)],
                  [(-1, 2), (None, 2)], [(3, 2), (3, 1), (None, 1)],
                  [(4, 1), (2, 1), (None, 1)], [(None, -1)]]:
        with pytest.raises(ValueError):
            RateTable(tiers)


def test_copy_iterable_and_result_isolation():
    tiers = [[2, 50], [None, 25]]
    table = RateTable(iter(tiers))
    tiers[0][1] = 999
    assert table.quote(3) == 125
    result = table.breakdown(3)
    result.clear()
    assert table.breakdown(3) == [(2, 50, 100), (1, 25, 25)]


def test_money_exactness():
    assert [format_money(c) for c in [0, 1, 10, 100, 12345]] == ['$0.00', '$0.01', '$0.10', '$1.00', '$123.45']
    assert format_money(10**20 + 1) == '$1000000000000000000.01'
    with pytest.raises(ValueError):
        format_money(-1)


def test_invoice_order_and_independent_tiers():
    table = RateTable([(2, 100), (None, 50)])
    items = iter([('API', 3), ('Storage', 0), ('API', 1)])
    assert format_invoice(table, items) == 'API: 3 units = $2.50\nStorage: 0 units = $0.00\nAPI: 1 units = $1.00\nTotal: $3.50'
    assert format_invoice(table, []) == 'Total: $0.00'


def test_negative_quantities():
    table = RateTable([(None, 5)])
    with pytest.raises(ValueError):
        table.quote(-1)
    with pytest.raises(ValueError):
        table.breakdown(-2)
    with pytest.raises(ValueError):
        format_invoice(table, [('bad', -1)])
