import copy
import pytest
from sales import parse_sales, aggregate_sales

def test_parse_reordered_headers_and_exact_prices():
    text = '\n product ,unit_price, region,quantity\n Tea ,2.5, North ,2\nCoffee,0.05,South,3\nWater,2,North,1\nFree,00.00,South,1\n'
    assert parse_sales(text) == [
        {'region': 'North', 'product': 'Tea', 'quantity': 2, 'unit_price_cents': 250},
        {'region': 'South', 'product': 'Coffee', 'quantity': 3, 'unit_price_cents': 5},
        {'region': 'North', 'product': 'Water', 'quantity': 1, 'unit_price_cents': 200},
        {'region': 'South', 'product': 'Free', 'quantity': 1, 'unit_price_cents': 0},
    ]

def test_csv_quoted_fields_blank_records_and_header_only():
    text = 'region,product,quantity,unit_price\n\n"East, coast","Tea ""green""\nloose",2,1.20\n'
    assert parse_sales(text) == [{'region': 'East, coast', 'product': 'Tea "green"\nloose', 'quantity': 2, 'unit_price_cents': 120}]
    assert parse_sales('region,product,quantity,unit_price\n\n') == []

def test_invalid_headers_and_record_shapes():
    for text in ('', '\n\n', 'region,product,quantity\n', 'region,product,quantity,quantity\n', 'region,product,quantity,other\n', 'region,product,quantity,unit_price,extra\n', 'region,product,quantity,unit_price\nA,B,1\n', 'region,product,quantity,unit_price\nA,B,1,2,x\n', 'region,product,quantity,unit_price\nA,"unterminated,1,2'):
        with pytest.raises(ValueError):
            parse_sales(text)

def test_invalid_required_fields_and_numbers():
    header = 'region,product,quantity,unit_price\n'
    for row in (' ,Tea,1,2', 'North, ,1,2', 'N,T,0,1', 'N,T,-1,1', 'N,T,01,1', 'N,T,1.0,1', 'N,T,1,-2', 'N,T,1,+2', 'N,T,1,1.234', 'N,T,1,.5', 'N,T,1,2.', 'N,T,1,nan'):
        with pytest.raises(ValueError):
            parse_sales(header + row)

def test_region_aggregation_and_revenue_ties():
    rows = parse_sales('region,product,quantity,unit_price\nB,Tea,2,1.50\nA,Tea,1,2\nB,Coffee,1,1\nA,Water,2,1\nC,Free,7,0\n')
    assert aggregate_sales(rows) == [
        {'group': 'A', 'quantity': 3, 'revenue_cents': 400, 'orders': 2},
        {'group': 'B', 'quantity': 3, 'revenue_cents': 400, 'orders': 2},
        {'group': 'C', 'quantity': 7, 'revenue_cents': 0, 'orders': 1},
    ]
    assert [r['group'] for r in aggregate_sales(rows, sort_by='quantity')] == ['C', 'A', 'B']

def test_product_grouping_sorting_and_input_immutability():
    rows = parse_sales('region,product,quantity,unit_price\nN,Z,1,1\nS,A,2,2\nN,Z,2,3\nS,a,1,0.01\n')
    before = copy.deepcopy(rows)
    assert aggregate_sales(rows, 'product', 'group') == [
        {'group': 'A', 'quantity': 2, 'revenue_cents': 400, 'orders': 1},
        {'group': 'Z', 'quantity': 3, 'revenue_cents': 700, 'orders': 2},
        {'group': 'a', 'quantity': 1, 'revenue_cents': 1, 'orders': 1},
    ]
    assert [r['group'] for r in aggregate_sales(rows, 'product', 'revenue')] == ['Z', 'A', 'a']
    assert rows == before

def test_empty_aggregation_and_option_validation():
    assert aggregate_sales([]) == []
    for group_by, sort_by in [('bad', 'revenue'), ('region', 'bad'), ('Region', 'group')]:
        with pytest.raises(ValueError):
            aggregate_sales([], group_by, sort_by)
