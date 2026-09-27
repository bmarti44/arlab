import csv
import io
from dataclasses import FrozenInstanceError
import pytest
from inventory import Product, Store, render_report
from inventory.models import Product as ModelProduct
from inventory.store import Store as ModuleStore
from inventory.report import render_report as module_report

def test_product_validation_and_exports():
    assert Product is ModelProduct and Store is ModuleStore and render_report is module_report
    p = Product('A-1_X', ' Tea ', 0)
    assert p == Product('A-1_X', ' Tea ', 0)
    assert p.name == ' Tea '
    with pytest.raises(FrozenInstanceError):
        p.price_cents = 1
    for args in (('', 'Tea', 1), ('lower', 'Tea', 1), ('_A', 'Tea', 1), ('A B', 'Tea', 1), ('A', '  ', 1), ('A', None, 1), ('A', 'Tea', -1), ('A', 'Tea', True), ('A', 'Tea', 1.2)):
        with pytest.raises(ValueError):
            Product(*args)

def test_store_add_adjust_and_lookup():
    s = Store()
    a = Product('A', 'Tea', 125)
    s.add(a, 2)
    assert s.get('A') == (a, 2)
    assert s.adjust('A', 3) == 5
    assert s.adjust('A', -5) == 0
    assert s.adjust('A', 0) == 0
    assert s.get('A') == (a, 0)
    with pytest.raises(KeyError):
        s.get('missing')
    with pytest.raises(KeyError):
        s.adjust('missing', True)

def test_store_failures_are_atomic():
    s = Store()
    p = Product('A', 'Tea', 100)
    s.add(p, 2)
    with pytest.raises(ValueError):
        s.add(Product('A', 'Changed', 900), 8)
    for delta in (-3, True, 1.5, '1'):
        with pytest.raises(ValueError):
            s.adjust('A', delta)
    for quantity in (-1, True, 2.5):
        with pytest.raises(ValueError):
            s.add(Product('B', 'Coffee', 100), quantity)
    assert s.items() == [(p, 2)]

def test_sorted_items_low_stock_and_value():
    s = Store()
    for sku, quantity, price in [('Z', 0, 100), ('A', 2, 125), ('M', 3, 5)]:
        s.add(Product(sku, sku, price), quantity)
    assert [p.sku for p, q in s.items()] == ['A', 'M', 'Z']
    assert s.low_stock(2) == ['A', 'Z']
    assert s.low_stock(0) == ['Z']
    assert s.total_value() == 265
    for threshold in (-1, True, 1.5):
        with pytest.raises(ValueError):
            s.low_stock(threshold)

def test_report_exact_order_and_money():
    s = Store()
    s.add(Product('B', 'Tea', 5), 3)
    s.add(Product('A', 'Coffee', 125), 2)
    s.add(Product('Z', 'Free', 0))
    assert render_report(s) == 'sku,name,quantity,unit_price,value\nA,Coffee,2,1.25,2.50\nB,Tea,3,0.05,0.15\nZ,Free,0,0.00,0.00\nTOTAL,,,,2.65\n'
    assert s.total_value() == 265

def test_report_csv_quoting_and_empty_store():
    assert render_report(Store()) == 'sku,name,quantity,unit_price,value\nTOTAL,,,,0.00\n'
    s = Store()
    s.add(Product('A', 'Tea, "green"\nloose', 101), 1)
    text = render_report(s)
    assert text == 'sku,name,quantity,unit_price,value\nA,"Tea, ""green""\nloose",1,1.01,1.01\nTOTAL,,,,1.01\n'
    assert list(csv.reader(io.StringIO(text)))[1] == ['A', 'Tea, "green"\nloose', '1', '1.01', '1.01']

def test_snapshot_lists_and_store_isolation():
    a, b = Store(), Store()
    p = Product('A', 'Tea', 100)
    a.add(p, 2)
    result = a.items()
    result.clear()
    low = a.low_stock(5)
    low.append('X')
    assert a.items() == [(p, 2)]
    assert a.low_stock(5) == ['A']
    assert b.items() == []
    assert b.total_value() == 0
