import pytest
from vending import VendingMachine


def test_purchase_state_transitions():
    machine = VendingMachine({'A': (35, 2)})
    assert machine.status() == {'state': 'idle', 'credit': 0, 'selected': None, 'stock': {'A': 2}}
    assert machine.select('A') is None
    assert machine.status()['state'] == 'collecting'
    assert machine.insert(25) == 25
    assert machine.insert(10) == 35
    assert machine.status()['state'] == 'ready'
    assert machine.vend() == {'item': 'A', 'change': 0}
    assert machine.status() == {'state': 'idle', 'credit': 0, 'selected': None, 'stock': {'A': 1}}


def test_prepaid_change_and_cancel():
    machine = VendingMachine({'A': (25, 1)})
    machine.insert(100)
    assert machine.status()['state'] == 'idle'
    machine.select('A')
    assert machine.vend() == {'item': 'A', 'change': 75}
    assert machine.cancel() == 0
    machine.insert(5)
    assert machine.cancel() == 5
    assert machine.status()['credit'] == 0


def test_selection_changes_preserve_credit():
    machine = VendingMachine({'A': (25, 1), 'B': (100, 2)})
    machine.select('A')
    machine.insert(25)
    machine.select('B')
    assert machine.status()['state'] == 'collecting'
    assert machine.status()['credit'] == 25
    assert machine.cancel() == 25
    assert machine.status()['selected'] is None
    assert machine.status()['stock'] == {'A': 1, 'B': 2}


def test_failures_preserve_transaction():
    machine = VendingMachine({'A': (50, 1), 'B': (25, 0)})
    with pytest.raises(ValueError):
        machine.vend()
    machine.select('A')
    machine.insert(10)
    before = machine.status()
    with pytest.raises(ValueError):
        machine.vend()
    with pytest.raises(ValueError):
        machine.select('B')
    with pytest.raises(KeyError):
        machine.select('missing')
    for coin in [1, -5, True, 25.0, '25']:
        with pytest.raises(ValueError):
            machine.insert(coin)
    assert machine.status() == before


def test_sold_out_and_restock():
    machine = VendingMachine({'A': (5, 1)})
    machine.select('A')
    machine.insert(5)
    machine.vend()
    with pytest.raises(ValueError):
        machine.select('A')
    assert machine.restock('A', 2) is None
    machine.select('A')
    machine.insert(5)
    machine.restock('A', 1)
    assert machine.status() == {'state': 'ready', 'credit': 5, 'selected': 'A', 'stock': {'A': 3}}
    for amount in [0, -1, True, 2.0]:
        with pytest.raises(ValueError):
            machine.restock('A', amount)
    with pytest.raises(KeyError):
        machine.restock('X', 1)
    assert machine.status()['stock']['A'] == 3


def test_inventory_validation():
    for inventory in [{'': (5, 1)}, {1: (5, 1)}, {'A': (0, 1)},
                      {'A': (True, 1)}, {'A': (5, -1)}, {'A': (5, False)}]:
        with pytest.raises(ValueError):
            VendingMachine(inventory)
    assert VendingMachine({}).status()['stock'] == {}


def test_inventory_and_snapshot_isolation():
    inventory = {'A': (10, 2)}
    machine = VendingMachine(inventory)
    inventory['A'] = (99, 0)
    snapshot = machine.status()
    snapshot['stock']['A'] = 100
    snapshot['credit'] = 500
    machine.select('A')
    machine.insert(10)
    assert machine.vend() == {'item': 'A', 'change': 0}
    assert machine.status()['stock'] == {'A': 1}
