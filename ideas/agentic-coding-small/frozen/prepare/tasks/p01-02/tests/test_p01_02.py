import pytest
from accounts import Account
from ledger import Bank

def test_account_boundary_and_amounts():
    a = Account('a', 5, 10)
    assert a.can_withdraw(15) is True
    assert a.can_withdraw(16) is False
    for amount in (0, -1, True, 1.5, '2'):
        assert a.can_withdraw(amount) is False

def test_open_validation_and_duplicates():
    b = Bank()
    b.open('a', -10, 10)
    assert b.balance('a') == -10
    assert b.history('a') == []
    for args in (('a', 100, 0), ('', 0, 0), (5, 0, 0), ('bad', -11, 10), ('bad', 0, -1), ('bad', True, 0), ('bad', 0, False), ('bad', 1.5, 0)):
        with pytest.raises(ValueError):
            b.open(*args)
    assert b.balance('a') == -10
    with pytest.raises(KeyError):
        b.balance('bad')

def test_deposit_and_withdraw_history():
    b = Bank()
    b.open('a', 100, 20)
    assert b.deposit('a', 30) == 130
    assert b.withdraw('a', 150) == -20
    assert b.history('a') == [('deposit', 30, None), ('withdraw', -150, None)]
    with pytest.raises(ValueError):
        b.withdraw('a', 1)
    assert b.balance('a') == -20
    assert len(b.history('a')) == 2

def test_transfer_conserves_money_and_records_once():
    b = Bank()
    b.open('a', 10, 5)
    b.open('b', 20)
    assert b.transfer('a', 'b', 15) is None
    assert (b.balance('a'), b.balance('b')) == (-5, 35)
    assert b.history('a') == [('transfer_out', -15, 'b')]
    assert b.history('b') == [('transfer_in', 15, 'a')]

def test_failed_transfers_are_atomic():
    b = Bank()
    b.open('a', 10)
    b.open('b', 20)
    for args, error in ((('a', 'b', 11), ValueError), (('a', 'absent', 4), KeyError), (('a', 'a', 1), ValueError), (('absent', 'b', 1), KeyError), (('a', 'b', 0), ValueError), (('a', 'b', True), ValueError)):
        with pytest.raises(error):
            b.transfer(*args)
        assert (b.balance('a'), b.balance('b')) == (10, 20)
        assert b.history('a') == b.history('b') == []

def test_invalid_amounts_and_lookup_precedence():
    b = Bank()
    b.open('a', 50)
    for method in (b.deposit, b.withdraw):
        for amount in (0, -1, True, 1.5, '2'):
            with pytest.raises(ValueError):
                method('a', amount)
        with pytest.raises(KeyError):
            method('missing', 0)
    with pytest.raises(KeyError):
        b.transfer('a', 'missing', -1)
    with pytest.raises(KeyError):
        b.history('missing')
    assert b.balance('a') == 50
    assert b.history('a') == []

def test_history_copies_and_bank_isolation():
    a, b = Bank(), Bank()
    a.open('x', 10)
    b.open('x', 20)
    a.deposit('x', 5)
    entries = a.history('x')
    entries.clear()
    assert a.history('x') == [('deposit', 5, None)]
    assert b.balance('x') == 20
    assert b.history('x') == []
