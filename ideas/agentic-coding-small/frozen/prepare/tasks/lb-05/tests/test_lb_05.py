import pytest
from bank_account import BankAccount


def test_opening_balance_validation():
    assert BankAccount().balance == 0
    a = BankAccount(100)
    assert a.balance == 100
    assert a.history() == []
    for bad in [-1, True, 1.5, "100"]:
        with pytest.raises(ValueError):
            BankAccount(bad)


def test_deposit_withdraw_and_ordered_ledger():
    a = BankAccount(100)
    assert a.deposit(50) == 150
    assert a.withdraw(20) == 130
    assert a.withdraw(130) == 0
    assert a.history() == [("deposit", 50, 150), ("withdraw", 20, 130),
                           ("withdraw", 130, 0)]


def test_transfer_records_both_accounts():
    a, b = BankAccount(90), BankAccount(10)
    assert a.transfer_to(b, 90) is None
    assert (a.balance, b.balance) == (0, 100)
    assert a.history() == [("transfer_out", 90, 0)]
    assert b.history() == [("transfer_in", 90, 100)]
    b.transfer_to(a, 25)
    assert a.history()[-1] == ("transfer_in", 25, 25)
    assert b.history()[-1] == ("transfer_out", 25, 75)


def test_invalid_amounts_are_atomic():
    a, b = BankAccount(50), BankAccount()
    a.deposit(10)
    saved = a.history()
    for amount in [0, -1, True, 2.5, "3"]:
        for operation in [a.deposit, a.withdraw, lambda value: a.transfer_to(b, value)]:
            with pytest.raises(ValueError):
                operation(amount)
    assert a.balance == 60
    assert a.history() == saved
    assert b.balance == 0 and b.history() == []


def test_rejected_withdrawals_and_destinations():
    a, b = BankAccount(5), BankAccount(2)
    for operation in [lambda: a.withdraw(6), lambda: a.transfer_to(b, 6),
                      lambda: a.transfer_to(a, 1), lambda: a.transfer_to(None, 1)]:
        with pytest.raises(ValueError):
            operation()
    assert (a.balance, b.balance) == (5, 2)
    assert a.history() == [] and b.history() == []


def test_history_is_a_snapshot():
    a = BankAccount()
    a.deposit(3)
    old = a.history()
    old.append(("withdraw", 3, 0))
    a.deposit(2)
    assert a.history() == [("deposit", 3, 3), ("deposit", 2, 5)]
    old.clear()
    assert len(a.history()) == 2
