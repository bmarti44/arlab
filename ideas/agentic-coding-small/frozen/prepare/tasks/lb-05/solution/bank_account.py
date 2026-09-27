class BankAccount:
    """An integer-cent account with an append-only internal ledger."""

    def __init__(self, opening_balance=0):
        if type(opening_balance) is not int or opening_balance < 0:
            raise ValueError("opening balance must be nonnegative cents")
        self._balance = opening_balance
        self._history = []

    @property
    def balance(self):
        """Return the available integer-cent balance."""
        return self._balance

    def deposit(self, amount):
        """Add positive cents and return the new balance."""
        if type(amount) is not int or amount <= 0:
            raise ValueError("amount must be positive integer cents")
        self._balance += amount
        self._history.append(("deposit", amount, self._balance))
        return self._balance

    def withdraw(self, amount):
        """Withdraw cents after validating the entire operation."""
        if type(amount) is not int or amount <= 0:
            raise ValueError("amount must be positive integer cents")
        if amount > self._balance:
            raise ValueError("insufficient funds")
        self._balance -= amount
        self._history.append(("withdraw", amount, self._balance))
        return self._balance

    def transfer_to(self, other, amount):
        """Move cents atomically, recording one entry in each ledger."""
        if not isinstance(other, BankAccount) or other is self:
            raise ValueError("destination must be a different account")
        if type(amount) is not int or amount <= 0:
            raise ValueError("amount must be positive integer cents")
        if amount > self._balance:
            raise ValueError("insufficient funds")
        self._balance -= amount
        other._balance += amount
        self._history.append(("transfer_out", amount, self._balance))
        other._history.append(("transfer_in", amount, other._balance))
        return None

    def history(self):
        """Return a fresh list of immutable transaction tuples."""
        return self._history[:]
