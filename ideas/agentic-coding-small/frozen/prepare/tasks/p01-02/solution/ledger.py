from accounts import Account

class Bank:
    def __init__(self):
        self._accounts = {}
        self._history = {}

    def open(self, account_id, balance=0, overdraft=0):
        if not isinstance(account_id, str) or not account_id:
            raise ValueError('invalid id')
        if account_id in self._accounts:
            raise ValueError('duplicate id')
        if type(balance) is not int or type(overdraft) is not int or overdraft < 0 or balance < -overdraft:
            raise ValueError('invalid opening values')
        self._accounts[account_id] = Account(account_id, balance, overdraft)
        self._history[account_id] = []

    def balance(self, account_id):
        return self._accounts[account_id].balance

    def _amount(self, amount):
        if type(amount) is not int or amount <= 0:
            raise ValueError('invalid amount')

    def deposit(self, account_id, amount):
        account = self._accounts[account_id]
        self._amount(amount)
        account.balance += amount
        self._history[account_id].append(('deposit', amount, None))
        return account.balance

    def withdraw(self, account_id, amount):
        account = self._accounts[account_id]
        self._amount(amount)
        if not account.can_withdraw(amount):
            raise ValueError('insufficient funds')
        account.balance -= amount
        self._history[account_id].append(('withdraw', -amount, None))
        return account.balance

    def transfer(self, source, target, amount):
        src = self._accounts[source]
        dst = self._accounts[target]
        if source == target:
            raise ValueError('same account')
        self._amount(amount)
        if not src.can_withdraw(amount):
            raise ValueError('insufficient funds')
        src.balance -= amount
        dst.balance += amount
        self._history[source].append(('transfer_out', -amount, target))
        self._history[target].append(('transfer_in', amount, source))

    def history(self, account_id):
        return list(self._history[account_id])
