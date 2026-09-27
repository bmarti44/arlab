from accounts import Account

class Bank:
    def __init__(self):
        self._accounts = {}
        self._history = {}

    def open(self, account_id, balance=0, overdraft=0):
        self._accounts[account_id] = Account(account_id, balance, overdraft)
        self._history[account_id] = []

    def balance(self, account_id):
        return self._accounts[account_id].balance

    def _amount(self, amount):
        if amount < 0:
            raise ValueError('negative amount')

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
        self._history[account_id].append(('withdraw', amount, None))
        return account.balance

    def transfer(self, source, target, amount):
        self.withdraw(source, amount)
        self.deposit(target, amount)

    def history(self, account_id):
        return self._history[account_id]
