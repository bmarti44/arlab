class BankAccount:
    def __init__(self, opening_balance=0):
        raise NotImplementedError

    @property
    def balance(self):
        raise NotImplementedError

    def deposit(self, amount):
        raise NotImplementedError

    def withdraw(self, amount):
        raise NotImplementedError

    def transfer_to(self, other, amount):
        raise NotImplementedError

    def history(self):
        raise NotImplementedError
