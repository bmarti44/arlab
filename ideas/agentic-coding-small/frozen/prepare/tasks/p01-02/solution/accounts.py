from dataclasses import dataclass

@dataclass
class Account:
    account_id: str
    balance: int = 0
    overdraft: int = 0

    def can_withdraw(self, amount):
        return type(amount) is int and amount > 0 and self.balance - amount >= -self.overdraft
