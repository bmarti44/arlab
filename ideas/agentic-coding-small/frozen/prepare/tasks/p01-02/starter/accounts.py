from dataclasses import dataclass

@dataclass
class Account:
    account_id: str
    balance: int = 0
    overdraft: int = 0

    def can_withdraw(self, amount):
        # The old implementation treats the boundary as exclusive.
        return self.balance - amount > -self.overdraft
