class VendingMachine:
    def __init__(self, inventory: dict[str, tuple[int, int]]):
        raise NotImplementedError

    def select(self, code: str) -> None:
        raise NotImplementedError

    def insert(self, cents: int) -> int:
        raise NotImplementedError

    def vend(self) -> dict:
        raise NotImplementedError

    def cancel(self) -> int:
        raise NotImplementedError

    def restock(self, code: str, amount: int) -> None:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError
