class SalesReport:
    def __init__(self):
        raise NotImplementedError

    def add(self, group: str, item: str, quantity: int, unit_cents: int) -> None:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError
