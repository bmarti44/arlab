def _integer(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer amount")
    return value


def _copy_inventory(inventory):
    result = {}
    for code, (price, stock) in inventory.items():
        if not isinstance(code, str) or not code:
            raise ValueError("invalid code")
        result[code] = [_integer(price, 1), _integer(stock, 0)]
    return result


class VendingMachine:
    """A selection and credit form one cancelable transaction."""

    def __init__(self, inventory: dict[str, tuple[int, int]]):
        self._inventory = _copy_inventory(inventory)
        self._selected = None
        self._credit = 0

    def select(self, code: str) -> None:
        price, stock = self._inventory[code]
        if stock == 0:
            raise ValueError("sold out")
        self._selected = code

    def insert(self, cents: int) -> int:
        if type(cents) is not int or cents not in (5, 10, 25, 100):
            raise ValueError("unsupported coin")
        self._credit += cents
        return self._credit

    def vend(self) -> dict:
        if self._selected is None:
            raise ValueError("select an item")
        price, stock = self._inventory[self._selected]
        if self._credit < price:
            raise ValueError("insufficient credit")
        code = self._selected
        change = self._credit - price
        self._inventory[code][1] = stock - 1
        self._selected = None
        self._credit = 0
        return {"item": code, "change": change}

    def cancel(self) -> int:
        refund = self._credit
        self._credit = 0
        self._selected = None
        return refund

    def restock(self, code: str, amount: int) -> None:
        if code not in self._inventory:
            raise KeyError(code)
        amount = _integer(amount, 1)
        self._inventory[code][1] += amount

    def status(self) -> dict:
        state = "idle"
        if self._selected is not None:
            price = self._inventory[self._selected][0]
            state = "ready" if self._credit >= price else "collecting"
        stock = {code: data[1] for code, data in self._inventory.items()}
        return {
            "state": state,
            "credit": self._credit,
            "selected": self._selected,
            "stock": stock,
        }
