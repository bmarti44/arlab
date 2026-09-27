class RateTable:
    def __init__(self, tiers):
        raise NotImplementedError

    def breakdown(self, quantity):
        raise NotImplementedError

    def quote(self, quantity):
        raise NotImplementedError


def format_money(cents):
    raise NotImplementedError


def format_invoice(table, items):
    raise NotImplementedError
