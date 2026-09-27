class Store:

    def __init__(self):
        raise NotImplementedError()

    def add(self, product, quantity=0):
        raise NotImplementedError()

    def adjust(self, sku, delta):
        raise NotImplementedError()

    def get(self, sku):
        raise NotImplementedError()

    def items(self):
        raise NotImplementedError()

    def total_value(self):
        raise NotImplementedError()

    def low_stock(self, threshold):
        raise NotImplementedError()
