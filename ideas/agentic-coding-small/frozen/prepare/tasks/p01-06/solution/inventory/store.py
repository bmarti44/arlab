class Store:
    def __init__(self):
        self._items = {}

    def add(self, product, quantity=0):
        if product.sku in self._items or type(quantity) is not int or quantity < 0:
            raise ValueError('invalid add')
        self._items[product.sku] = (product, quantity)

    def adjust(self, sku, delta):
        product, quantity = self._items[sku]
        if type(delta) is not int or quantity + delta < 0:
            raise ValueError('invalid adjustment')
        quantity += delta
        self._items[sku] = (product, quantity)
        return quantity

    def get(self, sku):
        return self._items[sku]

    def items(self):
        return [self._items[sku] for sku in sorted(self._items)]

    def total_value(self):
        return sum(p.price_cents * q for p, q in self._items.values())

    def low_stock(self, threshold):
        if type(threshold) is not int or threshold < 0:
            raise ValueError('invalid threshold')
        return [sku for sku in sorted(self._items) if self._items[sku][1] <= threshold]
