from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int

    def __post_init__(self):
        if not isinstance(self.sku, str) or re.fullmatch(r'[A-Z0-9][A-Z0-9_-]*', self.sku) is None:
            raise ValueError('invalid sku')
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('invalid name')
        if type(self.price_cents) is not int or self.price_cents < 0:
            raise ValueError('invalid price')
