from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int

    def __post_init__(self):
        raise NotImplementedError()
