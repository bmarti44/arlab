def parse_reading(text: str) -> tuple[float, str]:
    raise NotImplementedError


def convert(value: int | float, from_unit: str, to_unit: str) -> float:
    raise NotImplementedError
