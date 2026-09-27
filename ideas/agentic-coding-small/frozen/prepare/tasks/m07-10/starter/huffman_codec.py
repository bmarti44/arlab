from huffman_tree import build_codes


def encode(text: str, codes: dict[str, str]) -> str:
    raise NotImplementedError


def decode(bits: str, codes: dict[str, str]) -> str:
    raise NotImplementedError


def compress(text: str) -> tuple[str, dict[str, str]]:
    raise NotImplementedError
