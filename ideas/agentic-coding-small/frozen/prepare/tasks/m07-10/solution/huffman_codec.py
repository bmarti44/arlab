"""Validate a prefix code and use its trie for decoding."""
from huffman_tree import build_codes


def _trie(codes):
    root = {}
    for char, code in codes.items():
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError("invalid symbol")
        if not isinstance(code, str) or not code or any(bit not in "01" for bit in code):
            raise ValueError("invalid codeword")
        node = root
        for bit in code:
            if "symbol" in node:
                raise ValueError("codeword extends another code")
            node = node.setdefault(bit, {})
        if node:
            raise ValueError("duplicate code or prefix conflict")
        node["symbol"] = char
    return root


def encode(text: str, codes: dict[str, str]) -> str:
    _trie(codes)
    pieces = []
    for char in text:
        if char not in codes:
            raise ValueError("symbol absent from codebook")
        pieces.append(codes[char])
    return "".join(pieces)


def decode(bits: str, codes: dict[str, str]) -> str:
    root = _trie(codes)
    node = root
    output = []
    for bit in bits:
        if bit not in "01":
            raise ValueError("nonbinary input")
        if bit not in node:
            raise ValueError("bit sequence has no code")
        node = node[bit]
        if "symbol" in node:
            output.append(node["symbol"])
            node = root
    if node is not root:
        raise ValueError("truncated codeword")
    return "".join(output)


def compress(text: str) -> tuple[str, dict[str, str]]:
    codes = build_codes(text)
    bits = encode(text, codes)
    return bits, codes
