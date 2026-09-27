import re

def normalize_word(word: str, allow_empty: bool=False) -> str:
    """Validate the shared vocabulary and canonicalize case."""
    raise NotImplementedError()

class TrieNode:
    """Zero score denotes a prefix that is not currently a complete word."""

    def __init__(self):
        raise NotImplementedError()

    def find(self, suffix: str):
        raise NotImplementedError()

    def collect(self, prefix: str) -> list[tuple[str, int]]:
        raise NotImplementedError()
