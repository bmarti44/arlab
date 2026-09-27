import re


def normalize_word(word: str, allow_empty: bool = False) -> str:
    """Validate the shared vocabulary and canonicalize case."""
    if not isinstance(word, str):
        raise ValueError("word must be a string")
    if word == "" and allow_empty:
        return ""
    if re.fullmatch(r"[A-Za-z]+", word) is None:
        raise ValueError("word must contain ASCII letters only")
    return word.lower()


class TrieNode:
    """Zero score denotes a prefix that is not currently a complete word."""

    def __init__(self):
        self.children = {}
        self.score = 0

    def find(self, suffix: str):
        current = self
        for letter in suffix:
            current = current.children.get(letter)
            if current is None:
                return None
        return current

    def collect(self, prefix: str) -> list[tuple[str, int]]:
        result = []
        stack = [(self, prefix)]
        while stack:
            node, word = stack.pop()
            if node.score:
                result.append((word, node.score))
            for letter, child in node.children.items():
                stack.append((child, word + letter))
        return result
