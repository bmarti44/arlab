from trie_nodes import TrieNode, normalize_word

class Trie:
    """Word scores accumulated along shared character prefixes."""

    def __init__(self):
        raise NotImplementedError()

    def add(self, word: str, weight: int=1) -> int:
        raise NotImplementedError()

    def remove(self, word: str) -> bool:
        raise NotImplementedError()

    def contains(self, word: str) -> bool:
        raise NotImplementedError()

    def prefix_search(self, prefix: str) -> list[str]:
        raise NotImplementedError()

    def autocomplete(self, prefix: str, limit: int=5) -> list[tuple[str, int]]:
        raise NotImplementedError()
