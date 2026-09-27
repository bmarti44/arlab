from trie_nodes import TrieNode, normalize_word


class Trie:
    """Word scores accumulated along shared character prefixes."""

    def __init__(self):
        self._root = TrieNode()

    def add(self, word: str, weight: int = 1) -> int:
        normalized = normalize_word(word)
        if type(weight) is not int or weight <= 0:
            raise ValueError("weight must be a positive integer")
        node = self._root
        for letter in normalized:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.score += weight
        return node.score

    def remove(self, word: str) -> bool:
        normalized = normalize_word(word)
        node = self._root.find(normalized)
        if node is None or node.score == 0:
            return False
        node.score = 0
        return True

    def contains(self, word: str) -> bool:
        normalized = normalize_word(word)
        node = self._root.find(normalized)
        return node is not None and node.score > 0

    def prefix_search(self, prefix: str) -> list[str]:
        normalized = normalize_word(prefix, allow_empty=True)
        node = self._root.find(normalized)
        if node is None:
            return []
        return sorted(word for word, score in node.collect(normalized))

    def autocomplete(self, prefix: str, limit: int = 5) -> list[tuple[str, int]]:
        normalized = normalize_word(prefix, allow_empty=True)
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        node = self._root.find(normalized)
        if node is None:
            return []
        candidates = node.collect(normalized)
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:limit]
