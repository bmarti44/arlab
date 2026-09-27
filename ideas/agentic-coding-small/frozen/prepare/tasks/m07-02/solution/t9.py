class T9Index:
    """A small mutable index from digit strings to normalized words."""

    _codes = {
        char: digit
        for digit, letters in zip("23456789", ("abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"))
        for char in letters
    }

    def __init__(self, words=()):
        self._words = {}
        for word in words:
            self.add(word)

    def add(self, word: str) -> bool:
        normalized = word.lower()
        if not word or any(char not in self._codes for char in normalized) or not word.isascii():
            raise ValueError("invalid word")
        if normalized in self._words:
            return False
        self._words[normalized] = "".join(self._codes[char] for char in normalized)
        return True

    def remove(self, word: str) -> bool:
        normalized = word.lower()
        if not word or any(char not in self._codes for char in normalized) or not word.isascii():
            raise ValueError("invalid word")
        if normalized not in self._words:
            return False
        del self._words[normalized]
        return True

    def lookup(self, digits: str) -> list[str]:
        if any(char not in "23456789" for char in digits):
            raise ValueError("invalid digits")
        return sorted(word for word, code in self._words.items() if code == digits)

    def complete(self, prefix: str, limit: int = 10) -> list[str]:
        if any(char not in "23456789" for char in prefix):
            raise ValueError("invalid prefix")
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("invalid limit")
        matches = sorted(word for word, code in self._words.items() if code.startswith(prefix))
        return matches[:limit]
