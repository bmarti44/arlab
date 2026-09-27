class T9Index:
    def __init__(self, words=()):
        raise NotImplementedError

    def add(self, word: str) -> bool:
        raise NotImplementedError

    def remove(self, word: str) -> bool:
        raise NotImplementedError

    def lookup(self, digits: str) -> list[str]:
        raise NotImplementedError

    def complete(self, prefix: str, limit: int = 10) -> list[str]:
        raise NotImplementedError
