from urllib.parse import quote_plus, unquote_plus


class Query:
    """An ordered sequence of decoded query pairs."""

    def __init__(self, text: str = ""):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._pairs = []
        if text.startswith("?"):
            text = text[1:]
        for segment in text.split("&"):
            if not segment:
                continue
            key, separator, value = segment.partition("=")
            self._pairs.append((unquote_plus(key), unquote_plus(value)))

    def getall(self, key: str) -> list[str]:
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        return [value for name, value in self._pairs if name == key]

    def append(self, key: str, value: str) -> None:
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("key and value must be strings")
        self._pairs.append((key, value))

    def set(self, key: str, value: str) -> None:
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("key and value must be strings")
        pairs = []
        found = False
        for name, old_value in self._pairs:
            if name != key:
                pairs.append((name, old_value))
            elif not found:
                pairs.append((key, value))
                found = True
        if not found:
            pairs.append((key, value))
        self._pairs = pairs

    def encode(self) -> str:
        encoded = []
        for key, value in self._pairs:
            encoded.append(quote_plus(key) + "=" + quote_plus(value))
        return "&".join(encoded)
