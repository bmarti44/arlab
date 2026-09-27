import re


class IdentifierConverter:
    """Convert ASCII code identifiers using explicit acronym boundaries."""

    _valid = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*", re.ASCII)
    _tokens = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")

    def split(self, name: str) -> list[str]:
        """Return normalized words after checking the entire identifier."""
        if self._valid.fullmatch(name) is None:
            raise ValueError("invalid identifier")
        words = []
        for chunk in name.split("_"):
            for match in self._tokens.finditer(chunk):
                words.append(match.group().lower())
        return words

    def to_snake(self, name: str) -> str:
        """Separate every token, including numeric tokens, by underscores."""
        words = self.split(name)
        return "_".join(words)

    def to_camel(self, name: str, upper: bool = False) -> str:
        """Render normalized tokens in lower or upper camel case."""
        words = self.split(name)
        rendered = []
        for index, word in enumerate(words):
            if index == 0 and not upper:
                rendered.append(word)
            else:
                rendered.append(word.capitalize())
        return "".join(rendered)

    def convert(self, name: str, style: str) -> str:
        """Select one of the three supported output conventions."""
        if style == "snake":
            return self.to_snake(name)
        if style == "camel":
            return self.to_camel(name)
        if style == "pascal":
            return self.to_camel(name, upper=True)
        raise ValueError("unknown style")
