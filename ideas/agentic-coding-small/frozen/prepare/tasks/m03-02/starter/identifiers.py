import re

class IdentifierConverter:
    """Convert ASCII code identifiers using explicit acronym boundaries."""
    _valid = re.compile('[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*', re.ASCII)
    _tokens = re.compile('[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+')

    def split(self, name: str) -> list[str]:
        """Return normalized words after checking the entire identifier."""
        raise NotImplementedError()

    def to_snake(self, name: str) -> str:
        """Separate every token, including numeric tokens, by underscores."""
        raise NotImplementedError()

    def to_camel(self, name: str, upper: bool=False) -> str:
        """Render normalized tokens in lower or upper camel case."""
        raise NotImplementedError()

    def convert(self, name: str, style: str) -> str:
        """Select one of the three supported output conventions."""
        raise NotImplementedError()
