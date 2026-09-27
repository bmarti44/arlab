class TextWrapper:
    """Whitespace-normalizing text layout with optional full justification."""

    def __init__(self, width):
        raise NotImplementedError()

    def words(self, text):
        """Treat newlines and tabs as ordinary word separators."""
        raise NotImplementedError()

    def wrap(self, text):
        """Greedily group words, allowing an oversized word on its own."""
        raise NotImplementedError()

    def justify(self, text):
        """Distribute padding across gaps, keeping the final line ragged."""
        raise NotImplementedError()

    def format(self, text, justify=False):
        """Render without a final newline or any empty-input special line."""
        raise NotImplementedError()
