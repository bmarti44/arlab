class ShellTokenizer:
    """Split a command string without executing or expanding it."""

    def __init__(self, text):
        raise NotImplementedError()

    def tokens(self):
        """Parse from the beginning on every call."""
        raise NotImplementedError()

    def _word(self):
        """Join adjacent ordinary and quoted fragments."""
        raise NotImplementedError()

    def _quoted(self, quote):
        """Read one fragment after its opening quote."""
        raise NotImplementedError()
