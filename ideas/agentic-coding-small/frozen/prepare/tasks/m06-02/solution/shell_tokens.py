class ShellTokenizer:
    """Split a command string without executing or expanding it."""

    def __init__(self, text):
        self.text = text
        self.position = 0

    def tokens(self):
        """Parse from the beginning on every call."""
        self.position = 0
        result = []
        while self.position < len(self.text):
            if self.text[self.position] in ' \t\r\n':
                self.position += 1
                continue
            result.append(self._word())
        return result

    def _word(self):
        """Join adjacent ordinary and quoted fragments."""
        pieces = []
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in ' \t\r\n':
                break
            self.position += 1
            if char in "\"'":
                pieces.append(self._quoted(char))
            elif char == '\\':
                if self.position == len(self.text):
                    raise ValueError('trailing escape')
                pieces.append(self.text[self.position])
                self.position += 1
            else:
                pieces.append(char)
        return ''.join(pieces)

    def _quoted(self, quote):
        """Read one fragment after its opening quote."""
        pieces = []
        while self.position < len(self.text):
            char = self.text[self.position]
            self.position += 1
            if char == quote:
                return ''.join(pieces)
            if char == '\\' and quote == '"':
                if self.position == len(self.text):
                    raise ValueError('trailing escape')
                char = self.text[self.position]
                self.position += 1
            pieces.append(char)
        raise ValueError('unmatched quote')
