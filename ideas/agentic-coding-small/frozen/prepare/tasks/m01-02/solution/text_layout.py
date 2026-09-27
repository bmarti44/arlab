class TextWrapper:
    """Whitespace-normalizing text layout with optional full justification."""

    def __init__(self, width):
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width

    def words(self, text):
        """Treat newlines and tabs as ordinary word separators."""
        return text.split()

    def wrap(self, text):
        """Greedily group words, allowing an oversized word on its own."""
        lines = []
        current = []
        length = 0
        for word in self.words(text):
            candidate_length = length + (1 if current else 0) + len(word)
            if current and candidate_length > self.width:
                lines.append(" ".join(current))
                current = []
                length = 0
            if current:
                length += 1
            current.append(word)
            length += len(word)
        if current:
            lines.append(" ".join(current))
        return lines

    def justify(self, text):
        """Distribute padding across gaps, keeping the final line ragged."""
        lines = self.wrap(text)
        result = []
        for index, line in enumerate(lines):
            words = line.split(" ")
            gaps = len(words) - 1
            if index == len(lines) - 1 or gaps == 0:
                result.append(line)
                continue
            spaces = self.width - sum(map(len, words))
            base, extra = divmod(spaces, gaps)
            pieces = [words[0]]
            for gap, word in enumerate(words[1:]):
                pieces.append(" " * (base + (gap < extra)))
                pieces.append(word)
            result.append("".join(pieces))
        return result

    def format(self, text, justify=False):
        """Render without a final newline or any empty-input special line."""
        lines = self.justify(text) if justify else self.wrap(text)
        return "\n".join(lines)
