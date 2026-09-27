def _tokenize(pattern):
    """Turn escaped syntax into literal tokens before matching text."""
    tokens = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 1
            if index == len(pattern):
                raise ValueError("trailing escape")
            tokens.append(("literal", pattern[index]))
        elif char == "*":
            if not tokens or tokens[-1][0] != "star":
                tokens.append(("star", None))
        elif char == "?":
            tokens.append(("any", None))
        else:
            tokens.append(("literal", char))
        index += 1
    return tokens


class GlobPattern:
    """Compile once and match with one dynamic-programming row per token.

    A row records which text prefixes match the pattern prefix processed so far.
    This avoids reconsidering the same wildcard alternatives recursively.
    """

    def __init__(self, pattern):
        self._tokens = _tokenize(pattern)

    def matches(self, text):
        """Return whether the complete text matches the complete pattern."""
        if not self._tokens:
            return text == ""
        if not text:
            return all(kind == "star" for kind, value in self._tokens)
        previous = [False] * (len(text) + 1)
        previous[0] = True
        for kind, value in self._tokens:
            current = [False] * (len(text) + 1)
            if kind == "star":
                current[0] = previous[0]
                for index in range(1, len(text) + 1):
                    # Either consume no text, or extend this star by one.
                    current[index] = previous[index] or current[index - 1]
            else:
                for index in range(1, len(text) + 1):
                    accepts = kind == "any" or text[index - 1] == value
                    current[index] = previous[index - 1] and accepts
            previous = current
        return previous[-1]

    def filter(self, strings):
        """Preserve input order and duplicates, including for generators."""
        result = []
        for text in strings:
            if self.matches(text):
                result.append(text)
        return result


def glob_match(pattern, text):
    """Validate and match a pattern for a single call.

    Construction deliberately happens even for empty text, so a malformed
    escape never gets hidden by a fast path for an empty candidate.
    """
    compiled = GlobPattern(pattern)
    return compiled.matches(text)
