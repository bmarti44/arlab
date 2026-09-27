def _tokenize(pattern):
    """Turn escaped syntax into literal tokens before matching text."""
    raise NotImplementedError()

class GlobPattern:
    """Compile once and match with one dynamic-programming row per token.

    A row records which text prefixes match the pattern prefix processed so far.
    This avoids reconsidering the same wildcard alternatives recursively.
    """

    def __init__(self, pattern):
        raise NotImplementedError()

    def matches(self, text):
        """Return whether the complete text matches the complete pattern."""
        raise NotImplementedError()

    def filter(self, strings):
        """Preserve input order and duplicates, including for generators."""
        raise NotImplementedError()

def glob_match(pattern, text):
    """Validate and match a pattern for a single call.

Construction deliberately happens even for empty text, so a malformed
escape never gets hidden by a fast path for an empty candidate."""
    raise NotImplementedError()
