from collections import Counter


def normalize_words(words, ignore_case: bool = False) -> list[str]:
    result = []
    for word in words:
        if not isinstance(word, str):
            raise TypeError("words must be strings")
        word = word.strip()
        if not word:
            continue
        if ignore_case:
            word = word.casefold()
        result.append(word)
    return result


def diff_words(before, after, ignore_case: bool = False) -> dict[str, list[str]]:
    before = normalize_words(before, ignore_case)
    after = normalize_words(after, ignore_case)
    available = Counter(after)
    matched = Counter()
    removed = []
    unchanged = []
    for word in before:
        if available[word] > 0:
            available[word] -= 1
            matched[word] += 1
            unchanged.append(word)
        else:
            removed.append(word)
    added = []
    for word in after:
        if matched[word] > 0:
            matched[word] -= 1
        else:
            added.append(word)
    return {"added": added, "removed": removed, "unchanged": unchanged}


class WordDiff:
    """Keep normalized snapshots so results cannot alias the caller."""

    def __init__(self, before, after, ignore_case: bool = False):
        self._before = normalize_words(before, ignore_case)
        self._after = normalize_words(after, ignore_case)
        self._result = diff_words(self._before, self._after)

    def result(self) -> dict[str, list[str]]:
        snapshot = {}
        for name, words in self._result.items():
            snapshot[name] = list(words)
        return snapshot

    def has_changes(self) -> bool:
        added = self._result["added"]
        removed = self._result["removed"]
        return bool(added or removed)

    def reversed(self) -> "WordDiff":
        # Recompute the matches because their order depends on the direction.
        return WordDiff(self._after, self._before)

    def format(self) -> str:
        """Render changes before retained words, without a trailing newline."""
        lines = []
        for name, prefix in (("removed", "-"), ("added", "+"), ("unchanged", "=")):
            for word in self._result[name]:
                lines.append(prefix + " " + word)
        return "\n".join(lines)
