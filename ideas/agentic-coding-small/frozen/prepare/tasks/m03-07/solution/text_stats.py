import re
from collections import Counter


_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_SENTENCE_END = re.compile(r"[.!?]+")


def words(text: str) -> list[str]:
    """Extract ASCII words while retaining internal straight apostrophes."""
    return [match.group().lower() for match in _WORD.finditer(text)]


def syllables(word: str) -> int:
    """Estimate syllables with a deterministic vowel-run heuristic."""
    if _WORD.fullmatch(word) is None:
        raise ValueError("expected one word")
    normalized = word.lower().replace("'", "")
    groups = 0
    previous_vowel = False
    for char in normalized:
        current_vowel = char in "aeiouy"
        if current_vowel and not previous_vowel:
            groups += 1
        previous_vowel = current_vowel
    silent_e = (
        len(normalized) > 2
        and normalized.endswith("e")
        and not normalized.endswith("le")
    )
    if silent_e and groups > 1:
        groups -= 1
    return max(1, groups)


class TextStats:
    """Editable text whose reports are computed from its current contents."""

    def __init__(self, text: str = ""):
        self._text = text

    def replace(self, text: str) -> None:
        self._text = text

    def append(self, text: str) -> None:
        self._text += text

    def summary(self) -> dict:
        tokens = words(self._text)
        word_count = len(tokens)
        syllable_count = sum(syllables(word) for word in tokens)
        sentence_count = sum(
            bool(words(chunk))
            for chunk in _SENTENCE_END.split(self._text)
        )
        if word_count == 0:
            ease = 0.0
        else:
            ease = round(
                206.835
                - 1.015 * word_count / sentence_count
                - 84.6 * syllable_count / word_count,
                2,
            )
        return {
            "words": word_count,
            "sentences": sentence_count,
            "syllables": syllable_count,
            "characters": len(self._text),
            "reading_ease": ease,
        }

    def most_common(self, limit: int = 5) -> list[tuple[str, int]]:
        if type(limit) is not int or limit < 0:
            raise ValueError("invalid limit")
        counts = Counter(words(self._text))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
