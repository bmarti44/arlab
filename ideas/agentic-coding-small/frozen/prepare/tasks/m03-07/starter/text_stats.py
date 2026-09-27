import re
from collections import Counter
_WORD = re.compile("[A-Za-z]+(?:'[A-Za-z]+)*")
_SENTENCE_END = re.compile('[.!?]+')

def words(text: str) -> list[str]:
    """Extract ASCII words while retaining internal straight apostrophes."""
    raise NotImplementedError()

def syllables(word: str) -> int:
    """Estimate syllables with a deterministic vowel-run heuristic."""
    raise NotImplementedError()

class TextStats:
    """Editable text whose reports are computed from its current contents."""

    def __init__(self, text: str=''):
        raise NotImplementedError()

    def replace(self, text: str) -> None:
        raise NotImplementedError()

    def append(self, text: str) -> None:
        raise NotImplementedError()

    def summary(self) -> dict:
        raise NotImplementedError()

    def most_common(self, limit: int=5) -> list[tuple[str, int]]:
        raise NotImplementedError()
