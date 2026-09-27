def normalize_words(words, ignore_case: bool = False) -> list[str]:
    raise NotImplementedError


def diff_words(before, after, ignore_case: bool = False) -> dict[str, list[str]]:
    raise NotImplementedError


class WordDiff:
    def __init__(self, before, after, ignore_case: bool = False):
        raise NotImplementedError

    def result(self) -> dict[str, list[str]]:
        raise NotImplementedError

    def has_changes(self) -> bool:
        raise NotImplementedError

    def reversed(self) -> "WordDiff":
        raise NotImplementedError

    def format(self) -> str:
        raise NotImplementedError
