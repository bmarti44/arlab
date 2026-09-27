def parse_heading(line: str) -> tuple[int, str] | None:
    raise NotImplementedError


class Outline:
    def __init__(self):
        self._entries = []

    def add(self, markdown: str) -> None:
        raise NotImplementedError

    def entries(self) -> list[tuple[int, str, str]]:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
