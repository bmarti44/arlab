from copy import deepcopy

def parse_path(path: str) -> list[str]:
    """Decode escaped separators without losing literal backslashes."""
    raise NotImplementedError()

class PathStore:

    def __init__(self, data: dict | None=None):
        raise NotImplementedError()

    def get(self, path: str, default=None):
        raise NotImplementedError()

    def set(self, path: str, value) -> None:
        raise NotImplementedError()

    def delete(self, path: str) -> bool:
        raise NotImplementedError()

    def snapshot(self) -> dict:
        raise NotImplementedError()
