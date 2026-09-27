from copy import deepcopy


def parse_path(path: str) -> list[str]:
    """Decode escaped separators without losing literal backslashes."""
    parts = []
    current = []
    escaped = False
    for char in path:
        if escaped:
            if char not in (".", "\\"):
                raise ValueError("invalid escape")
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            if not current:
                raise ValueError("empty segment")
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped or not current:
        raise ValueError("incomplete path")
    parts.append("".join(current))
    return parts


class PathStore:
    def __init__(self, data: dict | None = None):
        self._data = deepcopy({} if data is None else data)

    def get(self, path: str, default=None):
        keys = parse_path(path)
        current = self._data
        for key in keys:
            if not isinstance(current, dict):
                raise TypeError("intermediate value is not a dict")
            if key not in current:
                return deepcopy(default)
            current = current[key]
        return deepcopy(current)

    def set(self, path: str, value) -> None:
        keys = parse_path(path)
        candidate = deepcopy(self._data)
        current = candidate
        for key in keys[:-1]:
            if not isinstance(current, dict):
                raise TypeError("intermediate value is not a dict")
            if key not in current:
                current[key] = {}
            current = current[key]
        if not isinstance(current, dict):
            raise TypeError("intermediate value is not a dict")
        current[keys[-1]] = deepcopy(value)
        self._data = candidate

    def delete(self, path: str) -> bool:
        keys = parse_path(path)
        current = self._data
        for key in keys[:-1]:
            if not isinstance(current, dict):
                raise TypeError("intermediate value is not a dict")
            if key not in current:
                return False
            current = current[key]
        if not isinstance(current, dict):
            raise TypeError("intermediate value is not a dict")
        if keys[-1] not in current:
            return False
        del current[keys[-1]]
        return True

    def snapshot(self) -> dict:
        return deepcopy(self._data)
