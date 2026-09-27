class Cache:
    def __init__(self, capacity: int, policy: str):
        raise NotImplementedError

    def get(self, key: str, default=None):
        raise NotImplementedError

    def put(self, key: str, value) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def snapshot(self) -> list[tuple[str, object]]:
        raise NotImplementedError


def lru_cache(capacity: int) -> Cache:
    raise NotImplementedError


def lfu_cache(capacity: int) -> Cache:
    raise NotImplementedError
