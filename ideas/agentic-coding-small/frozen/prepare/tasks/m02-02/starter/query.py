class Query:
    def __init__(self, text: str = ""):
        raise NotImplementedError

    def getall(self, key: str) -> list[str]:
        raise NotImplementedError

    def append(self, key: str, value: str) -> None:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def encode(self) -> str:
        raise NotImplementedError
