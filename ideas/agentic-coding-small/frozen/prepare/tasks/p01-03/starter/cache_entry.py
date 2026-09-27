from dataclasses import dataclass

@dataclass
class Entry:
    value: object
    expires_at: float | None

    def expired(self, now):
        raise NotImplementedError()
