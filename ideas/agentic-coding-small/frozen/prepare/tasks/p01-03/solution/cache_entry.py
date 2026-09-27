from dataclasses import dataclass

@dataclass
class Entry:
    value: object
    expires_at: float | None

    def expired(self, now):
        return self.expires_at is not None and now >= self.expires_at
