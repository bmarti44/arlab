from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    id: str
    artist: str
    seconds: int


def prepare_tracks(tracks) -> list[Track]:
    raise NotImplementedError
