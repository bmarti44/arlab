from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    id: str
    artist: str
    seconds: int


def prepare_tracks(tracks) -> list[Track]:
    """Materialize and validate the playlist without changing its order."""
    result = list(tracks)
    if len(result) > 9:
        raise ValueError("at most nine tracks are supported")
    ids = set()
    for track in result:
        if not isinstance(track, Track):
            raise ValueError("expected Track")
        if not isinstance(track.id, str) or not track.id:
            raise ValueError("invalid track id")
        if not isinstance(track.artist, str) or not track.artist:
            raise ValueError("invalid artist")
        if type(track.seconds) is not int or track.seconds <= 0:
            raise ValueError("invalid duration")
        if track.id in ids:
            raise ValueError("duplicate track id")
        ids.add(track.id)
    return result
