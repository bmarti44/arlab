import random
from playlist import Track, prepare_tracks


def shuffle_playlist(tracks, seed: int, first: str | None = None) -> list[Track]:
    """Find the first valid path through a once-shuffled candidate list."""
    candidates = prepare_tracks(tracks)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    prefix = []
    if first is not None:
        if not isinstance(first, str):
            raise ValueError("first must be a track id")
        chosen = None
        for index, track in enumerate(candidates):
            if track.id == first:
                chosen = index
                break
        if chosen is None:
            raise ValueError("unknown first track")
        prefix.append(candidates.pop(chosen))
    random.Random(seed).shuffle(candidates)

    # Use an explicit depth-first stack so candidate order stays constant.
    path = list(prefix)
    used = [False] * len(candidates)
    chosen_indices = []
    next_indices = [0]
    target_length = len(prefix) + len(candidates)
    while len(path) < target_length:
        start = next_indices[-1]
        found = None
        for index in range(start, len(candidates)):
            if used[index]:
                continue
            if path and candidates[index].artist == path[-1].artist:
                continue
            found = index
            break
        if found is not None:
            next_indices[-1] = found + 1
            next_indices.append(0)
            used[found] = True
            chosen_indices.append(found)
            path.append(candidates[found])
            continue
        next_indices.pop()
        if not chosen_indices:
            raise ValueError("no valid artist spacing")
        index = chosen_indices.pop()
        used[index] = False
        path.pop()
    return path
