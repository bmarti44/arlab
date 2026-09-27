import random
from dataclasses import FrozenInstanceError
import pytest
from playlist import Track, prepare_tracks
from shuffler import shuffle_playlist


def test_prepare_snapshot_and_frozen_track():
    tracks = [Track('a', 'Artist', 30), Track('b', ' ', 1)]
    prepared = prepare_tracks(iter(tracks))
    assert prepared == tracks
    prepared.pop()
    assert len(tracks) == 2
    with pytest.raises(FrozenInstanceError):
        tracks[0].seconds = 5


def test_prepare_validation():
    bad = [Track('', 'X', 1), Track('a', '', 1), Track('a', 'X', 0),
           Track('a', 'X', True), Track('a', 'X', 1.5), Track(1, 'X', 2),
           Track('a', None, 2), ('a', 'X', 1)]
    for track in bad:
        with pytest.raises(ValueError):
            prepare_tracks([track])
    with pytest.raises(ValueError):
        prepare_tracks([Track('a', 'X', 1), Track('a', 'Y', 2)])
    with pytest.raises(ValueError):
        prepare_tracks([Track(str(n), str(n), 1) for n in range(10)])


def test_seeded_order_without_conflicts():
    tracks = [Track(str(n), str(n), n + 1) for n in range(6)]
    expected = list(tracks)
    random.Random(17).shuffle(expected)
    assert shuffle_playlist(tracks, 17) == expected
    assert shuffle_playlist(iter(tracks), 17) == expected
    assert [track.id for track in tracks] == ['0', '1', '2', '3', '4', '5']


def test_backtracking_and_exact_priority():
    tracks = [Track('a1', 'A', 1), Track('a2', 'A', 1), Track('a3', 'A', 1),
              Track('b1', 'B', 1), Track('c1', 'C', 1)]
    # Expected orders are fixed examples of the specified candidate priority.
    assert [t.id for t in shuffle_playlist(tracks, 0)] == ['a3', 'c1', 'a2', 'b1', 'a1']
    assert [t.id for t in shuffle_playlist(tracks, 4)] == ['a1', 'b1', 'a3', 'c1', 'a2']


def test_pinned_first_and_case_sensitive_artist():
    tracks = [Track('a', 'A', 1), Track('b', 'A', 2), Track('c', 'a', 3), Track('d', 'B', 4)]
    candidates = [tracks[0], tracks[2], tracks[3]]
    random.Random(3).shuffle(candidates)
    # For this seed the shuffled suffix is already valid after pinned b.
    assert shuffle_playlist(tracks, 3, first='b') == [tracks[1]] + candidates
    pair = [Track('upper', 'A', 1), Track('lower', 'a', 1)]
    assert shuffle_playlist(pair, 5, first='upper') == pair


def test_impossible_and_invalid_requests():
    with pytest.raises(ValueError):
        shuffle_playlist([Track('a', 'A', 1), Track('b', 'A', 1)], 0)
    tracks = [Track('a', 'A', 1), Track('b', 'A', 1), Track('c', 'B', 1)]
    with pytest.raises(ValueError):
        shuffle_playlist(tracks, 0, first='c')
    for first in ['missing', 2]:
        with pytest.raises(ValueError):
            shuffle_playlist(tracks, 0, first=first)
    for seed in [True, 1.0, '1']:
        with pytest.raises(ValueError):
            shuffle_playlist([], seed)
    with pytest.raises(ValueError):
        shuffle_playlist([], 0, first='a')
    with pytest.raises(ValueError):
        shuffle_playlist([Track('a', 'A', 0)], 0)


def test_empty_singleton_and_global_random_isolation():
    assert shuffle_playlist([], 0) == []
    track = Track('one', 'Only', 10)
    result = shuffle_playlist([track], -9, first='one')
    assert result == [track]
    saved = random.getstate()
    try:
        random.seed(123)
        before = random.getstate()
        shuffle_playlist([track, Track('two', 'Other', 2)], 7)
        assert random.getstate() == before
    finally:
        random.setstate(saved)
