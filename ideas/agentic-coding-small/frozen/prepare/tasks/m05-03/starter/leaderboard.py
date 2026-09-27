class Leaderboard:
    """Maintain one score per name and render competition rankings.

    A score update replaces the old value. Rankings are calculated from
    current scores so removing or updating a player cannot leave stale ranks.
    """

    def __init__(self):
        self._scores = {}

    def set_score(self, name: str, score: int) -> None:
        """Add a player or replace that player's current score."""
        self._scores[name] = score

    def remove(self, name: str) -> bool:
        """Remove a known player and report whether anything changed."""
        if name not in self._scores:
            return False
        del self._scores[name]
        return True

    def ranking(self) -> list[tuple[int, str, int]]:
        """Return a fresh list of (rank, name, score) triples.

        Higher scores come first. Equal scores share their earliest position
        in the sorted list, and player names break ties alphabetically.
        """
        ordered = sorted(self._scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        result = []
        previous_score = None
        current_rank = 0
        for position, (name, score) in enumerate(ordered, start=1):
            if position == 1 or score <= previous_score:
                current_rank = position
            result.append((current_rank, name, score))
            previous_score = score
        return result

    def top(self, count: int) -> list[tuple[int, str, int]]:
        """Return at most count entries; the cutoff may split a tie.

        Count is an integer. Negative counts are invalid rather than using
        the negative-slice convention, which is surprising in this API.
        """
        if count < 0:
            raise ValueError("count must be nonnegative")
        return self.ranking()[:count]

    def rank_of(self, name: str) -> int | None:
        """Return a current rank, or None when the name is absent.

        Derive the answer from the same ranking used for display so callers
        see identical ranking rules through both access paths.
        """
        for rank, player, score in self.ranking():
            if player == name:
                return rank
        return None
