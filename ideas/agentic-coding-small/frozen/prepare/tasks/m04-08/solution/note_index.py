from tag_query import normalize_tag, parse_query


class NoteIndex:
    """Store note records alongside an inverted tag membership index."""

    def __init__(self):
        self._notes = {}
        self._tags = {}

    def _unlink(self, note_id, tags):
        for tag in tags:
            members = self._tags[tag]
            members.remove(note_id)
            if not members:
                del self._tags[tag]

    def upsert(self, note_id, text, tags):
        """Normalize all tags before changing either side of the index."""
        normalized = frozenset(normalize_tag(tag) for tag in tags)
        if note_id in self._notes:
            old_text, old_tags = self._notes[note_id]
            self._unlink(note_id, old_tags)
        self._notes[note_id] = (text, normalized)
        for tag in normalized:
            self._tags.setdefault(tag, set()).add(note_id)

    def remove(self, note_id):
        text, tags = self._notes[note_id]
        self._unlink(note_id, tags)
        del self._notes[note_id]

    def search(self, query):
        """Intersect OR groups, then subtract excluded memberships."""
        groups, excluded = parse_query(query)
        candidates = set(self._notes)
        for group in groups:
            matching = set()
            for tag in group:
                matching.update(self._tags.get(tag, ()))
            candidates.intersection_update(matching)
        for tag in excluded:
            candidates.difference_update(self._tags.get(tag, ()))
        result = []
        for note_id in sorted(candidates):
            text, tags = self._notes[note_id]
            result.append((note_id, text, tuple(sorted(tags))))
        return result

    def tag_counts(self):
        """Return counts detached from the mutable membership sets."""
        counts = {}
        for tag, members in self._tags.items():
            counts[tag] = len(members)
        return counts
