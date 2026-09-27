from tag_query import normalize_tag, parse_query


class NoteIndex:
    def __init__(self):
        raise NotImplementedError

    def upsert(self, note_id, text, tags):
        raise NotImplementedError

    def remove(self, note_id):
        raise NotImplementedError

    def search(self, query):
        raise NotImplementedError

    def tag_counts(self):
        raise NotImplementedError
