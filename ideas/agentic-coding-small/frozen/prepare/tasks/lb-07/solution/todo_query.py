"""Apply combined todo filters to a store snapshot."""

from todo_store import TodoStore


def select(store: TodoStore, done=None, tag=None, text="", limit=None):
    """Filter, order incomplete first, then truncate the matching records."""
    if done is not None and type(done) is not bool:
        raise ValueError("done must be bool or None")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    needle = text.strip().casefold()
    normalized_tag = None
    if tag is not None:
        normalized_tag = tag.strip().casefold()
        if not normalized_tag:
            raise ValueError("tag cannot be empty")
    matches = []
    for record in store.items():
        if done is not None and record["done"] != done:
            continue
        if normalized_tag is not None and normalized_tag not in record["tags"]:
            continue
        if needle not in record["title"].casefold():
            continue
        matches.append(record)
    matches.sort(key=lambda record: (record["done"], record["id"]))
    if limit is not None:
        matches = matches[:limit]
    return matches
