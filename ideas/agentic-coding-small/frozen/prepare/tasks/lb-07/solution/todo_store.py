"""In-memory todo records with stable IDs and defensive copies."""


def _copy(record):
    return {"id": record["id"], "title": record["title"],
            "done": record["done"], "tags": record["tags"][:]}


class TodoStore:
    def __init__(self):
        self._records = {}
        self._next_id = 1

    def add(self, title, tags=()):
        """Validate the complete record before allocating an ID."""
        title = title.strip()
        if not title:
            raise ValueError("title cannot be empty")
        normalized = set()
        for tag in tags:
            tag = tag.strip().casefold()
            if not tag:
                raise ValueError("tag cannot be empty")
            normalized.add(tag)
        task_id = self._next_id
        self._records[task_id] = {
            "id": task_id,
            "title": title,
            "done": False,
            "tags": sorted(normalized),
        }
        self._next_id += 1
        return task_id

    def get(self, task_id):
        """Fetch a detached record, raising KeyError if absent."""
        return _copy(self._records[task_id])

    def set_done(self, task_id, done):
        """Update only the completion flag of an existing record."""
        if type(done) is not bool:
            raise ValueError("done must be a bool")
        self._records[task_id]["done"] = done
        return None

    def remove(self, task_id):
        """Remove a record, reporting whether the ID existed."""
        if task_id not in self._records:
            return False
        del self._records[task_id]
        return True

    def items(self):
        """Return detached records in ascending ID order."""
        return [_copy(self._records[key]) for key in sorted(self._records)]
