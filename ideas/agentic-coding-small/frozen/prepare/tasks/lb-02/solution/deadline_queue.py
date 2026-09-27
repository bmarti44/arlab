class DeadlineQueue:
    """Tasks ordered by priority, deadline, then insertion order."""

    def __init__(self):
        self._tasks = {}
        self._sequence = 0

    def add(self, task_id, priority, deadline=None):
        """Register a unique task without comparing IDs for ordering."""
        if task_id in self._tasks:
            raise ValueError("duplicate task id")
        record = (priority, deadline, self._sequence)
        self._tasks[task_id] = record
        self._sequence += 1
        return None

    def cancel(self, task_id):
        """Remove a pending task, reporting whether it existed."""
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        return True

    def pending(self):
        """Return a fresh list in scheduling order."""
        def order(task_id):
            priority, deadline, sequence = self._tasks[task_id]
            return (priority, deadline is None,
                    0 if deadline is None else deadline, sequence)

        return sorted(self._tasks, key=order)

    def pop(self, now):
        """Discard expired tasks, then remove the first remaining task."""
        expired = []
        for task_id, (_, deadline, _) in self._tasks.items():
            if deadline is not None and deadline < now:
                expired.append(task_id)
        for task_id in expired:
            del self._tasks[task_id]
        ordered = self.pending()
        if not ordered:
            return None
        task_id = ordered[0]
        del self._tasks[task_id]
        return task_id

    def __len__(self):
        """Count pending tasks, including those not yet purged by pop."""
        return len(self._tasks)
