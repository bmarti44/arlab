class DeadlineQueue:
    def __init__(self):
        self._tasks = {}

    def add(self, task_id, priority, deadline=None):
        raise NotImplementedError

    def cancel(self, task_id):
        raise NotImplementedError

    def pending(self):
        raise NotImplementedError

    def pop(self, now):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
