class TodoStore:
    def __init__(self):
        self._records = {}

    def add(self, title, tags=()):
        raise NotImplementedError

    def get(self, task_id):
        raise NotImplementedError

    def set_done(self, task_id, done):
        raise NotImplementedError

    def remove(self, task_id):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError
