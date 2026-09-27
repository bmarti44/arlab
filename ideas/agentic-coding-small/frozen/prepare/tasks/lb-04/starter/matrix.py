class Matrix:
    def __init__(self, rows):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def to_list(self):
        raise NotImplementedError

    def add(self, other):
        raise NotImplementedError

    def multiply(self, other):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError
