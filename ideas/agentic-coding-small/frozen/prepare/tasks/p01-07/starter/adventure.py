from world import make_world

class Game:

    def __init__(self):
        raise NotImplementedError()

    def look(self):
        raise NotImplementedError()

    def command(self, text):
        raise NotImplementedError()

    def snapshot(self):
        raise NotImplementedError()

    @classmethod
    def from_snapshot(cls, data):
        raise NotImplementedError()
