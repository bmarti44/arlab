class TextBuffer:
    def __init__(self, text=""):
        self._text = text

    @property
    def text(self):
        return self._text

    def insert(self, position, text):
        raise NotImplementedError

    def delete(self, start, end):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError
