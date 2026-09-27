class TextBuffer:
    """An editable string with snapshot-based undo and redo."""

    def __init__(self, text=""):
        self._text = text
        self._undo = []
        self._redo = []

    @property
    def text(self):
        """Return the current text."""
        return self._text

    def insert(self, position, text):
        """Insert text, recording only edits that change the buffer."""
        if not 0 <= position <= len(self._text):
            raise IndexError("position outside buffer")
        if not text:
            return None
        self._undo.append(self._text)
        self._redo.clear()
        self._text = self._text[:position] + text + self._text[position:]
        return None

    def delete(self, start, end):
        """Delete the half-open interval [start, end)."""
        if not 0 <= start <= end <= len(self._text):
            raise IndexError("invalid interval")
        if start == end:
            return ""
        removed = self._text[start:end]
        self._undo.append(self._text)
        self._redo.clear()
        self._text = self._text[:start] + self._text[end:]
        return removed

    def undo(self):
        """Restore the previous snapshot if one is available."""
        if not self._undo:
            return False
        self._redo.append(self._text)
        self._text = self._undo.pop()
        return True

    def redo(self):
        """Reapply one undone edit if one is available."""
        if not self._redo:
            return False
        self._undo.append(self._text)
        self._text = self._redo.pop()
        return True
