import re


def parse_heading(line: str) -> tuple[int, str] | None:
    """Parse one ATX heading without applying document fence state."""
    match = re.fullmatch(r" {0,3}(#{1,6})(?:[ \t]+(.*))?", line)
    if match is None:
        return None
    raw = match.group(2) or ""
    raw = re.sub(r"\s+#+\s*$", "", " " + raw)
    text = raw.strip()
    return len(match.group(1)), text


def _fence(line):
    """Return the marker's character, length, and remaining text."""
    match = re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker), match.group(2)


class Outline:
    """Accumulate headings, retaining counters across documents."""

    def __init__(self):
        self._counters = [0] * 6
        self._entries = []

    def add(self, markdown: str) -> None:
        """Each input is a document fragment with independent fence state."""
        active = None
        for line in markdown.splitlines():
            fence = _fence(line)
            if active is not None:
                if fence is not None:
                    char, length, tail = fence
                    same_char = char == active[0]
                    long_enough = length >= active[1]
                    if same_char and long_enough and not tail.strip():
                        active = None
                continue
            if fence is not None:
                active = fence[:2]
                continue
            heading = parse_heading(line)
            if heading is None:
                continue
            level, text = heading
            index = level - 1
            self._counters[index] += 1
            for deeper in range(level, 6):
                self._counters[deeper] = 0
            number = ".".join(str(x) for x in self._counters[:level])
            self._entries.append((level, number, text))

    def entries(self) -> list[tuple[int, str, str]]:
        """Tuples are immutable; copying the outer list isolates the result."""
        return list(self._entries)

    def render(self) -> str:
        lines = []
        for level, number, text in self._entries:
            lines.append(number + " " + text)
        return "\n".join(lines)

    def reset(self) -> None:
        self._entries.clear()
        self._counters = [0] * 6
