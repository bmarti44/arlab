from dataclasses import dataclass
import re


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


HEADER = re.compile(r'@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@')


def parse_patch(patch):
    if patch == '':
        return []
    lines = patch.split('\n')
    if lines[-1] == '':
        lines.pop()
    if len(lines) < 2:
        raise ValueError('missing file headers')
    for line, prefix in zip(lines[:2], ('--- ', '+++ ')):
        if not line.startswith(prefix) or len(line) == len(prefix):
            raise ValueError('invalid file header')
    hunks = []
    position = 2
    while position < len(lines):
        match = HEADER.fullmatch(lines[position])
        if match is None:
            raise ValueError('invalid hunk header')
        old_start, old_count, new_start, new_count = match.groups()
        old_start = int(old_start)
        new_start = int(new_start)
        old_count = 1 if old_count is None else int(old_count)
        new_count = 1 if new_count is None else int(new_count)
        if (old_count and old_start == 0) or (new_count and new_start == 0):
            raise ValueError('nonempty ranges start at one')
        body = []
        position += 1
        old_seen = new_seen = 0
        while position < len(lines) and not lines[position].startswith('@@'):
            line = lines[position]
            if not line or line[0] not in ' +-':
                raise ValueError('invalid hunk body')
            prefix, content = line[0], line[1:]
            body.append((prefix, content))
            old_seen += prefix in ' -'
            new_seen += prefix in ' +'
            position += 1
        if old_seen != old_count or new_seen != new_count:
            raise ValueError('hunk counts do not match body')
        hunks.append(Hunk(old_start, old_count, new_start, new_count, tuple(body)))
    return hunks
