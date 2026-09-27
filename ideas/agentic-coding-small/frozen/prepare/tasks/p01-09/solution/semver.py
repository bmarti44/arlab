from dataclasses import dataclass
from functools import total_ordering
import re

_CORE = r'(?:0|[1-9][0-9]*)'
_PATTERN = re.compile(r'(' + _CORE + r')\.(' + _CORE + r')\.(' + _CORE + r')(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?')

@total_ordering
@dataclass(frozen=True, eq=False)
class Version:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text):
        match = _PATTERN.fullmatch(text)
        if match is None:
            raise ValueError('invalid version')
        major, minor, patch, pre, build = match.groups()
        prerelease = tuple(pre.split('.')) if pre else ()
        if any(v.isdigit() and len(v) > 1 and v.startswith('0') for v in prerelease):
            raise ValueError('leading zero in prerelease')
        return cls(int(major), int(minor), int(patch), prerelease, tuple(build.split('.')) if build else ())

    def __str__(self):
        result = f'{self.major}.{self.minor}.{self.patch}'
        if self.prerelease:
            result += '-' + '.'.join(self.prerelease)
        if self.build:
            result += '+' + '.'.join(self.build)
        return result

    def _key(self):
        pre = tuple((0, int(v)) if v.isdigit() else (1, v) for v in self.prerelease)
        return (self.major, self.minor, self.patch, not self.prerelease, pre)

    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self):
        return hash(self._key())
