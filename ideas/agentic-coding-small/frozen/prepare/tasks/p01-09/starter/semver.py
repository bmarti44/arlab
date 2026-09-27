from dataclasses import dataclass
from functools import total_ordering
import re
_CORE = '(?:0|[1-9][0-9]*)'
_PATTERN = re.compile('(' + _CORE + ')\\.(' + _CORE + ')\\.(' + _CORE + ')(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?(?:\\+([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?')

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
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def _key(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __lt__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()
