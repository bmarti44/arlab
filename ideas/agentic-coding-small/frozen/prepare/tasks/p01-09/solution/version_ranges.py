import re
from semver import Version

_NUM = r'(?:0|[1-9][0-9]*)'

class VersionRange:
    def __init__(self, spec):
        self._alternatives = []
        for alternative in spec.split('||'):
            tokens = alternative.split()
            if not tokens:
                raise ValueError('empty range alternative')
            constraints = []
            for token in tokens:
                constraints.extend(self._parse_token(token))
            self._alternatives.append(constraints)

    def _parse_token(self, token):
        if token == '*':
            return []
        wildcard = re.fullmatch(r'(' + _NUM + r')(?:\.(' + _NUM + r'))?\.\*', token)
        if wildcard:
            major = int(wildcard.group(1))
            minor = wildcard.group(2)
            if minor is None:
                return [('>=', Version(major, 0, 0)), ('<', Version(major + 1, 0, 0))]
            minor = int(minor)
            return [('>=', Version(major, minor, 0)), ('<', Version(major, minor + 1, 0))]
        if token.startswith(('~', '^')):
            version = Version.parse(token[1:])
            if token[0] == '~':
                upper = Version(version.major, version.minor + 1, 0)
            elif version.major:
                upper = Version(version.major + 1, 0, 0)
            elif version.minor:
                upper = Version(0, version.minor + 1, 0)
            else:
                upper = Version(0, 0, version.patch + 1)
            return [('>=', version), ('<', upper)]
        for op in ('>=', '<=', '>', '<', '='):
            if token.startswith(op):
                return [(op, Version.parse(token[len(op):]))]
        return [('=', Version.parse(token))]

    def contains(self, version):
        if isinstance(version, str):
            version = Version.parse(version)
        def matches(op, bound):
            if op == '=':
                return version == bound
            if op == '>':
                return version > bound
            if op == '>=':
                return version >= bound
            if op == '<':
                return version < bound
            return version <= bound
        return any(all(matches(op, bound) for op, bound in alternative) for alternative in self._alternatives)
