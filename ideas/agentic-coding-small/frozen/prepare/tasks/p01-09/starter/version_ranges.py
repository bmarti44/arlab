import re
from semver import Version
_NUM = '(?:0|[1-9][0-9]*)'

class VersionRange:

    def __init__(self, spec):
        raise NotImplementedError()

    def _parse_token(self, token):
        raise NotImplementedError()

    def contains(self, version):
        raise NotImplementedError()
