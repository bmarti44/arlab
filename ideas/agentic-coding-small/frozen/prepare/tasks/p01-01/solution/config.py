import re

class Config:
    def __init__(self):
        self._sections = {}

    def get(self, section, key, default=None):
        return self._sections.get(section, {}).get(key, default)

    def get_int(self, section, key, default=None):
        missing = object()
        value = self.get(section, key, missing)
        if value is missing:
            return default
        if re.fullmatch(r'[+-]?[0-9]+', value) is None:
            raise ValueError('invalid integer')
        return int(value, 10)

    def get_bool(self, section, key, default=None):
        missing = object()
        value = self.get(section, key, missing)
        if value is missing:
            return default
        value = value.lower()
        if value in ('true', 'yes', 'on', '1'):
            return True
        if value in ('false', 'no', 'off', '0'):
            return False
        raise ValueError('invalid boolean')

def parse(text):
    result = Config()
    section = ''
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', ';')):
            continue
        if line.startswith('['):
            if not line.endswith(']'):
                raise ValueError('invalid section')
            name = line[1:-1].strip()
            if not name or '[' in name or ']' in name:
                raise ValueError('invalid section')
            section = name
            result._sections.setdefault(section, {})
        else:
            if '=' not in line:
                raise ValueError('invalid assignment')
            key, value = (part.strip() for part in line.split('=', 1))
            if not key:
                raise ValueError('empty key')
            result._sections.setdefault(section, {})[key] = value
    return result
