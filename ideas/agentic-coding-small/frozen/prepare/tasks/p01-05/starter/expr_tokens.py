import re
_NUMBER = re.compile('(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+)')
_IDENT = re.compile('[A-Za-z_][A-Za-z0-9_]*')

def tokenize(text):
    raise NotImplementedError()
