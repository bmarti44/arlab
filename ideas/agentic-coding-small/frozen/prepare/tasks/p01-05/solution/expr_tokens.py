import re

_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)')
_IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

def tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
            continue
        if char in '+-*/()':
            tokens.append((char, char))
            pos += 1
            continue
        match = _NUMBER.match(text, pos)
        kind = 'NUMBER'
        if match is None:
            match = _IDENT.match(text, pos)
            kind = 'IDENT'
        if match is None:
            raise ValueError('unexpected character')
        tokens.append((kind, match.group()))
        pos = match.end()
    tokens.append(('EOF', ''))
    return tokens
