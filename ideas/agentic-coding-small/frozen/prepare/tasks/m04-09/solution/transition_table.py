import re


def parse_table(text):
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append([field.strip() for field in line.split(',')])
    if not rows:
        raise ValueError('empty transition table')
    header = rows[0]
    if header[0] != 'state' or len(header) < 2:
        raise ValueError('invalid table header')
    symbols = tuple(header[1:])
    if len(set(symbols)) != len(symbols):
        raise ValueError('duplicate alphabet symbol')
    if any(re.fullmatch(r'[a-z]', symbol) is None for symbol in symbols):
        raise ValueError('invalid alphabet symbol')
    transitions = {}
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError('wrong number of columns')
        state = row[0]
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', state) is None:
            raise ValueError('invalid state name')
        if state in transitions:
            raise ValueError('duplicate state')
        transitions[state] = {
            symbol: destination
            for symbol, destination in zip(symbols, row[1:])
            if destination != '-'
        }
    if not transitions:
        raise ValueError('no states')
    for mapping in transitions.values():
        for destination in mapping.values():
            if destination not in transitions:
                raise ValueError('unknown destination')
    return symbols, transitions
