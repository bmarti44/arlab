import re

TOKEN = re.compile(r"'[^'\r\n]*'|\"[^\"\r\n]*\"|[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+|!=|<=|>=|[=<>,]")
RESERVED = {'where', 'select', 'order', 'and', 'asc', 'desc'}
OPERATORS = {'=', '!=', '<', '<=', '>', '>='}


def _field(token):
    if token in RESERVED or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token) is None:
        raise ValueError('expected field name')
    return token


def parse_query(text):
    """Tokenize completely, then consume clauses in their fixed order."""
    tokens = []
    end = 0
    for match in TOKEN.finditer(text):
        if text[end:match.start()].strip():
            raise ValueError('unknown token')
        token = match.group()
        if tokens and match.start() == end:
            separators = OPERATORS | {','}
            if token not in separators and tokens[-1] not in separators:
                raise ValueError('words must be separated by whitespace')
        tokens.append(token)
        end = match.end()
    if text[end:].strip():
        raise ValueError('unknown token')

    def take():
        if not tokens:
            raise ValueError('unexpected end of query')
        return tokens.pop(0)

    result = {'where': [], 'select': None, 'order': None}
    if tokens and tokens[0] == 'where':
        take()
        while True:
            field = _field(take())
            operator = take()
            if operator not in OPERATORS:
                raise ValueError('unknown comparison')
            raw = take()
            if raw.startswith(("'", '"')):
                value = raw[1:-1]
            elif re.fullmatch(r'-?[0-9]+', raw):
                value = int(raw)
            else:
                raise ValueError('expected integer or quoted string')
            result['where'].append((field, operator, value))
            if not tokens or tokens[0] != 'and':
                break
            take()
    if tokens and tokens[0] == 'select':
        take()
        fields = [_field(take())]
        while tokens and tokens[0] == ',':
            take()
            field = _field(take())
            if field in fields:
                raise ValueError('duplicate selected field')
            fields.append(field)
        result['select'] = fields
    if tokens and tokens[0] == 'order':
        take()
        field = _field(take())
        direction = take()
        if direction not in ('asc', 'desc'):
            raise ValueError('expected asc or desc')
        result['order'] = (field, direction)
    if tokens:
        raise ValueError('unexpected trailing tokens')
    return result
