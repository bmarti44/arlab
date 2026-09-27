import re


def normalize_cell(name):
    cell = name.strip().upper()
    if re.fullmatch(r'[A-Z]+[1-9][0-9]*', cell) is None:
        raise ValueError('invalid cell name')
    return cell


def parse_formula(formula):
    text = formula.strip()
    if not text.startswith('='):
        raise ValueError('formula must start with =')
    body = text[1:].strip()
    terms = []
    position = 0
    sign = 1
    if body[:1] in ('+', '-'):
        sign = -1 if body[0] == '-' else 1
        position = 1
    while True:
        match = re.match(r'\s*([A-Za-z]+[1-9][0-9]*|[0-9]+)', body[position:])
        if match is None:
            raise ValueError('expected operand')
        token = match.group(1)
        operand = int(token) if token.isascii() and token.isdigit() else token.upper()
        terms.append((sign, operand))
        position += match.end()
        while position < len(body) and body[position].isspace():
            position += 1
        if position == len(body):
            return terms
        operator = body[position]
        if operator not in '+-':
            raise ValueError('expected operator')
        sign = 1 if operator == '+' else -1
        position += 1


class Spreadsheet:
    """Store parsed terms, evaluating dependencies on demand."""

    def __init__(self):
        self._cells = {}

    def set(self, name, value):
        cell = normalize_cell(name)
        if type(value) is int:
            terms = [(1, value)]
        elif isinstance(value, str):
            terms = parse_formula(value)
        else:
            raise TypeError('expected integer or formula')
        self._cells[cell] = terms

    def get(self, name):
        cell = normalize_cell(name)
        active = set()
        memo = {}

        def evaluate(current):
            if current in active:
                raise ValueError('cyclic reference')
            if current in memo:
                return memo[current]
            terms = self._cells[current]
            active.add(current)
            total = 0
            for sign, operand in terms:
                value = evaluate(operand) if isinstance(operand, str) else operand
                total += sign * value
            active.remove(current)
            memo[current] = total
            return total

        return evaluate(cell)

    def delete(self, name):
        cell = normalize_cell(name)
        del self._cells[cell]
