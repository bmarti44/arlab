import math
from expr_tokens import tokenize

class _Parser:
    def __init__(self, tokens, variables):
        self.tokens = tokens
        self.variables = variables
        self.pos = 0

    def kind(self):
        return self.tokens[self.pos][0]

    def expression(self):
        value = self.term()
        while self.kind() in ('+', '-'):
            op = self.kind()
            self.pos += 1
            right = self.term()
            value = value + right if op == '+' else value - right
        return value

    def term(self):
        value = self.unary()
        while self.kind() in ('*', '/'):
            op = self.kind()
            self.pos += 1
            right = self.unary()
            value = value * right if op == '*' else value / right
        return value

    def unary(self):
        if self.kind() in ('+', '-'):
            sign = self.kind()
            self.pos += 1
            value = self.unary()
            return value if sign == '+' else -value
        return self.primary()

    def primary(self):
        kind, text = self.tokens[self.pos]
        if kind == 'NUMBER':
            self.pos += 1
            return float(text)
        if kind == 'IDENT':
            self.pos += 1
            if text not in self.variables:
                raise KeyError(text)
            value = self.variables[text]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('invalid variable')
            return float(value)
        if kind == '(':
            self.pos += 1
            value = self.expression()
            if self.kind() != ')':
                raise ValueError('missing closing parenthesis')
            self.pos += 1
            return value
        raise ValueError('expected operand')

def evaluate(text, variables=None):
    parser = _Parser(tokenize(text), {} if variables is None else variables)
    result = parser.expression()
    if parser.kind() != 'EOF':
        raise ValueError('trailing tokens')
    return float(result)
