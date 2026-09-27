import math
from expr_tokens import tokenize

class _Parser:

    def __init__(self, tokens, variables):
        raise NotImplementedError()

    def kind(self):
        raise NotImplementedError()

    def expression(self):
        raise NotImplementedError()

    def term(self):
        raise NotImplementedError()

    def unary(self):
        raise NotImplementedError()

    def primary(self):
        raise NotImplementedError()

def evaluate(text, variables=None):
    raise NotImplementedError()
