import re
TOKEN = re.compile('\'[^\'\\r\\n]*\'|\\"[^\\"\\r\\n]*\\"|[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+|!=|<=|>=|[=<>,]')
RESERVED = {'where', 'select', 'order', 'and', 'asc', 'desc'}
OPERATORS = {'=', '!=', '<', '<=', '>', '>='}

def _field(token):
    raise NotImplementedError()

def parse_query(text):
    """Tokenize completely, then consume clauses in their fixed order."""
    raise NotImplementedError()
