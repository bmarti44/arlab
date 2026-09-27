import operator
from query_parser import parse_query
COMPARISONS = {'=': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge}

def _matches(row, conditions):
    raise NotImplementedError()

def run_query(rows, text):
    """Filter, sort complete rows, and finally project into fresh dicts."""
    raise NotImplementedError()
