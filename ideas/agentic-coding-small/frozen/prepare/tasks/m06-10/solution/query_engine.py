import operator
from query_parser import parse_query

COMPARISONS = {
    '=': operator.eq, '!=': operator.ne,
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}


def _matches(row, conditions):
    for field, op, value in conditions:
        if field not in row or type(row[field]) is not type(value):
            return False
        if not COMPARISONS[op](row[field], value):
            return False
    return True


def run_query(rows, text):
    """Filter, sort complete rows, and finally project into fresh dicts."""
    query = parse_query(text)
    selected = [row for row in rows if _matches(row, query['where'])]
    if query['order'] is not None:
        field, direction = query['order']
        present = [row for row in selected if field in row]
        missing = [row for row in selected if field not in row]
        present.sort(key=lambda row: row[field], reverse=direction == 'desc')
        selected = present + missing
    fields = query['select']
    if fields is None:
        return [dict(row) for row in selected]
    return [{field: row.get(field) for field in fields} for row in selected]
