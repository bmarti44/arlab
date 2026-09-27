import csv
import io
import re

INTEGER = re.compile(r"[+-]?[0-9]+")
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _kind(value):
    """Classify one nonblank cell using the documented lexical grammar."""
    if value.lower() in {"true", "false"}:
        return "bool"
    if INTEGER.fullmatch(value):
        return "int"
    if DECIMAL.fullmatch(value):
        return "float"
    return "str"


def sniff_column(values):
    """Infer one label, ignoring blanks and widening mixed numeric cells."""
    kinds = set()
    for value in values:
        value = value.strip()
        kinds.add(_kind(value))
    if not kinds:
        return "unknown"
    if kinds == {"bool"}:
        return "bool"
    if kinds == {"int"}:
        return "int"
    if kinds == {"float"}:
        return "float"
    return "str"


def sniff_csv(text):
    """Infer columns from a CSV string while validating its table shape."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return {}
    if not header:
        raise ValueError("empty header")
    if any(not name.strip() for name in header):
        raise ValueError("blank column name")
    if len(set(header)) != len(header):
        raise ValueError("duplicate column name")
    columns = [[] for name in header]
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError("ragged row")
        for column, value in zip(columns, row):
            column.append(value)
    return {name: sniff_column(column)
            for name, column in zip(header, columns)}
