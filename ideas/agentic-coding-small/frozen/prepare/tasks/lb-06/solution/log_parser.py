"""Read timestamp | level | message records from UTF-8 logs."""

import re
from datetime import datetime

LEVELS = ("INFO", "WARN", "ERROR")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")


def parse_line(line):
    """Parse a record; blank lines and full-line comments return None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    fields = line.split("|", 2)
    if len(fields) != 3:
        raise ValueError("expected timestamp | level | message")
    timestamp, level, message = (field.strip() for field in fields)
    if _TIMESTAMP.fullmatch(timestamp) is None:
        raise ValueError("invalid timestamp format")
    datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    if level not in LEVELS:
        raise ValueError("unknown level")
    if not message:
        raise ValueError("empty message")
    return {"timestamp": timestamp, "level": level, "message": message}


def parse_file(path):
    """Return valid records and physical line numbers of rejected records."""
    records = []
    invalid = []
    with open(path, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                record = parse_line(line)
            except ValueError:
                invalid.append(number)
                continue
            if record is not None:
                records.append(record)
    return records, invalid
