"""Summarize parsed logs and render a deterministic text report."""

from log_parser import LEVELS, parse_file


def summarize(records):
    """Aggregate an iterable without modifying its records."""
    counts = {level: 0 for level in LEVELS}
    total = 0
    earliest = None
    latest = None
    for record in records:
        total += 1
        counts[record["level"]] += 1
        timestamp = record["timestamp"]
        if earliest is None or timestamp < earliest:
            earliest = timestamp
        if latest is None or timestamp > latest:
            latest = timestamp
    return {
        "total": total,
        "counts": counts,
        "earliest": earliest,
        "latest": latest,
    }


def report_file(path):
    """Parse a file and report counts, time bounds, and invalid line numbers."""
    records, invalid = parse_file(path)
    summary = summarize(records)
    lines = [f"TOTAL {summary['total']}"]
    for level in LEVELS:
        lines.append(f"{level} {summary['counts'][level]}")
    earliest = summary["earliest"]
    latest = summary["latest"]
    lines.append(f"FIRST {earliest if earliest is not None else '-'}")
    lines.append(f"LAST {latest if latest is not None else '-'}")
    rejected = ",".join(str(number) for number in invalid)
    lines.append(f"INVALID {rejected or '-'}")
    return "\n".join(lines) + "\n"
