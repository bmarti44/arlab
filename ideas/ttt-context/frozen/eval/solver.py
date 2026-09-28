"""Independent text solver (tests only; EVALUATE-side so RUN never sees it): answers a question by parsing the
rendered document text line by line. The tests check solve(decoded document tokens, decoded question) == gold for
every item, i.e. the gold answer follows from the document the model is given (and from nothing else)."""
import re

_T = r"^\d\d:\d\d "
LINE = {
    "move": re.compile(_T + r"Crate (\S+) was moved to dock (\w+)\.$"),
    "assign": re.compile(_T + r"(\w+ \w+) was assigned to team (\w+)\.$"),
    "room": re.compile(_T + r"Team (\w+) moved into room (\d+)\.$"),
    "approve": re.compile(_T + r"(\w+ \w+) approved shipment (\S+)\.$"),
    "code": re.compile(_T + r"The code for locker (\w+) was set to (\d+)\.$"),
}
QUESTION = {
    "state": re.compile(r"At the end of the log, at which dock is crate (\S+)\?"),
    "hop2": re.compile(r"At the end of the log, which room does the team of (\w+ \w+) occupy\?"),
    "count": re.compile(r"How many shipments did (\w+ \w+) approve in total\?"),
    "kv": re.compile(r"What is the latest code for locker (\w+)\?"),
}


def _facts(text: str, kind: str) -> list[tuple[str, str]]:
    rx = LINE[kind]
    return [m.groups() for m in (rx.match(line) for line in text.split("\n")) if m]


def solve(doc_text: str, question: str) -> str | None:
    for kind, rx in QUESTION.items():
        m = rx.search(question)
        if not m:
            continue
        x = m.group(1)
        if kind == "state":
            v = [d for c, d in _facts(doc_text, "move") if c == x]
            return v[-1] if v else None
        if kind == "hop2":
            teams = [t for p, t in _facts(doc_text, "assign") if p == x]
            rooms = [r for t, r in _facts(doc_text, "room") if teams and t == teams[-1]]
            return rooms[-1] if rooms else None
        if kind == "count":
            return str(sum(1 for p, _ in _facts(doc_text, "approve") if p == x))
        v = [c for loc, c in _facts(doc_text, "code") if loc == x]
        return v[-1] if v else None
    return None
