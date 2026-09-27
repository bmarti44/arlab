import re


def assemble(source: str) -> list[tuple]:
    labels = {}
    pending = []
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            name, line = line.split(":", 1)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("invalid label")
            if name in labels:
                raise ValueError("duplicate label")
            labels[name] = len(pending)
            line = line.strip()
        if line:
            pending.append(line.split())
    result = []
    counts = {"SET": 2, "ADD": 2, "JMP": 1, "JNZ": 2, "OUT": 1, "HALT": 0}
    for tokens in pending:
        op, *args = tokens
        if op not in counts or len(args) != counts[op]:
            raise ValueError("invalid instruction")
        if op in ("SET", "ADD", "JNZ", "OUT"):
            if args[0] not in ("a", "b", "c", "d"):
                raise ValueError("invalid register")
        if op in ("SET", "ADD"):
            if not re.fullmatch(r"[+-]?[0-9]+", args[1]):
                raise ValueError("invalid integer")
            args[1] = int(args[1])
        if op in ("JMP", "JNZ"):
            target = args[-1]
            if target not in labels:
                raise ValueError("unknown label")
            args[-1] = labels[target]
        result.append((op, *args))
    return result
