"""Build a pilot data split 'all' from raw task dirs (same layout PREPARE writes). Usage: mkdata.py OUT TASKDIR..."""
import json
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1]) / "all"
shutil.rmtree(out, ignore_errors=True)
(out / "public").mkdir(parents=True)
(out / "private" / "tests").mkdir(parents=True)
with open(out / "public" / "tasks.jsonl", "w") as fp, open(out / "private" / "tasks.jsonl", "w") as fq:
    for td in sorted(Path(p) for p in sys.argv[2:]):
        t = json.loads((td / "task.json").read_text())
        t["starter"] = {p.relative_to(td / "starter").as_posix(): p.read_text() for p in sorted((td / "starter").rglob("*")) if p.is_file()}
        fp.write(json.dumps({k: t[k] for k in ("id", "title", "difficulty", "instructions", "starter")}) + "\n")
        fq.write(json.dumps({k: t[k] for k in ("id", "difficulty", "sources", "expected")}) + "\n")
        shutil.copytree(td / "tests", out / "private" / "tests" / t["id"])
print(len(sys.argv) - 2, "tasks")
