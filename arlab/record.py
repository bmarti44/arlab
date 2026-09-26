"""record.json (atomic), results.tsv regeneration, constraints.md (PLAN §3.3, §3.6)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

COLUMNS = ["id", "commit", "status", "primary", "delta", "se", "hypothesis_tag", "model", "propose_s", "run_s", "eval_s",
           "wait_s", "peak_mem_gb", "gpu_temp_max", "description"]
UNCOUNTED = ("interrupted", "infra_error")  # not experiments: no limits, no streaks
CONSTRAINT_RES = [
    (re.compile(r"(CUDA out of memory.{0,160})"), "CUDA OOM: {}"),
    (re.compile(r"(no kernel image is available.{0,120}|not (?:currently )?(?:implemented|supported) (?:for|on) .{0,120})", re.I), "unsupported op/kernel: {}"),
    (re.compile(r"(nan (?:loss )?at step \d+|loss(?: is)? nan.{0,60})", re.I), "NaN: {}"),
]


def write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_records(campaign: Path) -> list[dict]:
    """All experiment records ordered by id (the source of truth)."""
    recs = [read_json(p) for p in sorted((Path(campaign) / "runs").glob("*/record.json"))]
    return sorted((r for r in recs if r), key=lambda r: r["id"])


def counted(records: list[dict]) -> list[dict]:
    return [r for r in records if r["status"] not in UNCOUNTED]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("\t", " ").replace("\n", " ")


def write_results_tsv(campaign: Path, records: list[dict]) -> None:
    lines = ["\t".join(COLUMNS)]
    for r in records:
        t = r.get("timings", {})
        row = {**r, **t, "commit": (r.get("commit") or "")[:10]}
        lines.append("\t".join(_fmt(row.get(c)) for c in COLUMNS))
    tmp = Path(campaign) / "results.tsv.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, Path(campaign) / "results.tsv")


def add_constraints(campaign: Path, run_id: str, log_text: str, learned: str | None) -> None:
    """Terse, deduplicated machine facts, each with its run id; at most 50 lines (newest kept)."""
    path = Path(campaign) / "constraints.md"
    lines = path.read_text().splitlines() if path.exists() else []
    facts = {ln.split("] ", 1)[-1] for ln in lines}
    new = []
    for rx, fmt in CONSTRAINT_RES:
        m = rx.search(log_text or "")
        if m:
            new.append(fmt.format(" ".join(m.group(1).split())[:200]))
    if learned:
        new.append(" ".join(str(learned).split())[:200])
    for fact in new:
        if fact not in facts:
            lines.append(f"- [{run_id}] {fact}")
            facts.add(fact)
    path.write_text("\n".join(lines[-50:]) + ("\n" if lines else ""))
