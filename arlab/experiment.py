"""One experiment: propose → apply → run → evaluate → decide → record (PLAN §3.3, §3.5, §3.6)."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import stats
from .agent import InfraError, Proposal
from .campaign import Campaign, debug_kill, now, tail
from .pack import compile_error, surface_files
from .record import add_constraints, counted, read_json

NOTES_MAX = 8192


# ------------------------------------------------------------------ the agent's view
def history_md(c: Campaign, records: list[dict], full: bool = False) -> str:
    st, p = c.state, c.pack
    _, inc_vals, inc_id = c.incumbent(records)
    s = p.seeds.screen
    base = c.baseline_vals()[s]["primary"]
    lines = ["# history.md (written by the runner)", "",
             f"Metric: {p.metric.name} ({p.metric.direction}). MES (judged only at the end): {p.metric.mes}.",
             f"Calibration (baseline, validation): {st.get('calibration')}; sigma={st.get('sigma', 0):.4g}.",
             f"Baseline on screen seed {s}: {base:.6g}. Incumbent ({inc_id}) on seed {s}: {inc_vals[s]['primary']:.6g}.",
             f"References (screen seed): {st.get('references') or 'none'}.", "",
             "| id | status | primary | delta | se | tag | description |", "|---|---|---|---|---|---|---|"]
    shown = records if full else records[-30:]
    for r in shown:
        f = lambda v: "" if v is None else f"{v:.4g}"
        lines.append(f"| {r['id']} | {r['status']} | {f(r.get('primary'))} | {f(r.get('delta'))} | {f(r.get('se'))} | "
                     f"{r.get('hypothesis_tag', '')} | {r.get('description', '')} |")
    tags: dict[str, dict] = {}
    for r in counted(records):
        t = tags.setdefault(r.get("hypothesis_tag") or "?", {})
        t[r["status"]] = t.get(r["status"], 0) + 1
    lines += ["", "## Hypothesis tags tried", ""] + [f"- {t}: {sum(v.values())} tried — " + ", ".join(f"{k} {n}" for k, n in sorted(v.items()))
                                                    for t, v in sorted(tags.items())]
    return "\n".join(lines) + "\n"


def build_view(c: Campaign, rdir: Path, records: list[dict], inc_commit: str, extra: str = "", full: bool = False) -> Path:
    view = rdir / "view"
    c.export(inc_commit, view)
    ctx = []
    for v in c.pack.agent.visible:
        dst = view / (v if v in ("program.md", "IDEA.md") else "frozen_run/" + v.removeprefix("frozen/run/"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.sealed / v, dst)
        os.chmod(dst, 0o444)
        ctx.append(dst.relative_to(view).as_posix())
    notes = c.dir / "notes.md"
    (view / "notes.md").write_text(notes.read_text() if notes.exists() else "")
    cons = c.dir / "constraints.md"
    (view / "constraints.md").write_text(cons.read_text() if cons.exists() else "")
    hist = history_md(c, records, full)
    (view / "history.md").write_text(hist)
    diff = c.git("diff", c.state["baseline_commit"], inc_commit)
    (view / "incumbent.diff").write_text(diff)
    program = (c.sealed / "program.md").read_text()
    surf = surface_files(c.sealed / "surface")
    prompt = [program, "", "---", "## Files in the current directory",
              f"- Editable surface (only these matter): {', '.join(surf)}",
              f"- Read-only context: {', '.join(ctx + ['history.md', 'constraints.md', 'incumbent.diff'])}",
              "- notes.md: your scratchpad, persisted across calls (newest 8 KB kept).", "", hist,
              "## constraints.md (facts about this machine)", (view / "constraints.md").read_text() or "(none yet)", "",
              "## incumbent.diff (baseline → incumbent)", "```diff", diff[:20000], "```", extra]
    (view / "prompt.md").write_text("\n".join(prompt) + "\n")
    return view


def save_notes(c: Campaign, view: Path):
    p = view / "notes.md"
    if p.exists():
        (c.dir / "notes.md").write_bytes(p.read_bytes()[-NOTES_MAX:])


def apply_view(c: Campaign, view: Path, rdir: Path) -> tuple[list[str], str | None]:
    """Copy back only surface files; py_compile changed .py. Returns (changed files, compile error)."""
    changed, ignored = [], []
    for f in surface_files(c.sealed / "surface"):
        src, dst = view / f, c.work / f
        if src.exists() and src.read_bytes() != dst.read_bytes():
            dst.write_bytes(src.read_bytes())
            changed.append(f)
    known = set(surface_files(c.sealed / "surface")) | {"notes.md", "constraints.md", "history.md", "incumbent.diff", "prompt.md",
                                                         "program.md", "IDEA.md", "crash.log"}
    for f in surface_files(view):
        if f not in known and not f.startswith("frozen_run/"):
            ignored.append(f)
    if ignored:
        (rdir / "ignored_changes.txt").write_text("\n".join(ignored) + "\n")
    errs = [e for f in changed if f.endswith(".py") and (e := compile_error(c.work / f))]
    return changed, (errs[0] if errs else None)


# ------------------------------------------------------------------ decisions
def guard_failures(c: Campaign, res: dict) -> list[str]:
    fails = []
    for g in c.pack.guards:
        v = res["metrics"].get(g.name)
        if v is None:
            fails.append(f"{g.name} missing")
            continue
        if g.max is not None and v > g.max:
            fails.append(f"{g.name}={v:.4g} > {g.max}")
        if g.min is not None and v < g.min:
            fails.append(f"{g.name}={v:.4g} < {g.min}")
        if g.max_ratio_vs_baseline is not None or g.min_ratio_vs_baseline is not None:
            b = c.baseline_metric(g.name)
            if not b:
                fails.append(f"{g.name}: no baseline value")
            elif g.max_ratio_vs_baseline is not None and v / b > g.max_ratio_vs_baseline:
                fails.append(f"{g.name}={v:.4g} is {v / b:.2f}x baseline > {g.max_ratio_vs_baseline}")
            elif g.min_ratio_vs_baseline is not None and v / b < g.min_ratio_vs_baseline:
                fails.append(f"{g.name}={v:.4g} is {v / b:.2f}x baseline < {g.min_ratio_vs_baseline}")
    return fails


def candidate_trial(c: Campaign, commit: str, seed: int, rdir: Path, eid: str, attempt: str = "") -> dict:
    """Screen/confirm run with the LOOP contended rule: re-run once, then `contended`."""
    res = None
    for k in range(2):
        tdir = rdir / f"s{seed}{attempt}{'-r2' if k else ''}"
        tick = (lambda el, cid: debug_kill(f"during_run:{eid}") if el > 0.7 and cid else None) if not attempt and not k else None
        res = c.trial(c.export(commit, tdir / "surface"), seed, "validation", tdir, tick)
        res["dir"] = tdir.name
        if res["status"] != "contended":
            return res
        c.log(f"{eid}: contended on seed {seed} ({res['reason']}); re-running once when free")
    return res


def run_experiment(c: Campaign, eid: str, records: list[dict]) -> dict:
    p = c.pack
    rdir = c.dir / "runs" / eid
    shutil.rmtree(rdir, ignore_errors=True)
    rdir.mkdir(parents=True)
    inc_commit, inc_vals, inc_id = c.incumbent(records)
    rec = {"id": eid, "parent": inc_commit, "incumbent": inc_id, "started": now(), "agent_calls": 0,
           "tokens": {"input": 0, "cached_input": 0, "output": 0}, "timings": {"propose_s": 0.0, "run_s": 0.0, "eval_s": 0.0, "wait_s": 0.0},
           "seeds": {"screen": p.seeds.screen, "confirm": p.seeds.confirm}, "results": {}, "primary": None, "delta": None, "se": None,
           "description": "", "hypothesis_tag": "", "model": "", "constraint_learned": None}
    name = f"{c.prefix}-agent-{eid}"

    def agent(call, *args) -> Proposal:
        t0 = time.monotonic()
        try:
            prop = call(*args)
        finally:
            rec["timings"]["propose_s"] += time.monotonic() - t0
        rec["agent_calls"] += 1
        for k in rec["tokens"]:
            rec["tokens"][k] += prop.tokens.get(k, 0)
        return prop

    def done(status, reason="", **kw):
        rec.update(status=status, reason=reason, finished=now(), **kw)
        for t in rdir.glob("s*"):
            c.clean_trial({}, t, keep_out=(status == "keep"))
        shutil.rmtree(rdir / "view" / "frozen_run", ignore_errors=True)
        return rec

    # plateau rescue: after rescue_after consecutive non-keeps (at most once per rescue_after experiments)
    since = 0
    for r in reversed(counted(records)):
        if r["status"] == "keep":
            break
        since += 1
        if r.get("rescue"):
            break
    if since >= p.campaign.rescue_after:
        kept = "\n".join(c.git("show", r["commit"]) for r in records if r["status"] == "keep")
        extra = ("\n# PLATEAU RESCUE\nDo NOT edit the surface this time. The last experiments did not improve the metric. "
                 "Study the full ledger above and every kept diff below, then write a strategy note of at most 30 lines into notes.md "
                 "(what has been learned, what to try next, what to stop trying). Finish with the JSON object, action \"skip\".\n\n"
                 f"## All kept diffs\n```diff\n{kept[:40000]}\n```\n")
        rview = build_view(c, rdir / "rescue", records, inc_commit, extra, full=True)
        try:
            agent(c.backend.rescue, rview, rdir, name)
            save_notes(c, rview)
            rec["rescue"] = True
        except InfraError as e:
            c.log(f"{eid}: rescue call failed ({e}); continuing without it")

    view = build_view(c, rdir, records, inc_commit)
    try:
        prop = agent(c.backend.propose, view, rdir, name)
    except InfraError as e:
        return done("infra_error", str(e)[:500])
    save_notes(c, view)
    rec.update(description=prop.description, hypothesis_tag=prop.hypothesis_tag, model=prop.model, constraint_learned=prop.constraint_learned)
    add_constraints(c.dir, eid, "", prop.constraint_learned)
    if prop.action == "skip":
        return done("skip", "agent chose skip")
    c.git("reset", "-q", "--hard", inc_commit)
    changed, err = apply_view(c, view, rdir)
    if not changed:
        return done("no_op", "edit with an empty diff")
    c.git("add", "-A")
    c.git("commit", "-q", "-m", f"{eid}: {prop.description}")
    commit = c.git("rev-parse", "HEAD")
    rec["commit"], rec["files"] = commit, changed
    (rdir / "diff.patch").write_text(c.git("diff", inc_commit, commit))

    def screen(attempt=""):
        if err_now[0]:
            return {"status": "crash", "reason": f"py_compile: {err_now[0]}", "log_tail": err_now[0], "run_s": 0, "eval_s": 0, "wait_s": 0}
        return candidate_trial(c, rec["commit"], p.seeds.screen, rdir, eid, attempt)

    def account(res):
        for k in ("run_s", "eval_s", "wait_s"):
            rec["timings"][k] += res.get(k, 0) or 0
        if res.get("peak_mem_gb") is not None:
            rec["peak_mem_gb"] = max(rec.get("peak_mem_gb") or 0, res["peak_mem_gb"])
        if res.get("gpu_temp_max") is not None:
            rec["gpu_temp_max"] = max(rec.get("gpu_temp_max") or 0, res["gpu_temp_max"])
        add_constraints(c.dir, eid, res.get("log_tail", ""), None)

    err_now = [err]
    res = screen()
    account(res)
    if res["status"] == "crash":  # one fix call, then one re-run of the screen
        try:
            fprop = agent(c.backend.fix, view, rdir, name, res.get("log_tail", ""))
        except InfraError as e:
            fprop = None
            c.log(f"{eid}: fix call failed: {e}")
        if fprop and fprop.action == "edit":
            save_notes(c, view)
            c.git("reset", "-q", "--hard", inc_commit)
            changed, err_now[0] = apply_view(c, view, rdir)
            if changed:
                c.git("add", "-A")
                c.git("commit", "-q", "-m", f"{eid}: {prop.description} (fixed)")
                rec["commit"], rec["files"], rec["fixed"] = c.git("rev-parse", "HEAD"), changed, fprop.description
                (rdir / "diff.patch").write_text(c.git("diff", inc_commit, rec["commit"]))
                res = screen("-fix")
                account(res)
    s = p.seeds.screen
    if res["status"] != "ok":
        c.git("reset", "-q", "--hard", inc_commit)
        return done(res["status"], res.get("reason", ""))
    rec["results"][str(s)] = strip(res)
    rec["primary"] = res["primary"]
    fails = guard_failures(c, res)
    if fails:
        c.git("reset", "-q", "--hard", inc_commit)
        return done("guard_fail", "; ".join(fails))
    sig, direction = c.state["sigma"], p.metric.direction
    scr = stats.compare({s: res}, {s: inc_vals[s]}, [s], direction, sig)
    rec.update(screen={"d": scr["d"], "se": scr["se"]}, delta=scr["d"], se=scr["se"])
    if not stats.screen_pass(scr, bool(p.seeds.confirm)):
        c.git("reset", "-q", "--hard", inc_commit)
        return done("discard", f"screen d={scr['d']:.4g} <= {1 if p.seeds.confirm else 2}·SE={scr['se']:.4g}")
    if p.seeds.confirm:
        conf = {}
        for seed in p.seeds.confirm:
            r2 = candidate_trial(c, rec["commit"], seed, rdir, eid, "")
            account(r2)
            if r2["status"] != "ok":
                c.git("reset", "-q", "--hard", inc_commit)
                return done(r2["status"], f"confirm seed {seed}: {r2.get('reason', '')}")
            fails = guard_failures(c, r2)
            if fails:
                c.git("reset", "-q", "--hard", inc_commit)
                return done("guard_fail", f"confirm seed {seed}: " + "; ".join(fails))
            conf[seed] = r2
            rec["results"][str(seed)] = strip(r2)
        cc = stats.compare(conf, inc_vals, p.seeds.confirm, direction, sig)
        rec.update(confirm={"d": cc["d"], "se": cc["se"], "per_seed": cc["per_seed"]}, delta=cc["d"], se=cc["se"])
        if not stats.confirm_pass(cc):
            c.git("reset", "-q", "--hard", inc_commit)
            return done("discard", f"confirm d={cc['d']:.4g} vs 2·SE={2 * cc['se']:.4g}, per-seed {[round(x, 5) for x in cc['per_seed']]}")
    c.git("tag", f"keep/{eid}", rec["commit"])
    return done("keep", "screen" + (" + confirm" if p.seeds.confirm else " (deterministic; holdout decides)") + " passed")


def strip(res: dict) -> dict:
    return {k: res.get(k) for k in ("primary", "metrics", "items", "dir")}
