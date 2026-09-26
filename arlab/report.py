"""report.md (PLAN §3.9) and the numbers `arlab status` shares with it."""
from __future__ import annotations

from pathlib import Path
from statistics import median

from .record import counted, load_records, write_results_tsv

SPEED_TARGET = 8.0


def speed(records: list[dict]) -> dict:
    c = [r for r in counted(records) if r.get("timings")]
    t = {k: sum(r["timings"].get(k, 0) or 0 for r in c) for k in ("propose_s", "run_s", "eval_s", "wait_s")}
    active = t["propose_s"] + t["run_s"] + t["eval_s"]
    per = [sum(r["timings"].get(k, 0) or 0 for k in ("propose_s", "run_s", "eval_s")) / 60 for r in c]
    return {**t, "active_h": active / 3600, "n": len(c), "per_hour": len(c) / (active / 3600) if active else 0.0,
            "median_min": median(per) if per else 0.0, "agent_share": t["propose_s"] / active if active else 0.0}


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def write_report(c) -> Path:
    st, p = c.state, c.pack
    records = load_records(c.dir)
    write_results_tsv(c.dir, records)
    cnt = counted(records)
    sp = speed(records)
    L = [f"# arlab report — {c.name} / {c.tag}", ""]
    v = st.get("verdict")
    L += [f"## Verdict: **{v or 'none'}**", "", f"Rule: {st.get('verdict_reason', '—')}", ""]
    if v == "not_found_at_this_scale":
        L += [f"The search found no effect ≥ MES ({p.metric.mes}) in {len(cnt)} experiments. This is not proof that no effect exists.", ""]
    h = st.get("holdout")
    if h:
        L += [f"Holdout (incumbent {st.get('incumbent')} vs {p.verdict.compare_to}; d > 0 = incumbent better): d = {fmt(h['d'])} ± {fmt(2 * h['se'])} (2·SE)"
              + (f"; per seed {[round(x, 5) for x in h['per_seed']]}" if h.get("per_seed") else "") + (f"; {h['skipped']}" if h.get("skipped") else ""), ""]
    L += [f"Stop reason: {st.get('stop_reason', '—')}", "", "## Calibration and power", "",
          f"- metric: {p.metric.name} ({p.metric.direction}), MES = {p.metric.mes} (fixed before calibration)",
          f"- baseline per calibration seed: {st.get('calibration')}",
          f"- sigma = {fmt(st.get('sigma'))}; validation items = {st.get('n_validation_items')}; holdout items = {st.get('n_holdout_items')}",
          f"- expected holdout SE = {fmt(st.get('expected_holdout_se'))}; power check (2·SE ≤ MES): "
          + ("**underpowered**" if st.get("underpowered") else "ok"),
          f"- determinism: {'seed-deterministic (sigma = 0)' if st.get('sigma') == 0 else 'seed-dependent'}; confirm seeds {p.seeds.confirm}",
          f"- references (screen seed): {st.get('references') or 'none'}", ""]
    if st.get("underpowered") and not p.acceptance.allow_underpowered:
        L += ["The holdout was not spent. To fix: enlarge the eval set (more items) or the budget / seeds so that "
              "2 × expected SE ≤ MES, and run under a new tag. MES is not changed.", ""]
    keeps = [r for r in records if r["status"] == "keep"]
    base = (st.get("calibration") or {}).get(str(p.seeds.screen))
    if cnt and base is not None:
        inc = keeps[-1]["primary"] if keeps else base
        L += [f"Validation (screen seed {p.seeds.screen}): baseline {fmt(base, 6)} → incumbent {fmt(inc, 6)}", ""]
    L += ["## Kept changes", ""]
    for r in keeps:
        L += [f"### {r['id']} — {r['description']} (`{r['hypothesis_tag']}`)", "",
              f"screen d = {fmt(r['screen']['d'])} ± {fmt(2 * r['screen']['se'])}"
              + (f"; confirm d = {fmt(r['confirm']['d'])} ± {fmt(2 * r['confirm']['se'])}, per seed {[round(x, 5) for x in r['confirm']['per_seed']]}"
                 if r.get("confirm") else ""), ""]
        diff = (c.dir / "runs" / r["id"] / "diff.patch")
        if diff.exists():
            L += ["```diff", diff.read_text()[:6000], "```", ""]
    if not keeps:
        L += ["(none)", ""]
    L += ["## Other experiments by hypothesis tag", ""]
    tags: dict[str, list] = {}
    for r in cnt:
        if r["status"] != "keep":
            tags.setdefault(r.get("hypothesis_tag") or "?", []).append(r)
    for t, rs in sorted(tags.items()):
        L.append(f"- **{t}** ({len(rs)}): " + "; ".join(f"{r['id']} {r['status']} ({r.get('description', '')[:60]})" for r in rs))
    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    L += ["", f"Status counts: {status_counts}", "", "## Constraints learned", ""]
    cons = c.dir / "constraints.md"
    L += [cons.read_text() if cons.exists() and cons.read_text().strip() else "(none)", ""]
    L += ["## Time and speed", "",
          f"- experiments: {sp['n']}; active hours: {sp['active_h']:.2f}; experiments/hour (active): {sp['per_hour']:.2f}"
          + (" — **speed target missed** (< 8/hour)" if sp["n"] and sp["per_hour"] < SPEED_TARGET else ""),
          f"- median minutes per experiment: {sp['median_min']:.2f}; agent share of active time: {sp['agent_share']:.0%}",
          f"- propose {sp['propose_s'] / 60:.1f} min, run {sp['run_s'] / 60:.1f} min, eval {sp['eval_s'] / 60:.1f} min, "
          f"GPU/memory wait {sp['wait_s'] / 60:.1f} min (excluded)", ""]
    tok = {k: sum(r.get("tokens", {}).get(k, 0) for r in records) for k in ("input", "cached_input", "output")}
    L += ["## Agent", "", f"- calls: {sum(r.get('agent_calls', 0) for r in records)}; tokens: {tok}",
          f"- models: {sorted({r.get('model') for r in records if r.get('model')})}", "",
          "## Environment", ""]
    for k in ("seal_hash", "data_hash", "data_dir", "image", "image_id", "uv_lock_hash", "codex_version", "arlab_commit", "baseline_commit"):
        L.append(f"- {k}: `{st.get(k)}`")
    L.append(f"- incumbent commit: `{keeps[-1]['commit'] if keeps else st.get('baseline_commit')}`")
    path = c.dir / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path
