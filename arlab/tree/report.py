"""report.md for a tree run: verdict, the pre-registered choice, the tree, cost."""
from __future__ import annotations

from pathlib import Path

from ..record import write_results_tsv
from ..report import fmt
from .replay import BETAS, objective


def online_summary(tree) -> dict:
    """The online analogue of a replay result (same objective, same units)."""
    nodes = sorted([n for n in tree.nodes if n["id"] != "root" and n["status"] != "pending"], key=lambda n: n["order"])
    best, curve = 0.0, []
    for n in nodes:
        if n["status"] == "ok" and n.get("score") is not None:
            best = max(best, n["score"])
        curve.append(round(best, 6))
    N = len(nodes)
    return {"N": N, "best": best, "curve": curve, "V": {str(b): objective(best, N, tree.meta["budget"], b) for b in BETAS}}


def write_tree_report(c) -> Path:
    st, p, t = c.state, c.pack, c.tree
    recs = c.records()
    write_results_tsv(c.dir, recs)
    m, on = t.meta, online_summary(t)
    L = [f"# arlab tree report — {c.name} / {c.tag}", "", f"## Verdict: **{st.get('verdict') or 'none'}**", "",
         f"Rule: {st.get('verdict_reason', '—')}", ""]
    if st.get("verdict") == "not_found_at_this_scale":
        L += [f"No effect ≥ MES ({p.metric.mes}) was found in {on['N']} tree nodes. This is not proof that no effect exists.", ""]
    h = st.get("holdout")
    root = f"the root ({m['root_from']})" if m.get("root_from") else "the baseline"
    cmp_ = root if p.verdict.compare_to == "baseline" else f"reference {p.verdict.compare_to}"
    if h:
        L += [f"Holdout (chosen node {st.get('incumbent')} vs {cmp_}; d > 0 = better): d = {fmt(h['d'])} ± {fmt(2 * h['se'])} (2·SE)"
              + (f"; per seed {[round(x, 5) for x in h['per_seed']]}" if h.get("per_seed") else "") + (f"; {h['skipped']}" if h.get("skipped") else ""), ""]
    ch = st.get("tree_choice") or {}
    L += ["## Pre-registered choice", "", f"Rule: {ch.get('rule', '—')}", "", "| node | confirm d | se | per seed / note |", "|---|---|---|---|"]
    L += [f"| {x['node']} | {fmt(x.get('d'))} | {fmt(x.get('se'))} | {x.get('per_seed') or x.get('why', '')} |" for x in ch.get("candidates", [])]
    L += ["", f"Chosen: **{ch.get('node') or 'none'}**", ""]
    if ch.get("node"):
        n = t.get(ch["node"])
        chain = " → ".join(x["id"] for x in t.path_to(n["id"]))
        L += [f"Chain: {chain}", "", f"{n.get('description', '')} (`{n.get('tag')}`)", ""]
        diff = c.git("diff", st["baseline_commit"], n["commit"])
        L += ["```diff", diff[:8000], "```", ""]
    L += ["## Search", "", f"- policy: `{m['policy']}` (sha {m['policy_sha']}); workers W = {m['W']}; budget {m['budget']} nodes; "
          f"batches {m.get('batches', 0)}; stop: {st.get('stop_reason', '—')}",
          f"- root: {root}; calibration {'copied from ' + st['calib_copied_from'] if st.get('calib_copied_from') else 'own'}",
          f"- online objective V(β) = max(0, best score) − β·N/budget: {on['V']} (best score {fmt(on['best'])} MES, N {on['N']})",
          f"- anytime best score after each node: {on['curve']}", ""]
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    L += [f"Status counts: {counts}", "", "## Best nodes (screen seed)", "", "| id | parent | depth | primary | score | tag | description |",
          "|---|---|---|---|---|---|---|"]
    best = sorted([n for n in t.nodes if n.get("score") is not None and n["id"] != "root"], key=lambda n: -n["score"])[:10]
    L += [f"| {n['id']} | {n['parent']} | {n['depth']} | {fmt(n.get('primary'))} | {fmt(n['score'])} | {n.get('tag')} | {n.get('description', '')[:80]} |"
          for n in best]
    L += ["", "## Tree", "", "```"]

    def walk(nid, ind):
        kids = sorted([x for x in t.nodes if x["parent"] == nid], key=lambda x: x["order"])
        for k in kids:
            L.append(f"{'  ' * ind}{k['id']} {k['status']:<10} {fmt(k.get('score')):>8}  {k.get('description', '')[:70]}")
            walk(k["id"], ind + 1)
    walk("root", 0)
    L += ["```", "", "## Calibration and cost", "",
          f"- metric {p.metric.name} ({p.metric.direction}), MES {p.metric.mes}; baseline {st.get('calibration')}; sigma {fmt(st.get('sigma'))}; "
          f"expected holdout SE {fmt(st.get('expected_holdout_se'))}{' (**underpowered**)' if st.get('underpowered') else ''}",
          f"- agent calls {sum(r.get('agent_calls', 0) for r in recs)}; tokens "
          f"{ {k: sum(r.get('tokens', {}).get(k, 0) for r in recs) for k in ('input', 'cached_input', 'output')} }",
          f"- active hours (sum over nodes): {sum((r.get('timings') or {}).get(k, 0) for r in recs for k in ('propose_s', 'run_s', 'eval_s')) / 3600:.2f}", ""]
    for k in ("seal_hash", "data_hash", "image_id", "codex_version", "arlab_commit", "baseline_commit"):
        L.append(f"- {k}: `{st.get(k)}`")
    path = c.dir / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path
