"""Dreaming: Codex rewrites the exploration policy offline against replayed trees (Dream-RSI §3.3).

M revisions, each seeing the earlier ones; the best by mean training-tree V is a candidate, accepted only if its
mean V on held-out trees is ≥ the current policy's (the overfitting guard the paper lacks). Weights never change.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from ..agent import CodexBackend, InfraError
from ..campaign import RUNS, now
from ..experiment import safe_read
from ..guards import FileLock
from ..record import load_records, read_json, write_json
from .model import Tree
from .policy import PolicyError, SandboxPolicy, resolve, sha
from .replay import BETA, mean_v, replay

PROMPT = Path(__file__).resolve().parent / "prompts" / "policy_dev.md"
MAX_POLICY = 200_000


def tree_root() -> Path:
    return Path(os.environ.get("ARLAB_RUNS", RUNS)) / "_tree"


def load_tree(spec: str) -> Tree:
    """'<pack>/<tag>' (a tree run or an imported greedy campaign) or a path to a tree.json."""
    p = Path(spec)
    if p.suffix == ".json" and p.exists():
        return Tree.load(p)
    runs = Path(os.environ.get("ARLAB_RUNS", RUNS))
    for cand in (runs / spec / "tree.json", tree_root() / "imported" / f"{spec.replace('/', '__')}.json"):
        if cand.exists():
            return Tree.load(cand)
    raise FileNotFoundError(f"no tree {spec} (run `arlab tree import {spec}` for a greedy campaign)")


def import_campaign(spec: str) -> Path:
    """A finished greedy campaign as a (star/chain-shaped) tree: parent = the incumbent each experiment started from."""
    runs = Path(os.environ.get("ARLAB_RUNS", RUNS))
    cdir = runs / spec
    st = read_json(cdir / "state.json", {})
    import yaml
    pk = yaml.safe_load((cdir / "sealed" / "pack.yaml").read_text())
    mes, sgn = pk["metric"]["mes"], 1.0 if pk["metric"]["direction"] == "maximize" else -1.0
    base = st["calibration"][str(pk["seeds"]["screen"])]
    t = Tree(tree_root() / "imported" / f"{spec.replace('/', '__')}.json",
             {"pack": spec.split("/")[0], "tag": spec.split("/")[1], "policy": "greedy (imported)", "policy_sha": "", "W": 1,
              "imported": now()})
    for r in load_records(cdir):
        if r["status"] in ("interrupted", "infra_error"):
            continue
        par = "root" if r.get("incumbent", "baseline") == "baseline" else f"e{r['incumbent']}"
        ok = r["status"] in ("keep", "discard") and r.get("primary") is not None
        t.nodes.append({"id": f"e{r['id']}", "parent": par, "depth": t.get(par)["depth"] + 1, "order": len(t.nodes),
                        "status": "ok" if ok else r["status"], "score": round(sgn * (r["primary"] - base) / mes, 6) if ok else None,
                        "tag": r.get("hypothesis_tag", ""), "cost_s": round(sum((r.get("timings") or {}).get(k, 0) for k in ("propose_s", "run_s", "eval_s")), 1),
                        "description": r.get("description", "")})
    t.meta["budget"] = len(t.nodes) - 1
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.save()
    return t.path


def evaluate(policy: Path, trees: list[Tree], sandbox=SandboxPolicy, horizon: int | None = None) -> list[dict]:
    out = []
    for t in trees:
        try:
            with sandbox(policy) as pol:
                rec = t.meta.get("budget") or len(t.nodes) - 1
                r = replay(pol, t.nodes, t.meta.get("W", 4), min(rec, horizon) if horizon else rec)
        except PolicyError as e:
            r = {"error": str(e)[-1500:], "V": {"0.0": -1.0, "0.5": -1.0, "1.0": -1.0}, "N": 0, "best": 0.0}
        out.append({"tree": f"{t.meta['pack']}/{t.meta['tag']}", **r})
    return out


def tree_md(t: Tree) -> str:
    L = [f"# tree {t.meta['pack']}/{t.meta['tag']} (recorded with policy {t.meta.get('policy')}, W={t.meta.get('W')})", "",
         "| id | parent | depth | order | status | score | tag | cost_s |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {n['id']} | {n['parent']} | {n['depth']} | {n['order']} | {n['status']} | {n.get('score')} | {n.get('tag', '')} | {n.get('cost_s')} |"
          for n in t.nodes]
    return "\n".join(L) + "\n"


def replay_md(res: list[dict]) -> str:
    L = ["# How the current policy did on each training tree (replay)", ""]
    for r in res:
        L += [f"## {r['tree']}", f"- V(β=0.5) = {r['V']['0.5']:.4f}; best gain {r['best']:.4f}; N {r['N']} of {r.get('N_ref')}; "
              f"stop {r.get('stop')}; invalid ids {r.get('invalid')}" + (f"\n- ERROR: {r['error']}" if r.get("error") else ""),
              f"- batches (recorded ids expanded): {r.get('trace')}", f"- best-so-far curve: {r.get('curve')}", ""]
    return "\n".join(L)


def dream(policy: str, train: list[str], heldout: list[str], name: str, M: int = 8, model: str = "gpt-6.1-sol",
          effort: str = "high", backend=None, sandbox=SandboxPolicy, horizon: int | None = None) -> dict:
    out = tree_root() / "dream" / name
    out.mkdir(parents=True, exist_ok=True)
    lock = FileLock(out / ".lock")
    if not lock.acquire(blocking=False):
        raise RuntimeError(f"another dream {name} is running")
    tr, ho = [load_tree(s) for s in train], [load_tree(s) for s in heldout]
    overlap = {tree_id(t) for t in tr} & {tree_id(t) for t in ho}
    if overlap:
        raise ValueError(f"held-out trees overlap the training trees: {sorted(overlap)}")
    cur = resolve(policy)
    shutil.copy2(cur, out / "policy-0.py")
    cur_train = evaluate(out / "policy-0.py", tr, sandbox, horizon)
    best = {"path": out / "policy-0.py", "train": cur_train, "V": mean_v(cur_train), "rev": 0, "description": "current policy"}
    attempts = [dict(rev=0, description="current policy", V_train=best["V"])]
    backend = backend or CodexBackend(model, effort, 1800)
    for i in range(1, M + 1):
        rdir = out / f"rev{i}"
        shutil.rmtree(rdir, ignore_errors=True)
        view = rdir / "view"
        (view / "trees").mkdir(parents=True)
        shutil.copy2(best["path"], view / "policy.py")
        (view / "API.md").write_text((Path(__file__).resolve().parent / "policy.py").read_text().split('"""')[1])
        for t in tr:
            (view / "trees" / f"{t.meta['pack']}__{t.meta['tag']}.md").write_text(tree_md(t))
        (view / "replay.md").write_text(replay_md(best["train"]))
        (view / "attempts.md").write_text("# Earlier revisions this round\n\n" + "\n".join(
            f"- rev {a['rev']}: V_train {a['V_train']:.4f} — {a['description']}" + (f" (error: {a['error'][:200]})" if a.get("error") else "")
            for a in attempts) + "\n")
        (view / "prompt.md").write_text(PROMPT.read_text() + "\n## Earlier revisions\n" + (view / "attempts.md").read_text())
        try:
            prop = backend._call(view, rdir / "agent", f"arlab-dream-{name}-{i}", backend.model, backend.effort)
        except InfraError as e:
            attempts.append(dict(rev=i, description="infra error", V_train=float("-inf"), error=str(e)))
            continue
        cand = out / f"policy-{i}.py"
        cand.write_bytes(safe_read(view / "policy.py", MAX_POLICY))  # regular file only; never follows agent symlinks
        if prop.action != "edit" or sha(cand) == sha(best["path"]):
            attempts.append(dict(rev=i, description=f"no change ({prop.description})", V_train=best["V"]))
            continue
        res = evaluate(cand, tr, sandbox, horizon)
        v = mean_v(res)
        err = next((r["error"] for r in res if r.get("error")), None)
        attempts.append(dict(rev=i, description=prop.description, V_train=v, error=err, tokens=prop.tokens))
        if v > best["V"] and not err:
            best = {"path": cand, "train": res, "V": v, "rev": i, "description": prop.description}
    base_ho = evaluate(out / "policy-0.py", ho, sandbox, horizon) if ho else []
    best_ho = evaluate(best["path"], ho, sandbox, horizon) if ho and best["rev"] else base_ho
    improved = best["rev"] > 0
    ho_err = any(r.get("error") for r in base_ho + best_ho)
    accepted = improved and not ho_err and (not ho or mean_v(best_ho) >= mean_v(base_ho))
    rep = {"name": name, "policy_in": str(cur), "train": train, "heldout": heldout, "M": M, "beta": BETA, "attempts": attempts,
           "best_rev": best["rev"], "V_train": {"current": mean_v(cur_train), "best": best["V"]},
           "V_heldout": {"current": mean_v(base_ho) if ho else None, "best": mean_v(best_ho) if ho else None},
           "accepted": accepted, "heldout_errors": ho_err, "best_sha": sha(best["path"]),
           "guard": "held-out V >= current" if ho else "UNGUARDED (no held-out trees)", "horizon": horizon, "finished": now()}
    if accepted:
        pol = tree_root() / "policies" / f"{name}.py"
        pol.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["path"], pol)
        assert sha(pol) == rep["best_sha"]
        rep["policy_out"] = str(pol)
    write_json(out / "report.json", rep)
    (out / "report.md").write_text(f"# dream {name}\n\n```json\n{json.dumps(rep, indent=1, default=str)}\n```\n")
    lock.release()
    return rep


def tree_id(t: Tree) -> str:
    return hashlib.sha256(json.dumps(sorted((n["id"], n["parent"], n.get("score"), n["status"]) for n in t.nodes),
                                     default=str).encode()).hexdigest()[:16]
