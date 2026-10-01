"""arlab tree run | replay | dream | import | status."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .dream import dream, evaluate, import_campaign, load_tree, tree_root
from .policy import resolve
from .replay import BETAS, mean_v


def unit(pack: str, tag: str) -> str:
    return f"arlab-tree-{pack.strip('_')}-{tag}".replace("_", "-")


def cmd_run(a):
    from ..cli import ARLAB_BIN
    from .online import TreeCampaign
    pack = Path(a.pack).resolve()
    if a.detach:
        c = TreeCampaign(pack, a.tag)  # validates the tag; writes nothing
        c.dir.mkdir(parents=True, exist_ok=True)
        log = c.dir / "runner.log"
        cmd = ["systemd-run", "--user", f"--unit={unit(pack.name, a.tag)}", "--collect", "-p", f"StandardOutput=append:{log}",
               "-p", f"StandardError=append:{log}", str(ARLAB_BIN), "tree", "run", str(pack), "--tag", a.tag,
               "--budget", str(a.budget), "--workers", str(a.workers)]
        for flag, v in (("--policy", a.policy and str(resolve(a.policy))), ("--root-from", a.root_from), ("--calib-from", a.calib_from),
                        ("--opening-from", a.opening_from)):
            if v:
                cmd += [flag, v]
        return subprocess.run(cmd).returncode
    return TreeCampaign(pack, a.tag, a.policy, a.budget, a.workers, script=Path(a.script).resolve() if a.script else None,
                        root_from=a.root_from, calib_from=a.calib_from, opening_from=a.opening_from).run()


def cmd_replay(a):
    trees = [load_tree(s) for s in a.trees]
    rows = []
    for p in a.policy:
        res = evaluate(resolve(p), trees)
        rows.append({"policy": p, **{f"V@{b}": round(mean_v(res, b), 4) for b in BETAS},
                     "per_tree": {r["tree"]: {"V": round(r["V"]["0.5"], 4), "N": r["N"], "best": round(r["best"], 4), "stop": r.get("stop"),
                                              **({"error": r["error"][-200:]} if r.get("error") else {})} for r in res}})
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for b in BETAS:
            rank = sorted(rows, key=lambda r: -r[f"V@{b}"])
            print(f"β={b}: " + "  >  ".join(f"{r['policy']} ({r[f'V@{b}']})" for r in rank))
        for r in rows:
            print(f"\n{r['policy']}:")
            for t, v in r["per_tree"].items():
                print(f"   {t:40} {v}")
    return 0


def cmd_dream(a):
    rep = dream(a.policy, a.train, a.heldout or [], a.name, a.m, a.model, a.effort, horizon=a.horizon)
    print(json.dumps({k: rep[k] for k in ("best_rev", "V_train", "V_heldout", "accepted", "guard")} | {"out": str(tree_root() / "dream" / a.name)}, indent=1))
    return 0


def cmd_block_eval(a):
    from .blocks import block_eval, load_campaign
    pack = Path(a.pack).resolve()
    camps = {arm: load_campaign(pack, tag) for arm, tag in (x.split("=", 1) for x in a.arm)}
    rec = block_eval(camps, a.seeds, Path(a.out).resolve())
    print(json.dumps({k: {kk: v[kk] for kk in ("node", "q", "nodes", "calls", "fallback")} for k, v in rec["arms"].items()}, indent=1))
    return 0


def cmd_protocol(a):
    from .blocks import analyze
    rep = analyze([Path(f) for f in a.blocks])
    print(json.dumps(rep, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    return 0


def cmd_import(a):
    for s in a.campaigns:
        print(import_campaign(s))
    return 0


def cmd_status(a):
    t = load_tree(a.spec)
    ok = [n for n in t.nodes if n.get("score") is not None and n["id"] != "root"]
    counts: dict[str, int] = {}
    for n in t.nodes[1:]:
        counts[n["status"]] = counts.get(n["status"], 0) + 1
    best = max(ok, key=lambda n: n["score"]) if ok else None
    print(json.dumps({"tree": a.spec, "policy": t.meta.get("policy"), "W": t.meta.get("W"), "budget": t.meta.get("budget"),
                      "nodes": len(t.nodes) - 1, "batches": t.meta.get("batches"), "status": counts,
                      "best": best and {k: best.get(k) for k in ("id", "parent", "depth", "score", "primary", "description")}}, indent=1))
    return 0


def add_parser(sub):
    tp = sub.add_parser("tree", help="discovery-tree search (docs/TREE-SEARCH.md)")
    ts = tp.add_subparsers(dest="tcmd", required=True)
    s = ts.add_parser("run")
    s.add_argument("pack"); s.add_argument("--tag", required=True)
    s.add_argument("--policy", help="built-in name or policy file (default parallel_refine; fixed at the first run)")
    s.add_argument("--budget", type=int, default=48, help="nodes (attempts)"); s.add_argument("--workers", type=int, default=4, help="W, batch width")
    s.add_argument("--root-from", help="<pack>/<tag>:<id>: start from that keep/node (scores are then relative to it)")
    s.add_argument("--calib-from", help="<pack>/<tag>: reuse its calibration if pack, data, image and root match")
    s.add_argument("--opening-from", help="<pack>/<tag>: start with that tree's first batch (same root; charged to this budget)")
    s.add_argument("--detach", action="store_true"); s.add_argument("--script", help=argparse.SUPPRESS)
    s.set_defaults(f=cmd_run)
    s = ts.add_parser("replay"); s.add_argument("--policy", nargs="+", required=True); s.add_argument("--trees", nargs="+", required=True)
    s.add_argument("--json", action="store_true"); s.set_defaults(f=cmd_replay)
    s = ts.add_parser("dream"); s.add_argument("--policy", required=True); s.add_argument("--train", nargs="+", required=True)
    s.add_argument("--heldout", nargs="*"); s.add_argument("--name", required=True); s.add_argument("-m", type=int, default=8)
    s.add_argument("--model", default="gpt-6.1-sol"); s.add_argument("--effort", default="high")
    s.add_argument("--horizon", type=int, help="replay node budget (default: each tree's recorded size)"); s.set_defaults(f=cmd_dream)
    s = ts.add_parser("block-eval", help="score each arm's chosen node and the root on shared fresh holdout seeds")
    s.add_argument("pack"); s.add_argument("--arm", nargs="+", required=True, help="ARM=tag, e.g. D=p3-b01-D")
    s.add_argument("--seeds", type=int, nargs="+", required=True); s.add_argument("--out", required=True); s.set_defaults(f=cmd_block_eval)
    s = ts.add_parser("protocol", help="pre-registered paired analysis over block.json files (docs/DREAM-PROTOCOL.md)")
    s.add_argument("blocks", nargs="+"); s.add_argument("--out"); s.set_defaults(f=cmd_protocol)
    s = ts.add_parser("import"); s.add_argument("campaigns", nargs="+", help="<pack>/<tag> of finished greedy campaigns"); s.set_defaults(f=cmd_import)
    s = ts.add_parser("status"); s.add_argument("spec", help="<pack>/<tag>"); s.set_defaults(f=cmd_status)
