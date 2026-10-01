"""TreeCampaign: grow a discovery tree online, then finalize with one pre-registered holdout.

Reuses the greedy runner end to end (CHECK/PREPARE/TESTS/SEAL, CALIBRATE, trials, guards, FINALIZE, locks); only the
LOOP is replaced. Each node = one Codex proposal applied on its *parent's* commit + one screen trial (+ one fix call
on a crash, like the greedy runner). Nodes are scored against the baseline in MES units; no keep/discard decisions.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import stats
from ..agent import AuthError, InfraError, ScriptedBackend, codex_login_ok
from ..campaign import Campaign, ConfigError, debug_kill, now
from ..experiment import NOTES_MAX, apply_view, build_view, candidate_trial, guard_failures, safe_read, strip
from ..campaign import DISK_FLOOR_GB
from ..guards import disk_free_gb
from ..pack import SEAL_ITEMS, hash_paths, surface_files
from ..record import add_constraints, read_json, write_json
from .model import DROPPED, Tree, clean_selection, public_view
from .policy import PolicyError, SandboxPolicy, resolve, sha

CALIB_KEYS = ("sigma", "n_validation_items", "n_holdout_items", "expected_holdout_se", "underpowered", "calibration",
              "references", "probe", "mem_gb")
TOP_K = 3


class TreeCampaign(Campaign):
    def __init__(self, pack_dir: Path, tag: str, policy: str | None = None, budget: int = 48, workers: int = 4,
                 script: Path | None = None, runs_root: Path | None = None, root_from: str | None = None,
                 calib_from: str | None = None):
        super().__init__(pack_dir, tag, script, runs_root)
        self.tree_path = self.dir / "tree.json"
        self.tree = Tree.load(self.tree_path)
        self.new_policy = None
        if self.tree is None:  # created on disk only in init_work, i.e. under the run lock
            pol = resolve(policy or "parallel_refine")
            self.new_policy = pol
            self.tree = Tree(self.tree_path, {"pack": self.name, "tag": tag, "policy": pol.stem, "policy_sha": sha(pol),
                                              "budget": budget, "W": workers, "root_from": root_from, "calib_from": calib_from,
                                              "created": now()})
        self.git_lock, self.trial_lock, self.file_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.local = threading.local()

    # ------------------------------------------------------------ hooks into Campaign
    def make_backend(self):
        if self.script:
            return ScriptedBackend(self.script, lambda: self.local.index)
        return super().make_backend()

    def init_work(self):
        """Baseline = the sealed surface, or the surface of --root-from (then every score is relative to that root)."""
        if self.new_policy is not None:
            shutil.copy2(self.new_policy, self.dir / "policy.py")  # the run always uses this copy (resume-safe)
            self.tree.save()
            self.new_policy = None
        if (self.work / ".git").exists() and self.state.get("baseline_commit"):
            return
        shutil.rmtree(self.work, ignore_errors=True)  # a half-initialized work/ (killed during init) is rebuilt
        shutil.copytree(self.sealed / "surface", self.work)
        subprocess.run(["chmod", "-R", "u+w", str(self.work)], check=True)
        rf = self.tree.meta.get("root_from")
        if rf:
            src, commit = self._source(rf)
            tmp = self.dir / "root-from"
            src.export(commit, tmp)
            for f in surface_files(self.sealed / "surface"):
                if (tmp / f).exists():
                    shutil.copy2(tmp / f, self.work / f)
            shutil.rmtree(tmp)
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"baseline{' (root from ' + rf + ')' if rf else ''}")
        self.git("tag", "baseline")
        self.save(baseline_commit=self.git("rev-parse", "HEAD"), baseline_tree=self.git("rev-parse", "HEAD^{tree}"), mode="tree", root_from=rf)

    def _source(self, spec: str) -> tuple[Campaign, str]:
        """'<pack>/<tag>:<id>' → (source campaign, commit). id = a greedy keep id or a tree node id."""
        loc, nid = spec.rsplit(":", 1)
        pack, tag = loc.split("/")
        src = Campaign(self.pack_dir.parent / pack, tag, runs_root=self.dir.parent.parent)
        if src.state.get("seal_hash") != self.state.get("seal_hash") and not _same_code(src, self):
            raise ConfigError(f"--root-from {spec}: sealed pack differs from this run's")
        rec = read_json(src.dir / "runs" / nid / "record.json")
        if rec:
            return src, rec["commit"]
        t = Tree.load(src.dir / "tree.json")
        if t:
            return src, t.get(nid)["commit"]
        raise ConfigError(f"--root-from {spec}: no record or node {nid}")

    def calibrate(self):
        cf = self.tree.meta.get("calib_from")
        if cf and "sigma" not in self.state:
            pack, tag = cf.split("/")
            src = Campaign(self.pack_dir.parent / pack, tag, runs_root=self.dir.parent.parent)
            same = all(src.state.get(k) == self.state.get(k) for k in ("seal_hash", "data_hash", "image_id"))
            same = same and src.git("rev-parse", f"{src.state.get('baseline_commit')}^{{tree}}") == self.state.get("baseline_tree")
            if same and "sigma" in src.state:
                shutil.copytree(src.dir / "calib", self.dir / "calib", dirs_exist_ok=True)
                self.save(**{k: src.state.get(k) for k in CALIB_KEYS}, calib_copied_from=cf)
                self.log(f"CALIBRATE reused from {cf}")
            else:
                self.log(f"--calib-from {cf}: pack/data/image/root differ; calibrating afresh")
        super().calibrate()

    def resume_records(self):
        """Pending/infra/contended nodes (runner died mid-batch) are dropped; the policy re-decides from the saved tree."""
        drop = {n["id"] for n in self.tree.nodes if n["status"] in DROPPED}
        drop |= {n["id"] for n in self.tree.nodes if n["parent"] in drop}  # cannot happen (batches are synchronous); defensive
        for nid in drop:
            shutil.rmtree(self.dir / "nodes" / nid, ignore_errors=True)
        self.tree.nodes = [n for n in self.tree.nodes if n["id"] not in drop]
        self.tree.save()
        if self.state.get("baseline_commit"):
            self.git("reset", "-q", "--hard", self.state["baseline_commit"])
            self.git("clean", "-qfdx")

    def records(self) -> list[dict]:
        """Nodes as ledger records (status counts toward the ≥10-experiments verdict rule)."""
        return [{"id": n["id"], "status": n["status"], "commit": n.get("commit"), "primary": n.get("primary"),
                 "delta": n.get("d"), "se": n.get("se"), "hypothesis_tag": n.get("tag"), "description": n.get("description", ""),
                 "model": n.get("model", ""), "agent_calls": n.get("agent_calls", 0), "tokens": n.get("tokens", {}),
                 "timings": n.get("timings", {})} for n in self.tree.nodes if n["id"] != "root" and n["status"] not in DROPPED]

    def incumbent(self, records):
        ch = self.state.get("tree_choice") or {}
        if ch.get("node"):
            return ch["commit"], {}, ch["node"]
        return self.state["baseline_commit"], self.baseline_vals(), "baseline"

    def report(self):
        from .report import write_tree_report
        return write_tree_report(self)

    # ------------------------------------------------------------ LOOP = grow the tree
    def stop_reason_tree(self) -> str | None:
        m = self.tree.meta
        streak = 0
        for n in reversed(self.tree.meta.get("infra_log", [])):
            if n != "infra":
                break
            streak += 1
        checks = [(self.tree.used() >= m["budget"], "budget"), ((self.dir / "STOP").exists(), "stop"),
                  (streak >= 6, "infra"),
                  (disk_free_gb(self.dir) < DISK_FLOOR_GB + 2 * self.state.get("probe", {}).get("out_gb", 0), "disk")]
        return next((why for hit, why in checks if hit), None)

    def loop(self):
        self.backend = self.backend or self.make_backend()
        m = self.tree.meta
        with SandboxPolicy(self.dir / "policy.py", name=f"{self.prefix}-policy") as pol:
            while True:
                why = self.stop_reason_tree()
                if why:
                    break
                left = m["budget"] - self.tree.used()
                self.save(phase="loop", current=f"batch {m.get('batches', 0) + 1}")
                try:
                    raw = pol.select(public_view(self.tree.nodes), left, m["W"])
                except PolicyError as e:
                    self.log(f"policy error: {e}")
                    why = "policy_error"
                    break
                sel, bad = clean_selection(raw, self.tree.nodes, m["W"], left)
                if bad:
                    self.log(f"policy returned {bad} ids outside A(T): {raw}")
                if not sel:
                    why = "policy_stop"
                    break
                m["batches"] = m.get("batches", 0) + 1
                m.setdefault("selections", []).append(sel)
                self.expand_batch(sel)
        self.save(stop_reason=why)
        self.log(f"STOP: {why} ({self.tree.used()} nodes)")

    def expand_batch(self, parents: list[str]):
        batch = []
        for p in parents:
            nid = self.tree.new_id()
            par = self.tree.get(p)
            n = {"id": nid, "parent": p, "depth": par["depth"] + 1, "order": max(x["order"] for x in self.tree.nodes) + 1, "status": "pending",
                 "score": None, "tag": "", "cost_s": 0.0, "round": self.tree.meta["batches"], "started": now()}
            self.tree.nodes.append(n)
            batch.append(n)
        self.tree.save()
        debug_kill(f"after_pending:{self.tree.meta['batches']}")
        for n in batch:  # siblings started from the same parent at the same moment (see tree_history)
            same = [x for x in batch if x["parent"] == n["parent"]]
            n["slot"] = [same.index(n) + 1, len(same)]
        with ThreadPoolExecutor(len(batch)) as ex:
            futs = [ex.submit(self.expand_node, n, i) for i, n in enumerate(batch)]
            outs = [f.result() for f in futs]
        auth = any(o == "auth" for o in outs)
        with self.file_lock:
            for n in batch:
                if n["status"] in DROPPED:
                    self.tree.nodes.remove(n)
                self.tree.meta.setdefault("infra_log", []).append("infra" if n["status"] in DROPPED else "ok")
            self.tree.save()
        if auth:
            self.save(paused="codex auth")
            self.log("PAUSED: codex auth")
            while not codex_login_ok():
                time.sleep(600)
            self.save(paused=None)
        elif all(n["status"] in DROPPED for n in batch):
            time.sleep(float(os.environ.get("ARLAB_BACKOFF_S", "60")))

    def tree_history(self, parent: dict, slot: list[int] | None = None) -> str:
        p, st = self.pack, self.state
        s = p.seeds.screen
        base = self.baseline_vals()[s]["primary"]
        f = lambda v: "" if v is None else f"{v:.4g}"
        ok = [n for n in self.tree.nodes if n["id"] != "root" and n["status"] not in DROPPED]
        chain = self.tree.path_to(parent["id"])
        L = ["# history.md (tree search; written by the runner)", "",
             f"Metric: {p.metric.name} ({p.metric.direction}). MES (judged only at the end): {p.metric.mes}.",
             f"Baseline on screen seed {s}: {base:.6g}; sigma={st.get('sigma', 0):.4g}. "
             "score = improvement over the baseline in MES units (1.0 = one MES; screen seed only, noisy).",
             "", "Several attempts are explored in parallel as a tree. Your edit starts from YOUR PARENT's code (the files "
             "in this directory); incumbent.diff shows baseline → your parent (every edit on your chain so far).", "",
             "## Your chain (root → your parent)", "", "| id | status | primary | score | tag | description |", "|---|---|---|---|---|---|"]
        for n in chain:
            L.append(f"| {n['id']} | {n['status']} | {f(n.get('primary', base if n['id'] == 'root' else None))} | {f(n['score'])} | "
                     f"{n.get('tag', '')} | {n.get('description', '') if n['id'] != 'root' else 'baseline'} |")
        if parent["id"] != "root" and parent["status"] != "ok":
            L += ["", f"Your parent did not produce a score ({parent['status']}: {parent.get('reason', '')[:300]}). "
                  "Fixing it or taking a different direction are both fine."]
        tried = [n for n in ok if n["parent"] == parent["id"]]
        if tried:
            L += ["", "## Already tried from your parent", "", "| id | status | score | description |", "|---|---|---|---|"]
            L += [f"| {n['id']} | {n['status']} | {f(n['score'])} | {n.get('description', '')} |" for n in tried]
        if slot and slot[1] > 1:
            L += ["", f"## Parallel siblings: you are attempt {slot[0]} of {slot[1]} started from your parent at the same moment",
                  "", "The others see exactly what you see. To avoid duplicate work, list for yourself the distinct directions that "
                  f"look most promising from here, rank them, and implement the one you rank #{slot[0]} (a genuinely different idea, "
                  "not a variant of a higher-ranked one). Ideas already tried from your parent (above) do not count as options."]
        best = sorted([n for n in ok if n["score"] is not None], key=lambda n: -n["score"])[:5]
        L += ["", "## Best attempts anywhere in the tree", "", "| id | depth | primary | score | description |", "|---|---|---|---|---|"]
        L += [f"| {n['id']} | {n['depth']} | {f(n.get('primary'))} | {f(n['score'])} | {n.get('description', '')} |" for n in best] or ["| — | | | | |"]
        others = [n for n in ok if n not in chain][-15:]
        L += ["", "## Other recent attempts (not on your chain)", "", "| id | parent | status | score | tag | description |",
              "|---|---|---|---|---|---|"]
        L += [f"| {n['id']} | {n['parent']} | {n['status']} | {f(n['score'])} | {n.get('tag', '')} | {n.get('description', '')} |" for n in others]
        tags: dict[str, dict] = {}
        for n in ok:
            t = tags.setdefault(n.get("tag") or "?", {})
            t[n["status"]] = t.get(n["status"], 0) + 1
        L += ["", "## Hypothesis tags tried (whole tree)", ""]
        L += [f"- {t}: {sum(v.values())} tried — " + ", ".join(f"{k} {c}" for k, c in sorted(v.items())) for t, v in sorted(tags.items())]
        return "\n".join(L) + "\n"

    def expand_node(self, n: dict, index: int) -> str | None:
        """Fill node n in place (thread). Returns "auth" on a Codex login failure."""
        p, s, nid = self.pack, self.pack.seeds.screen, n["id"]
        self.local.index = n["order"] - 1
        ndir = self.dir / "nodes" / nid
        shutil.rmtree(ndir, ignore_errors=True)
        ndir.mkdir(parents=True)
        par = self.tree.get(n["parent"])
        pcommit = par.get("commit") or par.get("parent_commit") or self.state["baseline_commit"]  # no_op/skip inherit
        n.update(agent_calls=0, tokens={"input": 0, "cached_input": 0, "output": 0},
                 timings={"propose_s": 0.0, "run_s": 0.0, "eval_s": 0.0, "wait_s": 0.0}, parent_commit=pcommit)
        name = f"{self.prefix}-agent-{nid}"

        def finish(status, reason=""):
            cost = sum(n["timings"][k] for k in ("propose_s", "run_s", "eval_s"))  # active time, excl. queueing
            n.update(status=status, reason=reason[:500], finished=now(), cost_s=round(cost, 1))
            for t in ndir.glob("s*"):
                self.clean_trial({}, t)
            shutil.rmtree(ndir / "view" / "frozen_run", ignore_errors=True)
            with self.file_lock:
                self.tree.save()
            self.log(f"{nid} (parent {n['parent']}) {status} score={n.get('score')} {n.get('description', '')[:70]}")

        def agent(call, *args):
            t0 = time.monotonic()
            try:
                prop = call(*args)
            finally:
                n["timings"]["propose_s"] += time.monotonic() - t0
            n["agent_calls"] += 1
            for k in n["tokens"]:
                n["tokens"][k] += prop.tokens.get(k, 0)
            return prop

        def apply(view, msg):
            with self.git_lock:
                self.git("reset", "-q", "--hard", pcommit)
                changed, err = apply_view(self, view, ndir)
                if not changed:
                    return None, None
                self.git("add", "-A")
                self.git("commit", "-q", "-m", f"{nid}: {msg}")
                commit = self.git("rev-parse", "HEAD")
                self.git("tag", "-f", f"node/{nid}", commit)  # keeps every attempt reachable (no gc)
                self.git("reset", "-q", "--hard", self.state["baseline_commit"])
            n.update(commit=commit, files=changed)
            (ndir / "diff.patch").write_text(self.git("diff", pcommit, commit))
            return commit, err

        def screen(commit, err, attempt=""):
            if err:
                return {"status": "crash", "reason": f"py_compile: {err}", "log_tail": err, "run_s": 0, "eval_s": 0, "wait_s": 0}
            with self.trial_lock:
                res = candidate_trial(self, commit, s, ndir, nid, attempt)
            for k in ("run_s", "eval_s", "wait_s"):
                n["timings"][k] += res.get(k, 0) or 0
            with self.file_lock:
                add_constraints(self.dir, nid, res.get("log_tail", ""), None)
            return res

        view = build_view(self, ndir, [], pcommit, hist=self.tree_history(par, n.get("slot")))
        pnotes = self.dir / "nodes" / par["id"] / "notes.md"
        (view / "notes.md").write_bytes(pnotes.read_bytes() if pnotes.exists() else b"")  # notes are per chain
        try:
            prop = agent(self.backend.propose, view, ndir, name)
        except AuthError:
            finish("infra_error", "codex auth")
            return "auth"
        except InfraError as e:
            finish("infra_error", str(e))
            return None
        (ndir / "notes.md").write_bytes(safe_read(view / "notes.md", NOTES_MAX))
        n.update(description=prop.description, tag=prop.hypothesis_tag, model=prop.model)
        with self.file_lock:
            add_constraints(self.dir, nid, "", prop.constraint_learned)
        if prop.action == "skip":
            return finish("skip", "agent chose skip")
        commit, err = apply(view, prop.description)
        if not commit:
            return finish("no_op", "edit with an empty diff")
        res = screen(commit, err)
        if res["status"] == "crash":
            try:
                fprop = agent(self.backend.fix, view, ndir, name, res.get("log_tail", ""))
            except (InfraError, AuthError) as e:
                fprop = None
                self.log(f"{nid}: fix call failed: {e}")
            if fprop and fprop.action == "edit":
                c2, err2 = apply(view, prop.description + " (fixed)")
                if c2:
                    n["fixed"] = fprop.description
                    res = screen(c2, err2, "-fix")
        if res["status"] != "ok":
            n["log_tail"] = (res.get("log_tail") or "")[-2000:]
            return finish(res["status"], res.get("reason", ""))
        n["result"], n["primary"] = strip(res), res["primary"]
        fails = guard_failures(self, res)
        if fails:
            return finish("guard_fail", "; ".join(fails))
        scr = stats.compare({s: res}, {s: self.baseline_vals()[s]}, [s], p.metric.direction, self.state["sigma"])
        n.update(d=scr["d"], se=scr["se"], score=round(scr["d"] / p.metric.mes, 6))
        return finish("ok")

    # ------------------------------------------------------------ FINALIZE: top-k on screen → confirm → one holdout
    def finalize(self):
        p = self.pack
        if "tree_choice" not in self.state and not (self.state.get("underpowered") and not p.acceptance.allow_underpowered):
            self.save(phase="confirm")
            cands = sorted([n for n in self.tree.nodes if n["id"] != "root" and n["status"] == "ok" and n["score"] > 0],
                           key=lambda n: (-n["score"], n["order"]))[:TOP_K]
            table = []
            for n in cands:
                if p.seeds.confirm:
                    vals, bad = {}, None
                    for seed in p.seeds.confirm:
                        tdir = self.dir / "nodes" / n["id"] / f"confirm-s{seed}"
                        res = read_json(tdir / "result.json")
                        if res is None:
                            self.log(f"CONFIRM {n['id']} seed {seed}")
                            res = self.settled(lambda: self.trial(self.export(n["commit"], tdir / "surface"), seed, "validation", tdir))
                            self.clean_trial(res, tdir)
                            write_json(tdir / "result.json", res)
                        if res["status"] != "ok" or guard_failures(self, res):
                            bad = res["status"] if res["status"] != "ok" else "guard_fail"
                            break
                        vals[seed] = res
                    if bad:
                        table.append({"node": n["id"], "d": None, "why": bad})
                        continue
                    cc = stats.compare(vals, self.baseline_vals(), p.seeds.confirm, p.metric.direction, self.state["sigma"])
                    table.append({"node": n["id"], "d": cc["d"], "se": cc["se"], "per_seed": cc["per_seed"]})
                else:
                    table.append({"node": n["id"], "d": n["d"], "se": n["se"], "why": "no confirm seeds: screen d"})
            ok = [t for t in table if t["d"] is not None and t["d"] > 0]
            pick = max(ok, key=lambda t: t["d"]) if ok else None
            self.save(tree_choice={"candidates": table, "node": pick and pick["node"],
                                   "commit": pick and self.tree.get(pick["node"])["commit"],
                                   "rule": f"top {TOP_K} by screen score, best confirm d > 0; one holdout run"})
            self.log(f"CHOICE {pick and pick['node']} from {[t['node'] for t in table]}")
        super().finalize()


CODE_NEUTRAL = ("guards", "agent", "campaign")  # pack.yaml keys that never change what a root commit computes


def _same_code(a: Campaign, b: Campaign) -> bool:
    """Root commits carry across packs that differ only in docs, seeds.holdout, guards, the agent or campaign limits."""
    import yaml
    if a.state.get("data_hash") != b.state.get("data_hash"):
        return False
    items = [i for i in SEAL_ITEMS if i not in ("pack.yaml", "program.md", "IDEA.md")] + ["arlab_lib"]  # docs: agent-facing only
    if hash_paths([a.sealed / i for i in items]) != hash_paths([b.sealed / i for i in items]):
        return False
    ya, yb = (yaml.safe_load((c.sealed / "pack.yaml").read_text()) for c in (a, b))
    for y in (ya, yb):
        y.get("seeds", {}).pop("holdout", None)
        for k in CODE_NEUTRAL:
            y.pop(k, None)
    return ya == yb
