"""Frozen EVALUATE for stencil-focus: fraction_required per item (stencil Exp 4B/4C primary) with the vendored,
unmodified MemoryCode checker; one item per dialogue.

The run is `invalid` when: the surface source contains any 5-word run from topics.json instruction texts or eval
queries; an item is missing or has a plan error; a recorded prompt differs from the prompt the frozen window
rebuilds from the recorded plan (so every reminder line is a verbatim sentence of sessions 0..s-1 and the header is
from the frozen set); a prompt exceeds W; a generation exceeds the cap.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
import types
import warnings

import window

RETOKENIZE_SLACK = 32      # re-tokenising generated text may differ from the generated ids by a few tokens
warnings.filterwarnings("ignore")


def write(obj):
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg})
    print("INVALID:", msg)
    raise SystemExit(0)


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def ngrams(text, n=5):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def topic_ngrams(topics):
    out = set()
    for t in topics["instructions"]:
        for x in t["text"]:
            out |= ngrams(x)
        out |= ngrams(t["eval_query"])
    return out


def surface_violation(work, topics):
    bad = topic_ngrams(topics)
    for path in sorted(glob.glob(f"{work}/**/*", recursive=True)):
        if os.path.isfile(path) and "__pycache__" not in path:
            hit = ngrams(open(path, errors="replace").read()) & bad
            if hit:
                return f"{os.path.relpath(path, work)} contains topics.json text: {' '.join(sorted(hit)[0])!r}"
    return None


def load_checker(private):
    code = f"{private}/vendor/memorycode/code"
    sys.modules.setdefault("fire", types.ModuleType("fire"))
    if code not in sys.path:
        sys.path.insert(0, code)
    spec = importlib.util.spec_from_file_location("memorycode_evaluate", f"{code}/evaluate_model_output.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def compute_score(text, obj, regex):
        try:
            return mod.compute_score(text, obj, regex)
        except Exception:  # noqa: BLE001  (e.g. null bytes or deep recursion in the model's code): check failed
            return 0.0
    return compute_score


def read(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def score_row(row, lab, mc, compute_score):
    text = row.get("text") or ""
    sc = mc.score_generation(text, lab["history_regex"], compute_score, lab["required"], lab["structure"])
    ids = window.tokenizer().encode(text).ids
    fail = mc.output_failures(text, ids, truncated=row.get("finish_reason") == "length", timed_out=bool(row.get("timed_out")))
    return sc, fail, len(ids)


def main():
    global a, PUB, PRIV
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, required=True)
    ap.add_argument("--data", default="/data")
    ap.add_argument("--work", default="/work")
    a = ap.parse_args()
    PUB, PRIV = f"{a.data}/public", f"{a.data}/private"
    topics = json.load(open(f"{PRIV}/vendor/memorycode/topics.json"))
    why = surface_violation(a.work, topics)
    if why:
        invalid(why)
    mc, _ = window._stencil()
    compute_score = load_checker(PRIV)
    items = {it["id"]: (it, window.W, "main") for it in read(f"{PUB}/items.jsonl")}
    items.update({it["id"]: (it, 1536, "real") for it in read(f"{PUB}/real.jsonl")})
    labels = {x["id"]: x for x in read(f"{PRIV}/labels.jsonl") + read(f"{PRIV}/real_labels.jsonl")}
    try:
        rows = read(f"{a.run}/outputs.jsonl")
    except (OSError, ValueError) as e:
        invalid(f"outputs.jsonl unreadable: {e}")
    by_id = {}
    for r in rows:
        if not isinstance(r, dict) or r.get("id") not in items or r["id"] in by_id:
            invalid(f"unexpected or duplicate output row: {str(r)[:120]}")
        by_id[r["id"]] = r
    missing = sorted(set(items) - set(by_id))
    if missing:
        invalid(f"{len(missing)} items without output, e.g. {missing[0]}")
    tok = window.tokenizer()
    scores, stats = {}, {"main": [], "real": []}
    for iid, (it, w, kind) in sorted(items.items()):
        r = by_id[iid]
        if r.get("plan_error"):
            invalid(f"plan error on {iid}: {r['plan_error'][:300]}")
        if r.get("gen_error"):
            invalid(f"generation error on {iid}: {r['gen_error'][:300]}")
        try:
            win = window.Window(it, w)
            plan = win.check(window.Plan(**r["plan"]))
            built = win.build(plan)
        except Exception as e:  # noqa: BLE001
            invalid(f"recorded plan of {iid} does not rebuild: {type(e).__name__}: {e}")
        if built["prompt"] != r.get("prompt"):
            invalid(f"recorded prompt of {iid} differs from the frozen rebuild (non-verbatim reminder or altered prompt)")
        if len(tok.encode(r["prompt"]).ids) > w:
            invalid(f"prompt of {iid} exceeds W={w}")
        sc, fail, n_text = score_row(r, labels[iid], mc, compute_score)
        if int(r.get("completion_tokens", 0)) > a.max_new or n_text > a.max_new + RETOKENIZE_SLACK:
            invalid(f"generation of {iid} exceeds the cap {a.max_new}")
        fr = sc["fraction_required"]
        if fr is None or not math.isfinite(fr):
            invalid(f"item {iid} has no required check (bank error)")
        stats[kind].append({"fr": fr, "strict": bool(sc["strict"]), "fail": mc.any_failure(fail), **fail,
                            "reminder_tokens": built["reminder_tokens"], "n_reminder": len(plan.ids),
                            "prompt_tokens": built["prompt_tokens"], "before": plan.placement == "before_thread",
                            "completion_tokens": int(r.get("completion_tokens", 0))})
        if kind == "main":
            scores[iid] = fr
    m = stats["main"]
    mean = lambda k, xs=m: sum(float(x[k]) for x in xs) / len(xs)  # noqa: E731
    metrics = {"fraction_required": mean("fr"), "strict_rate": mean("strict"), "output_failure_rate": mean("fail"),
               "invalid_code_rate": mean("invalid"), "cap_hit_rate": mean("truncated"), "degenerate_rate": mean("degenerate"),
               "timeout_rate": mean("timed_out"), "reminder_tokens_mean": mean("reminder_tokens"),
               "reminder_sentences_mean": mean("n_reminder"), "prompt_tokens_mean": mean("prompt_tokens"),
               "before_thread_share": mean("before"), "completion_tokens_mean": mean("completion_tokens"), "n_items": len(m)}
    if stats["real"]:
        metrics.update(real_fraction_required=mean("fr", stats["real"]), real_output_failure_rate=mean("fail", stats["real"]),
                       real_n=len(stats["real"]))
    write({"valid": True, "primary": metrics["fraction_required"], "metrics": metrics, "items": scores, "message": "ok"})
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
