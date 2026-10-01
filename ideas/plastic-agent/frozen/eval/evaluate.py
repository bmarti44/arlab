"""Frozen EVALUATE for plastic-agent. Never imports the surface; scores only what the frozen harness saved (the
per-world adapters) by running held-out tasks against the private FauxOS simulator.

The arm is decided by frozen code (common.arm_of on /work/adapt.py, cross-checked against stats.json), never by the
surface: "adapter" (any candidate), or the frozen references "none", "icl", "placebo".
For each world j: merge adapter j (if any) into the base weights, then
  own      world j's tasks, prompt = system prompt with world j's tool names + goal (NO transcript; arm "icl" instead
           puts world j's transcript in the system prompt and uses no adapter). Every arm is scored on its own worlds;
           for the placebo reference, adapter j was trained by the harness on world j's TWIN transcript.
  twin     (arm "adapter" only) world j's tasks with the candidate's adapter of world j's TWIN (adapters/<j>x: trained by
           the same surface on the twin's transcript; the twin has the SAME tool names, operation set and argument
           vocabulary, and other semantics): the world-specific learning control. twin_success, world_specific_gain =
           success - twin_success, and the guard world_specific_share = (success - twin_success) / (success -
           none_success) (min 0.5; 1.0 when success - none_success < 0.025, vacuous for a run that gains nothing).
           A surface that learns only name / vocabulary priors adapts the same way to both and gets share ~0.
           Cost: one more merge + one more pass over the split's tasks (est. ~15-40 s validation, ~35-100 s holdout).
  battery  this adapter's share of the forgetting battery: GSM8K items, guard-world tasks with the guard world's
           transcript in context ("can it still drive elsewhere"), OASST2 text NLL
Each task: greedy program (<= 64 tokens), executed from the world's post-exploration state; on an execution or syntax
error one retry that shows the failing line and its output. Success = no error, exact gold state (state tasks) and the
gold value in the last output (answer tasks). Base-model results (no adapter) are cached in /cache.
Token accounting per task (context positions of each model call, the shared system prefix included in each call):
initial_context_tokens (the first call), retry_prompt_tokens (the retry call's full context, 0 without a retry),
prefill_tokens = their sum (every prompt position processed, retries included: the guard), decode_tokens (generated),
inference_tokens = prefill_tokens + decode_tokens.

metrics.json: primary = success (own-world task success) for every arm; items = per task 0/1 for the primary. In v2
"fam" worlds every op has exactly 6 tasks per world, so success = the mean over worlds of per-op macro averages;
s_familiar / s_dev / s_reserved (and none_*) report op-type strata.
Guards read battery_drop, prefill_tokens and world_specific_share. The forgetting battery (guard world, GSM8K items,
text rows) is the split's own: validation and holdout batteries are disjoint. A run whose supervisor recorded a sandbox
failure or a thread / process violation in any world is invalid.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/frozen")
import lora  # noqa: E402
from common import (GSM8K_SYSTEM, GSM8K_USER, MAX_GSM8K_TOKENS, MAX_PROGRAM_TOKENS, MODEL_DIR, STOP_IDS,  # noqa: E402
                    arm_of, chat_prefix, chat_user, retry_suffix, system_text, task_suffix)
from engine import Engine  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import gsm_score, world_specific_share  # noqa: E402

EVAL_VERSION = "pa-eval-3"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/data")
ap.add_argument("--cache", default="/cache")
ap.add_argument("--work", default="/work", help="the surface that produced the run (read only to identify the arm)")
ap.add_argument("--frozen", default="/frozen")
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default=MODEL_DIR, help="tests only: a tiny random model")
ap.add_argument("--max-new", type=int, default=MAX_PROGRAM_TOKENS, help="tests only")
a = ap.parse_args()


def write(obj):
    tmp = a.out + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, a.out)


def invalid(msg):
    write({"valid": False, "primary": None, "metrics": {}, "items": None, "message": msg})
    sys.exit(0)


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


D = a.data
order = json.load(open(f"{D}/public/order.json"))
pub = [json.load(open(f"{D}/public/worlds/{wid}.json")) for wid in order]
priv = json.load(open(f"{D}/private/worlds.json"))
guard = json.load(open(f"{D}/private/guard.json"))
gsm = json.load(open(f"{D}/private/gsm8k.json"))
text = np.load(f"{D}/private/text.npy")
sys.path.insert(0, f"{D}/private")
import fauxos  # noqa: E402  (the private simulator)

# ---------------------------------------------------------------- run outputs: well-formed or invalid
try:
    stats = json.load(open(f"{a.run}/stats.json"))
    wids = [w["id"] for w in stats["worlds"]]
    donors = [w["donor"] for w in stats["worlds"]]
    passes = stats["worlds"] + stats["twins"]
    tw_ids, tw_donors = [w["id"] for w in stats["twins"]], [w["donor"] for w in stats["twins"]]
    has = {w["id"]: bool(w["adapter"]) for w in passes}
    nan = bool(stats["nan"]) or any(bool(w["nan_seen"]) for w in passes)
    violations = [f"{w['id']}: {v}" for w in passes for v in w["violations"]]
    sandbox_ok = all(w["sandbox"]["ok"] is True for w in passes)
except Exception as e:  # missing, wrong format
    invalid(f"missing or unreadable run outputs: {e!r}"[:500])
arm = arm_of(a.work, a.frozen)
if stats.get("arm") != arm:
    invalid(f"stats.json arm {stats.get('arm')!r} != the arm of this surface ({arm})")
if wids != order:
    invalid("stats.json does not cover exactly this split's worlds in order")
twin_ids = [f"{w}x" for w in order]
if donors != (twin_ids if arm == "placebo" else order):
    invalid("stats.json: every candidate world must be adapted on its own transcript")
if (tw_ids, tw_donors) != ((twin_ids, twin_ids) if arm == "adapter" else ([], [])):
    invalid("stats.json: a candidate run must carry exactly one twin (control) pass per world, on the twin's transcript")
if nan:
    invalid("training diverged (a non-finite loss or gradient in some train() call)")
if not sandbox_ok or violations or stats.get("violations"):
    invalid(f"the surface process broke isolation (threads / processes / sandbox): {violations[:5]}")
on_disk = sorted(os.listdir(f"{a.run}/adapters")) if os.path.isdir(f"{a.run}/adapters") else []
if len(has) != len(passes) or on_disk != sorted(w for w, h in has.items() if h):
    invalid("adapter files do not match stats.json")
if arm in ("none", "icl") and any(has.values()):
    invalid(f"the {arm} arm must not adapt")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

dev = a.device
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
E = Engine(model, tok, dev)
adapters = {}
for wid, h in has.items():
    if h:
        try:
            cfg, tens = lora.load(f"{a.run}/adapters/{wid}")
        except Exception as e:
            invalid(f"adapter {wid} unreadable: {e!r}"[:500])
        why = lora.validate(model, cfg, tens)
        if why:
            invalid(f"adapter {wid}: {why}")
        adapters[wid] = (cfg, tens)


# ---------------------------------------------------------------- scoring primitives
def decode(ids) -> str:
    return tok.decode([t for t in ids if t not in STOP_IDS])


def run_tasks(system: str, spec: dict, end: dict, tasks: list[dict], batch: int) -> dict:
    """{task id: {"s", "retry", "first_error", "ctx0", "ctx_retry", "dec"}} for greedy programs with one retry on error.
    ctx0 / ctx_retry = context positions of the first / retry call (prefix + suffix), dec = generated tokens."""
    pre_ids = E.encode(chat_prefix(system))
    pre = E.prefix_kv(pre_ids)
    suf = [E.encode(task_suffix(t["goal"])) for t in tasks]
    gen1 = E.generate(pre, suf, a.max_new, batch_size=batch)
    first = [decode(o) for o in gen1]
    res = [fauxos.run_program(spec, end, txt) for txt in first]
    out = {t["id"]: {"s": fauxos.score(t, r), "retry": False, "first_error": r["error"]["obs"][:40] if r["error"] else None,
                     "ctx0": len(pre_ids) + len(s), "ctx_retry": 0, "dec": len(g)} for t, r, s, g in zip(tasks, res, suf, gen1)}
    redo = [i for i, r in enumerate(res) if r["error"] is not None]
    if redo:
        suf2 = [E.encode(retry_suffix(tasks[i]["goal"], first[i], res[i]["error"]["line"], res[i]["error"]["obs"])) for i in redo]
        gen2 = E.generate(pre, suf2, a.max_new, batch_size=batch)
        for i, s2, g in zip(redo, suf2, gen2):
            o = out[tasks[i]["id"]]
            o.update(s=fauxos.score(tasks[i], fauxos.run_program(spec, end, decode(g))), retry=True,
                     ctx_retry=len(pre_ids) + len(s2), dec=o["dec"] + len(g))
    del pre
    return out


def run_gsm(items: list[dict]) -> dict:
    if not items:
        return {}
    pre = E.prefix_kv(E.encode(chat_prefix(GSM8K_SYSTEM)))
    outs = E.generate(pre, [E.encode(chat_user(GSM8K_USER.format(q=x["q"]))) for x in items], MAX_GSM8K_TOKENS if a.max_new == MAX_PROGRAM_TOKENS else a.max_new, batch_size=32)
    return {x["id"]: gsm_score(decode(o), x["a"]) for x, o in zip(items, outs)}


@torch.no_grad()
def run_text(rows: np.ndarray) -> list[float]:
    out = []
    for i in range(0, len(rows), 8):
        r = torch.as_tensor(rows[i:i + 8], device=dev)
        h = model.model(input_ids=r[:, :-1], use_cache=False).last_hidden_state
        lp = torch.log_softmax(model.lm_head(h).float(), -1)
        out += (-lp.gather(-1, r[:, 1:].unsqueeze(-1)).squeeze(-1)).mean(-1).tolist()
    return out


G_SYS = system_text(guard["tools"], guard["transcript"])
BATCH = {"plain": 48, "icl": 16, "guard": 32}

# ---------------------------------------------------------------- base model (no adapter): cached, surface-independent
key = hashlib.sha256("|".join([EVAL_VERSION, a.model, dev, str(a.max_new)] + [sha(f"{D}/{p}") for p in (
    ["public/order.json"] + [f"public/worlds/{wid}.json" for wid in order] + ["private/worlds.json", "private/guard.json", "private/gsm8k.json", "private/text.npy"])]).encode()).hexdigest()[:16]
base_path = f"{a.cache}/base-{key}.json"
icl_path = f"{a.cache}/icl-{key}.json"
base = json.load(open(base_path)) if os.path.exists(base_path) else None
if base is None:
    base = {"none": {}, "guard": {}, "gsm": {}, "text": []}
    for w in pub:
        p = priv[w["id"]]
        base["none"].update(run_tasks(system_text(w["tools"]), p["spec"], p["end"], p["tasks"], BATCH["plain"]))
    base["guard"] = {k: v["s"] for k, v in run_tasks(G_SYS, guard["spec"], guard["end"], guard["tasks"], BATCH["guard"]).items()}
    base["gsm"] = run_gsm(gsm)
    base["text"] = run_text(text)
    os.makedirs(a.cache, exist_ok=True)
    json.dump(base, open(base_path + ".tmp", "w"))
    os.replace(base_path + ".tmp", base_path)

# ---------------------------------------------------------------- per-world evaluation
n = len(pub)
own, mism, bat_gsm, bat_guard, bat_text = {}, {}, {}, {}, [None] * len(text)
for j, w in enumerate(pub):
    p = priv[w["id"]]
    ad = adapters.get(w["id"])
    ctx = lora.Merged(model, *ad) if ad else None
    if ctx:
        ctx.__enter__()
    try:
        if arm == "icl":
            own.update(run_tasks(system_text(w["tools"], w["transcript"]), p["spec"], p["end"], p["tasks"], BATCH["icl"]))
        elif ad:
            own.update(run_tasks(system_text(w["tools"]), p["spec"], p["end"], p["tasks"], BATCH["plain"]))
        else:
            own.update({t["id"]: base["none"][t["id"]] for t in p["tasks"]})
        mine_gsm = [x for k, x in enumerate(gsm) if k % n == j]
        mine_guard = [t for k, t in enumerate(guard["tasks"]) if k % n == j]
        mine_text = [k for k in range(len(text)) if k % n == j]
        if ad:
            bat_gsm.update(run_gsm(mine_gsm))
            bat_guard.update({k: v["s"] for k, v in run_tasks(G_SYS, guard["spec"], guard["end"], mine_guard, BATCH["guard"]).items()})
            for k, v in zip(mine_text, run_text(text[mine_text])):
                bat_text[k] = v
        else:
            bat_gsm.update({x["id"]: base["gsm"][x["id"]] for x in mine_gsm})
            bat_guard.update({t["id"]: base["guard"][t["id"]] for t in mine_guard})
            for k in mine_text:
                bat_text[k] = base["text"][k]
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    if arm == "adapter":          # the control: world j scored with the adapter the surface made from j's twin
        tw = adapters.get(f"{w['id']}x")
        if bool(tw) != bool(adapters.get(w["id"])):   # astra r3: returning None only on twins would fake world-specificity
            invalid(f"world {w['id']}: adapter returned for exactly one of target/twin (the surface must treat both alike)")
        if tw:
            with lora.Merged(model, *tw):
                mism.update({k: v["s"] for k, v in run_tasks(system_text(w["tools"]), p["spec"], p["end"], p["tasks"], BATCH["plain"]).items()})
        else:
            mism.update({t["id"]: base["none"][t["id"]]["s"] for t in p["tasks"]})
    if lora.n_wrapped(model):
        invalid("internal: model left wrapped")

# ---------------------------------------------------------------- metrics
tasks = [t for w in pub for t in priv[w["id"]]["tasks"]]
own_items = {t["id"]: float(own[t["id"]]["s"]) for t in tasks}
success = float(np.mean(list(own_items.values())))
if arm == "icl":
    json.dump(own_items, open(icl_path + ".tmp", "w"))
    os.replace(icl_path + ".tmp", icl_path)
icl = json.load(open(icl_path)) if os.path.exists(icl_path) else None
none_s = float(np.mean([base["none"][t["id"]]["s"] for t in tasks]))
mism_s = float(np.mean([mism[t["id"]] for t in tasks])) if arm == "adapter" else None
icl_s = float(np.mean([icl[t["id"]] for t in tasks])) if icl and set(icl) == set(own_items) else None
b_items = [base["gsm"][x["id"]] for x in gsm] + [base["guard"][t["id"]] for t in guard["tasks"]]
a_items = [bat_gsm[x["id"]] for x in gsm] + [bat_guard[t["id"]] for t in guard["tasks"]]
if not all(math.isfinite(v) for v in bat_text):
    invalid("non-finite text NLL")
m = {"success": success, "none_success": none_s,
     "gain_vs_none": success - none_s,
     "battery_base": float(np.mean(b_items)), "battery": float(np.mean(a_items)),
     "battery_drop": float(np.mean(b_items) - np.mean(a_items)),
     "gsm8k_drop": float(np.mean([base["gsm"][x["id"]] - bat_gsm[x["id"]] for x in gsm])),
     "guardworld_drop": float(np.mean([base["guard"][t["id"]] - bat_guard[t["id"]] for t in guard["tasks"]])),
     "text_nll_ratio": float(np.mean(bat_text) / np.mean(base["text"])),
     "prefill_tokens": float(np.mean([own[t["id"]]["ctx0"] + own[t["id"]]["ctx_retry"] for t in tasks])),
     "initial_context_tokens": float(np.mean([own[t["id"]]["ctx0"] for t in tasks])),
     "retry_prompt_tokens": float(np.mean([own[t["id"]]["ctx_retry"] for t in tasks])),
     "decode_tokens": float(np.mean([own[t["id"]]["dec"] for t in tasks])),
     "inference_tokens": float(np.mean([own[t["id"]]["ctx0"] + own[t["id"]]["ctx_retry"] + own[t["id"]]["dec"] for t in tasks])),
     "retry_rate": float(np.mean([bool(own[t["id"]]["retry"]) for t in tasks])),
     "syntax_error_rate": float(np.mean([bool((own[t["id"]]["first_error"] or "").startswith("syntax")) for t in tasks])),
     "n_adapters": float(sum(has[w] for w in wids)), "n_twin_adapters": float(sum(has[w] for w in tw_ids)), "adapter_params_m": float(max([r["adapter_params_m"] for r in stats["worlds"]] or [0])),
     "adapt_s_mean": float(np.mean([r["adapt_s"] for r in stats["worlds"]])),
     "killed_worlds": float(sum(bool(r["killed"]) for r in passes)),
     "adapt_s_twin_mean": float(np.mean([r["adapt_s"] for r in stats["twins"]])) if stats["twins"] else 0.0,
     "train_tokens_mean": float(np.mean([r["train_tokens"] for r in stats["worlds"]])),
     "gen_tokens_mean": float(np.mean([r["gen_tokens"] for r in stats["worlds"]]))}
if mism_s is not None:
    m["twin_success"], m["world_specific_gain"] = mism_s, success - mism_s
    m["world_specific_share"] = world_specific_share(success, mism_s, none_s)
if icl_s is not None:
    m["icl_success"] = icl_s
    if icl_s - none_s > 0.02:
        m["gap_closure"] = (success - none_s) / (icl_s - none_s)
for tpl in sorted({t["template"] for t in tasks}):
    m[f"s_{tpl}"] = float(np.mean([own_items[t["id"]] for t in tasks if t["template"] == tpl]))
for st in sorted({t.get("stratum") for t in tasks} - {None}):   # v2 fam worlds: familiar / dev / reserved op types
    m[f"s_{st}"] = float(np.mean([own_items[t["id"]] for t in tasks if t.get("stratum") == st]))
    m[f"none_{st}"] = float(np.mean([base["none"][t["id"]]["s"] for t in tasks if t.get("stratum") == st]))
for k in (1, 2):
    sel = [own_items[t["id"]] for t in tasks if min(t["n_calls"], 2) == k]
    if sel:
        m[f"s_calls{k}"] = float(np.mean(sel))
items, primary = own_items, success
write({"valid": True, "primary": primary, "metrics": m, "items": items, "message": f"ok ({arm})"})
print(json.dumps(m))
