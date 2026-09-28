"""Pilot report + go/no-go for stencil-focus (build/pilot.sh calls it; stdlib only).

  python3 pilot_report.py <pilot-out-dir> <cap-used>

Rule (IDEA.md, fixed before the pilot): build only if oracle - baseline >= 0.10 (2 x MES) AND a 300-item run
projects to <= 20 min; else try 250/250; else reconsider. The pilot also fixes the generation cap (smallest of
512 / 768 / 1,024 whose cap-hit rate is <= 5% in every arm, from the completion lengths of this run) and
run.timeout_s (2 x the projected 300-item + 28-real RUN time, rounded up to a minute).
"""
import json
import math
import sys
from pathlib import Path

OUT, CAP = Path(sys.argv[1]), int(sys.argv[2])
ARMS = ["off", "baseline", "evicted_1024", "oracle"]
MES, GATE_DIFF, GATE_MIN = 0.05, 0.10, 20.0
res = {}
for arm in ARMS:
    m = json.loads((OUT / arm / "result" / "metrics.json").read_text())
    t = json.loads((OUT / arm / "timing.json").read_text())
    rows = [json.loads(line) for line in open(OUT / arm / "out" / "outputs.jsonl")]
    comp = [r.get("completion_tokens", 0) for r in rows if r.get("kind") == "main"]
    res[arm] = {"valid": m["valid"], "primary": m["primary"], "items": m["items"] or {}, "metrics": m["metrics"],
                "run_s": t["run_s"], "gen_tokens": t["generation_tokens"], "prompt_tokens": t["prompt_tokens"],
                "n": len(comp), "comp": comp}


def paired(a, b):
    ia, ib = res[a]["items"], res[b]["items"]
    d = [ia[k] - ib[k] for k in sorted(ia) if k in ib]
    mu = sum(d) / len(d)
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else float("nan")
    return {"diff": mu, "sd": sd, "two_se": 2 * sd / math.sqrt(len(d)), "n": len(d)}


rep = {"cap_used": CAP, "arms": {}, "vs_baseline": {}}
for arm, r in res.items():
    hit = {c: sum(x >= c for x in r["comp"]) / max(r["n"], 1) for c in sorted({512, 768, 1024, CAP}) if c <= CAP}
    rep["arms"][arm] = {"valid": r["valid"], "fraction_required": r["primary"],
                        "output_failure_rate": r["metrics"].get("output_failure_rate"),
                        "invalid_code_rate": r["metrics"].get("invalid_code_rate"),
                        "cap_hit_at": hit, "completion_mean": sum(r["comp"]) / max(r["n"], 1),
                        "run_s": round(r["run_s"], 1), "min_per_100": round(r["run_s"] / 60 * 100 / max(r["n"], 1), 2),
                        "decode_tok_s": round(r["gen_tokens"] / max(r["run_s"], 1e-9), 1),
                        "reminder_tokens_mean": r["metrics"].get("reminder_tokens_mean")}
for arm in ("oracle", "off", "evicted_1024"):
    rep["vs_baseline"][arm] = paired(arm, "baseline")
per100 = max(a["min_per_100"] for a in rep["arms"].values())
rep["projection_min"] = {"300": round(per100 * 3.28, 1), "250": round(per100 * 2.78, 1)}   # + the 28 real items (holdout)
headroom = rep["vs_baseline"]["oracle"]["diff"]
caps = [c for c in sorted({512, 768, 1024, CAP}) if c <= CAP and all(a["cap_hit_at"].get(c, 1) <= 0.05 for a in rep["arms"].values())]
rep["cap_recommended"] = caps[0] if caps else CAP
rep["timeout_s_recommended"] = max(1200, int(math.ceil(2 * rep["projection_min"]["300"]) * 60))
if not caps:
    rep["decision"] = f"NO-GO: every arm hits the {CAP}-token cap on > 5% of items; raise the cap or reconsider"
elif headroom >= GATE_DIFF and rep["projection_min"]["300"] <= GATE_MIN:
    rep["decision"] = "GO: 300/300"
elif headroom >= GATE_DIFF and rep["projection_min"]["250"] <= GATE_MIN:
    rep["decision"] = "GO: 250/250 (set N_DIALOGUES in frozen/prepare/bank.py, budget.limit accordingly)"
else:
    rep["decision"] = "NO-GO: reconsider (oracle - baseline {:.3f}, need >= {}; projection 300/250 items {}/{} min, need <= {})".format(
        headroom, GATE_DIFF, rep["projection_min"]["300"], rep["projection_min"]["250"], GATE_MIN)
if caps and rep["cap_recommended"] < CAP:
    rep["note"] = (f"the recommended cap {rep['cap_recommended']} is below the cap used ({CAP}): re-run "
                   f"build/pilot.sh <dir> {rep['cap_recommended']} to confirm headroom and failure rates at that cap")
base_fail = rep["arms"]["baseline"]["output_failure_rate"] or 0
if base_fail < 0.02:
    rep["warning"] = ("baseline output_failure_rate < 0.02: the 1.15x ratio guard would reject almost any new failure; "
                      "replace it by an absolute guard (e.g. {name: output_failure_rate, max: 0.05}) before sealing")
(OUT / "pilot_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1))
for arm, a in rep["arms"].items():
    print(f"{arm:13s} fr={a['fraction_required']:.3f} fail={a['output_failure_rate']:.3f} cap_hit={a['cap_hit_at']} "
          f"comp={a['completion_mean']:.0f} run={a['run_s']}s ({a['min_per_100']} min/100) decode={a['decode_tok_s']} tok/s")
o = rep["vs_baseline"]["oracle"]
print(f"\nDECISIONS.md line: stencil-focus pilot ({o['n']} pilot dialogues, cap {CAP}): oracle - baseline = {o['diff']:+.3f} "
      f"(sd {o['sd']:.3f}, 2SE {o['two_se']:.3f}); off - baseline = {rep['vs_baseline']['off']['diff']:+.3f}; "
      f"evicted_1024 - baseline = {rep['vs_baseline']['evicted_1024']['diff']:+.3f}; projection 300 items "
      f"{rep['projection_min']['300']} min -> {rep['decision']}; cap {rep['cap_recommended']}, run.timeout_s "
      f"{rep['timeout_s_recommended']}.")
