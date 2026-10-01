#!/bin/bash
# Manual GPU pilot (owner only; needs an idle GPU): the frozen harness + evaluator for one arm on one split, outside a
# campaign, holding the arlab GPU lock so it never overlaps an arlab GPU campaign. Needs the prepared data (run
# `arlab check --static ideas/plastic-agent` first). No vLLM: everything runs in the pack image with HF transformers.
#   pilot.sh <arm> <out-dir> [seed] [split]     arm = baseline (surface/) | none | icl | placebo (frozen/run/ref_<arm>)
#                                               | custom (SRC_DIR=<dir with adapt.py>, e.g. a campaign's winner)
#   pilot.sh gate <root>                        data-prep gate: runs none + icl on validation seed 1, prints GO / NO-GO
#   ADAPT=180 GEN_TOKENS=400000 TRAIN_TOKENS=1500000 pilot.sh ...   override the per-world adapt budget (calibration)
# Suggested order: gate -> baseline -> placebo (then timing: adapt_s per world in run.log, RUN/EVAL seconds below).
set -eu
P=/home/bmarti44/arlab/ideas/plastic-agent
if [ "$1" = gate ]; then
  ROOT=$2
  "$0" none "$ROOT/none" 1 validation
  "$0" icl "$ROOT/icl" 1 validation
  python3 - "$ROOT" <<'EOF'
import json, sys
r = sys.argv[1]
none, icl = (json.load(open(f"{r}/{a}/result/metrics.json")) for a in ("none", "icl"))
assert none["valid"] and icl["valid"], (none["message"], icl["message"])
n, i = none["primary"], icl["primary"]
ok = n <= 0.15 and i - n >= 0.15
print(f"none={n:.3f} icl={i:.3f} icl-none={i - n:+.3f} (items: {len(none['items'])}); "
      f"prefill tokens: none {none['metrics']['prefill_tokens']:.0f}, icl {icl['metrics']['prefill_tokens']:.0f}")
print("per template (none / icl):", {k[2:]: (round(none['metrics'][k], 2), round(icl['metrics'][k], 2))
                                     for k in sorted(icl["metrics"]) if k.startswith("s_")})
print(f"GATE (none <= 0.15 and icl - none >= 0.15): {'GO' if ok else 'NO-GO'}")
if not ok:
    print("NO-GO: retune frozen/prepare (fauxos.N_OPS / TEMPLATES / VERB_MIX, prepare.N_EXPLORE) and re-run the gate;"
          " do not start a campaign.")
EOF
  exit 0
fi
ARM=$1; OUT=$2; SEED=${3:-1}; SPLIT=${4:-validation}
case $ARM in
  baseline) SRC=$P/surface ;;
  none|icl|placebo) SRC=$P/frozen/run/ref_$ARM ;;
  custom) SRC=${SRC_DIR:?custom arm needs SRC_DIR=<dir with adapt.py>} ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
ADAPT=${ADAPT:-180}; GEN_TOKENS=${GEN_TOKENS:-400000}; TRAIN_TOKENS=${TRAIN_TOKENS:-1500000}
DATA=${DATA:-$(ls -td /home/bmarti44/arlab-data/plastic-agent/*/ | grep -v tmp | head -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-plastic-agent:' | head -1)
CACHE=/home/bmarti44/.cache/arlab/plastic-agent/pilot-eval-cache
mkdir -p "$OUT/out" "$OUT/result" "$OUT/surface" "$CACHE"; chmod 777 "$OUT/out" "$OUT/result"
cp "$SRC/adapt.py" "$OUT/surface/adapt.py"
sha256sum "$OUT/surface/adapt.py"      # the frozen harness/evaluator pick the arm from this hash (common.arm_of)
echo "arm $ARM seed $SEED split $SPLIT; adapt $ADAPT s, gen $GEN_TOKENS, train $TRAIN_TOKENS per world; data $DATA image $IMG"
echo "waiting for the arlab GPU lock ..."
exec 9>/home/bmarti44/.cache/arlab/gpu.lock; flock 9
# Only the model snapshot RUN/EVALUATE use is mounted (not the whole HF cache: no dataset is reachable). A campaign
# mounts the full cache read-only; RUN-side code never references it beyond MODEL_DIR (see REVIEW.md, finding 5).
MODEL_REPO=hub/models--Qwen--Qwen3-1.7B
HFM="-v /home/bmarti44/.cache/huggingface/$MODEL_REPO:/hf/$MODEL_REPO:ro"
ENV="-e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1"
T0=$(date +%s)
docker run --rm --name arlab-pilot-plastic-run --gpus all --network none --shm-size 8g --user 1000:1000 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v $DATA/train:/data/train:ro -v $DATA/$SPLIT/public:/data/public:ro -v "$OUT/out":/out $HFM \
  $IMG python /frozen/harness.py --out /out --seed $SEED --split $SPLIT --adapt-seconds $ADAPT --gen-tokens $GEN_TOKENS \
  --train-tokens $TRAIN_TOKENS 2>&1 | grep -v -E '^(=|NVIDIA|PyTorch|Copyright|$)' | tee "$OUT/run.log"
T1=$(date +%s)
docker run --rm --name arlab-pilot-plastic-eval --gpus all --network none --shm-size 8g --user 0:0 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v $P/frozen/eval:/eval:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v "$OUT/out":/run_out:ro -v $DATA/$SPLIT/public:/data/public:ro -v $DATA/$SPLIT/private:/data/private:ro -v "$OUT/result":/result \
  -v "$CACHE":/cache $HFM \
  $IMG sh -c "python /eval/evaluate.py --run /run_out --out /result/metrics.json; chown -R 1000:1000 /result" 2>&1 | tail -1
T2=$(date +%s)
echo "RUN seconds: $((T1 - T0))  EVALUATE seconds: $((T2 - T1))  (the first EVALUATE per split also fills the base-model cache)" | tee -a "$OUT/run.log"
python3 -c "import json; m = json.load(open('$OUT/result/metrics.json')); print('valid', m['valid'], 'primary', m['primary'], m['message'])"
