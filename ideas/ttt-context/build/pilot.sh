#!/bin/bash
# Manual GPU pilot (owner only; needs an idle GPU): the frozen harness + evaluator on one split, outside a campaign,
# holding the arlab GPU lock so it never overlaps an arlab GPU campaign. Needs the prepared data (arlab check --static).
#   pilot.sh <split> <out-dir> <arm> <ttt-seconds> [seed] [limit]
#   arm: baseline (surface/ttt.py) | no_ttt | no_doc (frozen/run/ref_<arm>/ttt.py)
#   STEPS=32 LR=1e-3 SPAN=128 pilot.sh ... baseline ...   overrides constants of the baseline surface (e.g. a qTTT-like arm)
#   timing (40 items):  pilot.sh validation /tmp/ttt-pilot/base40 baseline 4 1 40
#   arms (400 items):   pilot.sh validation /tmp/ttt-pilot/no_ttt no_ttt 4
set -eu
SPLIT=$1; OUT=$2; ARM=$3; TS=$4; SEED=${5:-1}; LIMIT=${6:-0}
P=/home/bmarti44/arlab/ideas/ttt-context
DATA=${DATA:-$(ls -td /home/bmarti44/arlab-data/ttt-context/*/ | grep -v tmp | head -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-ttt-context:' | head -1)
mkdir -p "$OUT/out" "$OUT/result" "$OUT/surface"; chmod 777 "$OUT/out" "$OUT/result"
case $ARM in
  baseline) cp $P/surface/ttt.py "$OUT/surface/ttt.py" ;;
  no_ttt|no_doc) cp $P/frozen/run/ref_$ARM/ttt.py "$OUT/surface/ttt.py" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
for K in STEPS LR SPAN TARGETS DOC_IN_CONTEXT; do
  V=${!K:-}
  if [ -n "$V" ]; then sed -i "s/^$K = [^#]*/$K = $V  /" "$OUT/surface/ttt.py"; fi
done
grep -E '^(DOC_IN_CONTEXT|PREFILL_DOC|TARGETS|STEPS|SPAN|LR) ' "$OUT/surface/ttt.py" || true
echo "data $DATA image $IMG; waiting for the arlab GPU lock ..."
exec 9>/home/bmarti44/.cache/arlab/gpu.lock; flock 9
ENV="-e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1"
T0=$(date +%s)
docker run --rm --name arlab-pilot-ttt-run --gpus all --network none --shm-size 8g --user 1000:1000 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v $DATA/$SPLIT/public:/data/public:ro -v "$OUT/out":/out -v /home/bmarti44/.cache/huggingface:/hf:ro \
  $IMG python /frozen/harness.py --out /out --seed $SEED --split $SPLIT --ttt-seconds $TS --limit $LIMIT 2>&1 \
  | grep -v -E '^(=|NVIDIA|PyTorch|Copyright|$)' | tee "$OUT/run.log"
echo "RUN seconds: $(( $(date +%s) - T0 ))" | tee -a "$OUT/run.log"
docker run --rm --name arlab-pilot-ttt-eval --network none --user 0:0 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v $P/frozen/eval:/eval:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro \
  -v "$OUT/out":/run_out:ro -v $DATA/$SPLIT/public:/data/public:ro -v $DATA/$SPLIT/private:/data/private:ro -v "$OUT/result":/result \
  -v /home/bmarti44/.cache/huggingface:/hf:ro $IMG \
  sh -c "python /eval/evaluate.py --run /run_out --out /result/metrics.json --ttt-seconds $TS --limit $LIMIT; chown -R 1000:1000 /result" | tail -1
