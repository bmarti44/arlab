#!/bin/bash
# Manual GPU pilot (owner only; needs an idle GPU): the frozen harness + evaluator on one split, outside a campaign,
# holding the arlab GPU lock so it never overlaps an arlab GPU campaign. Needs the prepared data (arlab check --static).
#   pilot.sh <split> <out-dir> <train-seconds> [seed]
#   KTRAIN="1, 2, 3, 4" KEVAL=4 pilot.sh ...   runs a loop-arm copy of the baseline surface instead
#   B0 (zero-shot):  pilot.sh validation /tmp/ll-b0 0
#   B1 (no loop):    pilot.sh validation /tmp/ll-b1 660
#   loop arm:        KTRAIN="1, 2, 3, 4" KEVAL=4 pilot.sh validation /tmp/ll-k4 660
set -eu
SPLIT=$1; OUT=$2; TS=$3; SEED=${4:-1}
P=/home/bmarti44/arlab/ideas/looped-latent
DATA=${DATA:-$(ls -td /home/bmarti44/arlab-data/looped-latent/*/ | grep -v tmp | head -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-looped-latent:' | head -1)
mkdir -p "$OUT/out" "$OUT/result" "$OUT/surface"; chmod 777 "$OUT/out" "$OUT/result"
cp $P/surface/loop.py "$OUT/surface/loop.py"
if [ -n "${KTRAIN:-}" ]; then
  sed -i -e "s/^LOOPS_TRAIN = (1,) /LOOPS_TRAIN = ($KTRAIN)/" -e "s/^LOOPS_EVAL = 1 /LOOPS_EVAL = ${KEVAL:-4} /" "$OUT/surface/loop.py"
fi
grep -E '^LOOPS_(TRAIN|EVAL)' "$OUT/surface/loop.py"
echo "data $DATA image $IMG; waiting for the arlab GPU lock ..."
exec 9>/home/bmarti44/.cache/arlab/gpu.lock; flock 9
ENV="-e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1"
T0=$(date +%s)
docker run --rm --name arlab-pilot-looped-run --gpus all --network none --shm-size 8g --user 1000:1000 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v $DATA/train:/data/train:ro -v $DATA/$SPLIT/public:/data/public:ro -v "$OUT/out":/out -v /home/bmarti44/.cache/huggingface:/hf:ro \
  $IMG python /frozen/harness.py --out /out --seed $SEED --split $SPLIT --train-seconds $TS 2>&1 | grep -v -E '^(=|NVIDIA|PyTorch|Copyright|$)' | tee "$OUT/run.log"
echo "RUN seconds: $(( $(date +%s) - T0 ))" | tee -a "$OUT/run.log"
docker run --rm --name arlab-pilot-looped-eval --gpus all --network none --user 0:0 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v $P/frozen/eval:/eval:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v "$OUT/out":/run_out:ro -v $DATA/$SPLIT/public:/data/public:ro -v $DATA/$SPLIT/private:/data/private:ro -v "$OUT/result":/result \
  -v /home/bmarti44/.cache/huggingface:/hf:ro $IMG sh -c "python /eval/evaluate.py --run /run_out --out /result/metrics.json; chown -R 1000:1000 /result" | tail -1
