#!/bin/bash
# Manual GPU pilot (owner/orchestrator only; needs an idle GPU): the frozen harness + evaluator for one arm on one
# split, outside a campaign, holding the arlab GPU lock. Needs the prepared data (arlab check --static) and the image.
#   pilot.sh <arm> <out-dir> [train-seconds=330] [seed=1] [split=validation]
#   arms: baseline (surface/), depth12 (frozen/run/ref_depth12), loop_naive (build/loop_naive)
#   depth12_2x = depth12 with train-seconds 660 (then budget.json exceeds 360: fine outside a campaign)
set -eu
ARM=$1; OUT=$2; TS=${3:-330}; SEED=${4:-1}; SPLIT=${5:-validation}
P=/home/bmarti44/arlab/ideas/latent-arch
case $ARM in
  baseline) SRC=$P/surface ;; depth12) SRC=$P/frozen/run/ref_depth12 ;; loop_naive) SRC=$P/build/loop_naive ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
DATA=${DATA:-$(ls -td /home/bmarti44/arlab-data/latent-arch/*/ | grep -v tmp | head -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-latent-arch:' | head -1)
mkdir -p "$OUT/out" "$OUT/result" "$OUT/surface"; chmod 777 "$OUT/out" "$OUT/result"
cp "$SRC"/model.py "$SRC"/train.py "$OUT/surface/"
echo "arm $ARM data $DATA image $IMG; waiting for the arlab GPU lock ..."
exec 9>/home/bmarti44/.cache/arlab/gpu.lock; flock 9
ENV="-e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
T0=$(date +%s)
docker run --rm --name arlab-pilot-latent-run --gpus all --network none --shm-size 8g --user 1000:1000 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v $DATA/train:/data/train:ro -v "$OUT/out":/out \
  $IMG python /frozen/harness.py --out /out --seed $SEED --split $SPLIT --train-seconds $TS 2>&1 | grep -v -E '^(=|NVIDIA|PyTorch|Copyright|$)' | tee "$OUT/run.log"
echo "RUN seconds: $(( $(date +%s) - T0 ))" | tee -a "$OUT/run.log"
T1=$(date +%s)
docker run --rm --name arlab-pilot-latent-eval --gpus all --network none --user 0:0 -w /tmp $ENV \
  -v $P/frozen/run:/frozen:ro -v $P/frozen/eval:/eval:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v "$OUT/surface":/work:ro \
  -v "$OUT/out":/run_out:ro -v $DATA/$SPLIT/private:/data/private:ro -v "$OUT/result":/result \
  $IMG sh -c "python /eval/evaluate.py --run /run_out --out /result/metrics.json --max-train-seconds 100000; chown -R 1000:1000 /result" | tail -1
echo "EVALUATE seconds: $(( $(date +%s) - T1 ))"
