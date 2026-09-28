#!/bin/bash
# GPU trace sampling for ttc-controller (IDEA.md "Sampling happens outside arlab's PREPARE"). Launch it detached;
# arlab never runs it. It holds the arlab GPU lock and waits while other work uses the GPU.
#   L=$HOME/arlab-runs/ttc-sample.log
#   pilot (100 train problems, prints the gate table):
#     systemd-run --user --unit=arlab-ttc-sample --collect -p StandardOutput=append:$L -p StandardError=append:$L \
#       /home/bmarti44/arlab/ideas/ttc-controller/build/sample.sh --pilot 100
#   full cache (resumes, keeps the pilot shard):      same command without --pilot
#   cuts if the pilot projects > 6 h:                 ... sample.sh --splits train:400,validation:800,holdout
#   transfer cache (after FINALIZE, outside the pack):
#     OUT=$HOME/arlab-data/ttc-controller/traces-qwen35-2b ... sample.sh --model Qwen/Qwen3.5-2B@15852e8c --n 16 --splits train:300,holdout
# Env: OUT (default frozen/prepare/traces), NEED_GB (MemAvailable to wait for, default 40),
#      SHARE_GPU=1 (start next to other GPU processes if memory allows; default waits for an idle GPU).
set -euo pipefail
P=/home/bmarti44/arlab/ideas/ttc-controller
OUT=${OUT:-$P/frozen/prepare/traces}
HF=/home/bmarti44/.cache/huggingface
VLLM_IMG=nvcr.io/nvidia/vllm:26.04-py3
NEED_GB=${NEED_GB:-40}
mkdir -p "$OUT"

# 1. a synthetic cache (build/fake_cache.py) is deleted first; a real one is resumed by sample.py
if [ -f "$OUT/MANIFEST.json" ] && grep -q '"synthetic": true' "$OUT/MANIFEST.json"; then
  echo "$(date -Is) removing the synthetic cache in $OUT"
  rm -f "$OUT"/*.npz "$OUT"/*.texts.json.gz "$OUT/MANIFEST.json" "$OUT/problems.jsonl"
fi

# 2. the frozen problem list (CPU, in the pack image that `arlab check --static` built)
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-ttc-controller:' | head -1)
[ -n "$IMG" ] || { echo "no arlab-ttc-controller image: run arlab check --static ideas/ttc-controller first"; exit 2; }
docker run --rm --name arlab-ttc-sample-list --network none --user 1000:1000 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v $P/frozen/prepare:/prepare:ro -v $HF:/hf:ro -v "$OUT":/traces "$IMG" \
  python /prepare/gsm8k.py --list --out /traces/problems.jsonl --hf /hf 2>&1 | tail -1

# 3. the arlab GPU lock (no arlab GPU campaign runs meanwhile), then wait for a free GPU and enough memory
exec 9>/home/bmarti44/.cache/arlab/gpu.lock
echo "$(date -Is) waiting for the arlab GPU lock"
flock 9
while :; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if { [ "$apps" -eq 0 ] || [ "${SHARE_GPU:-0}" = 1 ]; } && [ "$avail" -ge "$NEED_GB" ]; then break; fi
  echo "$(date -Is) waiting_gpu: $apps compute processes, MemAvailable $avail GB (need $NEED_GB)"
  sleep 300
done

# 4. sample into resumable shards + MANIFEST.json
DIGEST=$(docker image inspect -f '{{.Id}}' $VLLM_IMG)
echo "$(date -Is) sampling with $VLLM_IMG ($DIGEST) into $OUT: $*"
docker run --rm --name arlab-ttc-sample --gpus all --shm-size 8g --network none --user 1000:1000 -w /tmp \
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e VLLM_IMAGE="$VLLM_IMG@$DIGEST" \
  -v $P/frozen/prepare:/prepare:ro -v $HF:/hf:ro -v "$OUT":/traces $VLLM_IMG \
  python /prepare/sample.py --problems /traces/problems.jsonl --out /traces "$@"
echo "$(date -Is) done; next: arlab check --static ideas/ttc-controller (PREPARE prints suggested_budget_tokens)"
