#!/bin/bash
# Manual pilot: vLLM service + frozen harness on one split with a logging copy of the baseline surface, then EVALUATE.
# pilot.sh <split> <out-dir> [model] [surface-dir]
set -u
SPLIT=$1; OUT=$2; MODEL=${3:-Qwen/Qwen3.5-4B}; SURF=${4:-/home/bmarti44/arlab/ideas/agentic-coding-small/build/pilot_agent}
P=/home/bmarti44/arlab/ideas/agentic-coding-small; DATA=${DATA:-$(ls -d /home/bmarti44/arlab-data/agentic-coding-small/*/ | grep -v tmp | tail -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-agentic-coding-small:' | head -1)
mkdir -p "$OUT/out" "$OUT/result"; chmod 777 "$OUT/out"
docker network create arlab-pilot-net >/dev/null 2>&1
vcid=$(docker run -d --name arlab-pilot-llm --network arlab-pilot-net --network-alias llm --gpus all --ipc=host \
  -v /home/bmarti44/.cache/huggingface:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve $MODEL --served-model-name llm --port 8000 --max-num-seqs ${SEQS:-16} --gpu-memory-utilization 0.30 --max-model-len 32768)
trap 'docker rm -f $vcid >/dev/null; docker network rm arlab-pilot-net >/dev/null' EXIT
IP=$(docker inspect -f '{{(index .NetworkSettings.Networks "arlab-pilot-net").IPAddress}}' $vcid)
for i in $(seq 1 120); do curl -sf -m 3 http://$IP:8000/health >/dev/null 2>&1 && break; sleep 10; done
curl -s http://$IP:8000/metrics | grep -E '^vllm:(prompt|generation)_tokens_total' > $OUT/metrics0.txt
docker logs $vcid 2>&1 | tail -3
T0=$(date +%s)
docker run --rm --name arlab-pilot-run --network arlab-pilot-net --user 1000:1000 --cpuset-cpus 5-9,15-19 \
  -e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v $P/frozen/run:/frozen:ro -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v $SURF:/work:ro \
  -v $DATA/$SPLIT/public:/data/public:ro -v $OUT/out:/out $IMG python /frozen/harness.py --out /out --seed 1 --split $SPLIT
echo "RUN seconds: $(( $(date +%s) - T0 ))"
curl -s http://$IP:8000/metrics | grep -E '^vllm:(prompt|generation)_tokens_total' > $OUT/metrics1.txt
docker run --rm --network none --user 0:0 -e PYTHONPATH=/frozen:/arlab_lib -v $P/frozen/run:/frozen:ro -v $P/frozen/eval:/eval:ro \
  -v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro -v $OUT/out:/run_out:ro -v $DATA/$SPLIT/private:/data/private:ro -v $OUT/result:/result \
  $IMG python /eval/evaluate.py --run /run_out --out /result/metrics.json
