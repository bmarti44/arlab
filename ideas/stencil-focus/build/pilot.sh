#!/bin/bash
# Pre-seal pilot (owner only; needs an idle GPU and the prepared data from `arlab check --static`).
# Headroom + timing on the 60-dialogue PILOT slice (in neither split; own chatter and templates, both pivot sides).
# Arms: off, baseline, evicted_1024, oracle (label-derived, <= 1,024 tokens), each = frozen harness + EVALUATE.
# Holds the arlab GPU lock, runs its own vLLM exactly as pack.yaml's service (mem_gb 30 -> 0.25, max_model_len 8192),
# then prints the go/no-go (build/pilot_report.py): oracle - baseline >= 0.10 and a 300-item run <= 20 min.
#   pilot.sh <out-dir> [max-new=1024]
set -euo pipefail
OUT=$1; CAP=${2:-1024}
P=/home/bmarti44/arlab/ideas/stencil-focus; HF=/home/bmarti44/.cache/huggingface
DATA=${DATA:-$(ls -td /home/bmarti44/arlab-data/stencil-focus/*/ | grep -v tmp | head -1)}
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^arlab-stencil-focus:' | head -1)
REV=1cfa9a7208912126459214e8b04321603b3df60c
NET=arlab-pilot-stencil-net; SVC=arlab-pilot-stencil-llm
mkdir -p "$OUT/plans"; chmod 777 "$OUT/plans"; rm -f "$OUT/plans/oracle_plans.json"
echo "data $DATA image $IMG cap $CAP; waiting for the arlab GPU lock ..."
exec 9>/home/bmarti44/.cache/arlab/gpu.lock; flock 9
ENV="-e PYTHONPATH=/frozen:/arlab_lib -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HOME=/hf -e HF_HUB_OFFLINE=1"
LIB="-v /home/bmarti44/arlab/arlab:/arlab_lib/arlab:ro"

docker run --rm --name arlab-pilot-stencil-oracle --network none --user 0:0 -w /tmp $ENV -v $P/frozen/run:/frozen:ro $LIB \
  -v $P/build:/pack_build:ro -v $DATA/pilot/public:/data/public:ro -v $DATA/pilot/private:/data/private:ro \
  -v "$OUT/plans":/out -v $HF:/hf:ro $IMG sh -c "python /pack_build/oracle_plans.py && chown -R 1000:1000 /out" 2>&1 | tail -1

docker network create $NET >/dev/null 2>&1 || true
vcid=$(docker run -d --name $SVC --network $NET --network-alias llm --gpus all --ipc=host -v $HF:/hf:ro -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 nvcr.io/nvidia/vllm:26.04-py3 sh -c "vllm serve Qwen/Qwen3-4B --revision $REV --served-model-name llm \
  --port 8000 --max-num-seqs 64 --enable-prefix-caching --gpu-memory-utilization 0.25 --max-model-len 8192")
trap 'docker rm -f $vcid >/dev/null 2>&1; docker network rm $NET >/dev/null 2>&1' EXIT
IP=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$NET\").IPAddress}}" $vcid)
for i in $(seq 1 180); do curl -sf -m 3 http://$IP:8000/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf -m 3 http://$IP:8000/health >/dev/null || { docker logs --tail 40 $vcid; exit 1; }
tokens() { curl -s http://$IP:8000/metrics | awk '/^vllm:(prompt|generation)_tokens_total/ {s[$1 ~ /prompt/ ? "p" : "g"] += $2} END {printf "%d %d", s["p"], s["g"]}'; }

for ARM in off baseline evicted_1024 oracle; do
  case $ARM in
    off) SURF=$P/frozen/run/ref_off; EXTRA="";;
    baseline) SURF=$P/surface; EXTRA="";;
    evicted_1024) SURF=$P/frozen/run/ref_evicted_1024; EXTRA="";;
    oracle) SURF=$P/surface; EXTRA="--plans /plans/oracle_plans.json";;
  esac
  A=$OUT/$ARM; rm -rf "$A"; mkdir -p "$A/out" "$A/result"; chmod 777 "$A/out" "$A/result"
  read P0 G0 <<< "$(tokens)"; T0=$(date +%s.%N)
  test -s "$OUT/plans/oracle_plans.json"
  docker run --rm --name arlab-pilot-stencil-run --network $NET --user 1000:1000 -w /tmp $ENV -v $P/frozen/run:/frozen:ro $LIB \
    -v $SURF:/work:ro -v $DATA/pilot/public:/data/public:ro -v "$OUT/plans":/plans:ro -v "$A/out":/out -v $HF:/hf:ro \
    $IMG python /frozen/harness.py --out /out --seed 1 --split pilot --max-new $CAP $EXTRA 2>&1 | tail -1 | tee "$A/run.log"
  T1=$(date +%s.%N); read P1 G1 <<< "$(tokens)"
  echo "{\"run_s\": $(echo "$T1 - $T0" | bc), \"prompt_tokens\": $((P1 - P0)), \"generation_tokens\": $((G1 - G0))}" > "$A/timing.json"
  docker run --rm --name arlab-pilot-stencil-eval --network none --user 0:0 -w /tmp $ENV -v $P/frozen/run:/frozen:ro \
    -v $P/frozen/eval:/eval:ro $LIB -v $SURF:/work:ro -v "$A/out":/run_out:ro -v $DATA/pilot/public:/data/public:ro \
    -v $DATA/pilot/private:/data/private:ro -v "$A/result":/result -v $HF:/hf:ro $IMG \
    sh -c "python /eval/evaluate.py --run /run_out --out /result/metrics.json --max-new $CAP && chown -R 1000:1000 /result" 2>&1 | tail -1
done
python3 $P/build/pilot_report.py "$OUT" "$CAP"
