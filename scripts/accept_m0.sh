#!/usr/bin/env bash
# accept-M0: bring-up. --cpu-only skips the GPU/vLLM parts (allowed while the GPU is busy).
set -euo pipefail
CPU_ONLY=0; [[ "${1:-}" == "--cpu-only" ]] && CPU_ONLY=1
cd "$(dirname "$0")/.."
fail() { echo "FAIL: $*"; exit 1; }
ok() { echo "ok: $*"; }

[[ -f CLAUDE.md && -f .claude/settings.json && -f docs/spark-notes.md ]] || fail "CLAUDE.md / settings / spark-notes missing"
for r in 'Bash(docker:*)' 'Bash(systemd-run:*)' 'Bash(make:*)' 'Bash(uv:*)' 'Bash(git:*)' 'Bash(nvidia-smi:*)'; do
  grep -qF "$r" .claude/settings.json || fail "allow rule $r missing"; done
ok "CLAUDE.md, .claude/settings.json, docs/spark-notes.md"
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); (( avail >= 60 )) || fail "only ${avail}G free"
ok "disk ${avail}G free"
docker image inspect arlab-agent:0.157.1 >/dev/null || fail "arlab-agent:0.157.1 missing"
[[ "$(docker run --rm arlab-agent:0.157.1 codex --version)" == *0.157.1* ]] || fail "agent image codex version"
ok "arlab-agent:0.157.1"

if (( ! CPU_ONLY )); then
  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp nvcr.io/nvidia/pytorch:25.10-py3 python -c "
import torch, torch.nn.functional as F
assert torch.cuda.get_device_capability() == (12, 1), torch.cuda.get_device_capability()
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16); (a @ a).sum().item()
q = torch.randn(2, 4, 256, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward(); torch.cuda.synchronize(); print('gpu-ok')
" 2>/dev/null | grep -q gpu-ok || fail "pytorch GPU check"
  ok "pytorch uid 1000: sm_121, bf16 matmul, SDPA fwd/bwd"
  vcid=$(docker run -d --name arlab-m0-accept-vllm --gpus all --ipc=host -p 127.0.0.1:18001:8000 \
    -v "$HOME/.cache/huggingface:/hf:ro" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 nvcr.io/nvidia/vllm:26.04-py3 \
    vllm serve Qwen/Qwen3.5-4B --gpu-memory-utilization 0.35 --max-model-len 4096 --port 8000) || fail "vLLM start"
  trap 'docker rm -f "$vcid" >/dev/null 2>&1 || true' EXIT   # only the container this script started
  for i in $(seq 1 60); do curl -sf -m 3 localhost:18001/health >/dev/null && break; sleep 10; done
  out=$(curl -sf localhost:18001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}') || fail "vLLM completion"
  [[ "$out" == *5* ]] || fail "vLLM answer: $out"
  ok "vLLM 26.04 Qwen3.5-4B @0.35 completion"
fi

# three containerized codex calls with the exact §3.6 command must return schema-valid JSON
owner_before=$(codex login status 2>&1 | grep -i "logged in" || true)   # ignore update notices etc.
for n in 1 2 3; do
  R=$(mktemp -d /tmp/arlab-m0-codex-XXXX); mkdir -p "$R/view" "$R/agent"
  printf 'LR = 0.1\n' > "$R/view/train.py"
  printf 'Make ONE change to train.py in the current directory: set LR to 0.0%s. Then finish with the JSON object required by the output schema (action "edit", a short description, hypothesis_tag "lr", constraint_learned null) and nothing after it.\n' "$n" > "$R/view/prompt.md"
  docker run --rm -i --name "arlab-m0-agent-$n" --user 1000:1000 -e HOME=/tmp \
    -v "$HOME/.cache/arlab/codex-home:/codex" -e CODEX_HOME=/codex \
    -v "$R/view:/work" -v "$R/agent:/out" -v "$PWD/arlab/proposal.schema.json:/schema.json:ro" \
    arlab-agent:0.157.1 codex exec --ephemeral -C /work --skip-git-repo-check \
      --dangerously-bypass-approvals-and-sandbox -m gpt-6-sol -c model_reasoning_effort='"high"' \
      --output-schema /schema.json -o /out/proposal.json --json - \
    < "$R/view/prompt.md" > "$R/agent/events.jsonl" || fail "codex call $n exit code"
  .venv/bin/python -c "
import json, sys; from arlab.agent import parse_proposal
p = parse_proposal(json.load(open('$R/agent/proposal.json'))); assert p.action == 'edit', p
assert '0.0$n' in open('$R/view/train.py').read()" || fail "codex call $n: proposal not schema-valid / no edit"
  ok "codex call $n schema-valid"
done
owner_after=$(codex login status 2>&1 | grep -i "logged in" || true)
[[ "$owner_after" == "$owner_before" ]] || fail "owner's codex login status changed: '$owner_before' -> '$owner_after'"
ok "owner's codex login unaffected"
echo "accept-M0: PASS$([[ $CPU_ONLY == 1 ]] && echo ' (cpu-only)')"
