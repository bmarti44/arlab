#!/bin/bash
# gen.sh <prefix> <n> "<themes>" : one containerized gpt-6-astra call writing <n> tasks to ~/arlab-data/agentic-gen/<prefix>/tasks
set -u
P=$1; N=$2; THEMES=$3
D=/home/bmarti44/arlab-data/agentic-gen/$P; mkdir -p "$D/tasks"
sed -e "s/__N__/$N/g" -e "s/__PREFIX__/$P/g" -e "s|__THEMES__|$THEMES|" /home/bmarti44/arlab/ideas/agentic-coding-small/build/gen_prompt.md > "$D/prompt.md"
docker run --rm -i --name "arlab-gen-$P" --user 1000:1000 -e HOME=/tmp \
  -v /home/bmarti44/.cache/arlab/codex-home:/codex -e CODEX_HOME=/codex -v "$D:/out" -w /out \
  arlab-agent:0.157.1 codex exec --ephemeral -C /out --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \
  -m gpt-6-astra -c model_reasoning_effort='"high"' -o /out/summary.txt - < "$D/prompt.md" > "$D/events.log" 2> "$D/stderr.log"
echo "exit $?" > "$D/done"
