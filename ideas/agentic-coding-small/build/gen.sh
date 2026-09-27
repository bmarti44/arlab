#!/bin/bash
# gen.sh <prefix> <n> "<themes>" : one containerized gpt-6-astra call writing <n> tasks to ~/arlab-data/agentic-gen/<prefix>/tasks
set -u
P=$1; N=$2; THEMES=$3
D=/home/bmarti44/arlab-data/agentic-gen/$P; mkdir -p "$D/tasks"
N=$N P=$P THEMES=$THEMES LEVELS="${LEVELS:-Aim for a mix: ~30% easy, ~40% medium, ~30% hard for such a model.}" python3 -c '
import os, sys
s = open("/home/bmarti44/arlab/ideas/agentic-coding-small/build/" + os.environ.get("PROMPT", "gen_prompt.md")).read()
for k in ("N", "PREFIX", "THEMES", "LEVELS"):
    s = s.replace("__%s__" % k, os.environ["P" if k == "PREFIX" else k])
sys.stdout.write(s)' > "$D/prompt.md"
docker run --rm -i --name "arlab-gen-$P" --user 1000:1000 -e HOME=/tmp \
  -v /home/bmarti44/.cache/arlab/codex-home:/codex -e CODEX_HOME=/codex -v "$D:/out" -w /out \
  arlab-agent:0.157.1 codex exec --ephemeral -C /out --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \
  -m gpt-6-astra -c model_reasoning_effort='"high"' -o /out/summary.txt - < "$D/prompt.md" > "$D/events.log" 2> "$D/stderr.log"
echo "exit $?" > "$D/done"
