#!/bin/bash
# Build tooling (run once, by the orchestrator, before the pack was sealed): one containerized Codex call that
# writes one frozen text file of the bank. The outputs were validated by build/assemble.py and frozen into
# frozen/prepare/paraphrases.json; this script is kept for provenance, it is never run by arlab.
#   gen.sh <task> <snapshot-dir>   task in: conventions fillers codetalk frames
#   <snapshot-dir> = the pinned path-limited stencil-llm archive (for vendor/memorycode/topics.json)
set -u
T=$1; SNAP=$2
B=/home/bmarti44/arlab/ideas/stencil-focus/build
D=/tmp/arlab-gen-stencil/$T; mkdir -p "$D"; chmod 777 "$D"
T=$T B=$B SNAP=$SNAP python3 - > "$D/prompt.md" <<'PY'
import json, os
b, t = os.environ["B"], os.environ["T"]
topics = json.load(open(os.environ["SNAP"] + "/vendor/memorycode/topics.json"))
s = open(f"{b}/prompts/{t}.md").read().replace("__COMMON__", open(f"{b}/prompts/common.md").read())
s = s.replace("__FILLERS__", "\n".join(f"- {f['id']}: {json.dumps(f['text'])}" for f in topics["fillers_instruction"]))
canon = [x for i in topics["instructions"] for x in i["text"]]
s += "\n## Canonical texts (never copy 5+ consecutive words from these)\n" + "\n".join("- " + c for c in canon) + "\n"
print(s)
PY
docker run --rm -i --name "arlab-gen-stencil-$T" --user 1000:1000 -e HOME=/tmp \
  -v /home/bmarti44/.cache/arlab/codex-home:/codex -e CODEX_HOME=/codex -v "$D:/out" -w /out \
  arlab-agent:0.157.1 codex exec --ephemeral -C /out --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \
  -m gpt-6-sol -c model_reasoning_effort='"high"' -o /out/summary.txt - < "$D/prompt.md" > "$D/events.log" 2> "$D/stderr.log"
echo "exit $?" > "$D/done"
