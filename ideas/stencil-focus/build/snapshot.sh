#!/bin/bash
# Host-side half of the pinned stencil-llm snapshot (run once by the orchestrator; PREPARE containers cannot see
# ~/stencil-llm). Exactly the path-limited archive IDEA.md specifies, gzip'd without timestamps, into
# frozen/prepare/, where PREPARE verifies its sha256 and the commit id git stores in the tar header before use.
# The tarball is GPL-2.0 (stencil-llm) + Apache-2.0 (MemoryCode) material for local use only; it is git-ignored.
set -eu
SHA=87b72a88cd56712a865359afd794a34b66f3d05d
OUT=/home/bmarti44/arlab/ideas/stencil-focus/frozen/prepare/stencil-$SHA.tar.gz
git -C /home/bmarti44/stencil-llm archive $SHA \
  src/stencil/__init__.py src/stencil/memorycode.py src/stencil/focus3.py src/stencil/focus2.py src/stencil/stats.py \
  vendor/memorycode data/g0/chat.jsonl | gzip -n -9 > "$OUT"
sha256sum "$OUT"
