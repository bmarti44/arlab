# stencil-focus — awaiting the owner's IDEA.md

This pack is waiting for `ideas/stencil-focus/IDEA.md` (the stencil-llm question to study; PLAN §6.4).
Once it exists, the orchestrator builds the pack through docs/ORCHESTRATOR.md like any other idea.

Constraints that will apply: `~/stencil-llm` (GPL-2.0) is used only as a pinned snapshot copied during PREPARE
(`git -C ~/stencil-llm archive <sha> | tar -x`), never modified; arlab never evaluates on items stencil-llm treats
as registered, SCREEN, TRAIN, final or holdout pools; stencil's FROZEN files are respected; the pack wraps GPL-2.0
code for local use only.
