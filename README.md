# arlab: an AI research lab that runs on one desktop computer

**arlab lets an AI agent do machine-learning research on its own, and then tells you honestly whether anything it
found is real.**

You give it an idea in plain English, such as "can a small language model think longer if it loops its middle
layers?". arlab turns the idea into an experiment. An AI coding agent then tries dozens of variations, one after
another or many at once. arlab runs every attempt on the local GPU and scores it with tests the agent never sees.
At the end it writes a verdict with the evidence behind it: **supported**, **not found at this scale**, or
**inconclusive**.

Everything runs on a single NVIDIA DGX Spark (a desktop-sized machine with one GPU), unattended, for hours or days.

---

## Why this is interesting

**Trying ideas has become cheap. Trusting the results has not.** AI agents can now write and run experiments all
day. The hard part is knowing which "improvements" are real. With enough attempts, some will look better by pure
luck. Some will quietly cheat, for example by reading the answer key or training longer than allowed. And the best
of many noisy results is almost always an overestimate.

arlab's main job is to make the results trustworthy, automatically:

- **It measures the noise first.** Before any experiment, it runs the unchanged starting code several times to see
  how much scores wobble by chance. An improvement only counts if it clearly beats that wobble.
- **It keeps a hidden exam.** Part of the data and some random seeds are locked away. They are used exactly once, at
  the very end, to check the final result. The agent can't tune itself to data it never sees.
- **It sets the bar before the race.** Each experiment declares in advance the smallest improvement worth caring
  about. The bar can't move after the fact.
- **It blocks cheating.** The agent works in a sealed container. It can only edit the files it is allowed to edit,
  can't see the grader or the answers, and runs under a fixed compute budget. Anything that breaks these rules is
  thrown out.
- **It reports nulls.** "We tried 20 ideas and none of them made a meaningful difference" is a real, useful result.
  arlab says so plainly, and says how big an effect it could have missed.

## New: it also learns *how to search*

Most AI-driven research loops are greedy: take the best result so far, tweak it, repeat. That gets stuck easily.
arlab can now run a **tree search**, inspired by Google DeepMind's **Dream-RSI** (September 2026). It works in three
steps:

1. **Explore many paths at once.** Every attempt becomes a branch point. The next attempt can refine any earlier
   attempt or start a fresh direction, so many lines of work stay alive at the same time.
2. **Look back at everything that was tried.** A finished search is a record of which paths led where. arlab can
   "re-play" that record to test a different strategy for free: what would have happened if we had followed other
   branches or stopped sooner?
3. **Improve the search strategy itself.** The agent reads those replays and rewrites the small program that decides
   *which paths to try next and when to stop*. The best version is used for the next real search.

The AI model's weights never change here. What improves is the lab's research strategy. The paper left some gaps:
no error bars, no testing on searches the strategy hadn't seen, no check that replays predict reality, and no
released code. arlab closes them:
- a new strategy is only adopted if it also does better on searches it has never seen;
- every "smarter" search is run side by side with a plain one on the same problem;
- the final answer still goes through the same single hidden-exam check.

Details: [docs/TREE-SEARCH.md](docs/TREE-SEARCH.md).

## How it works

```
  your idea (plain English)
        │
        ▼
  an experiment "pack": starting code the agent may edit + a frozen, hidden grader + a budget
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  the agent proposes one change  →  arlab runs it on the GPU           │
  │  →  the hidden grader scores it  →  compare against measured noise    │  × dozens of attempts
  │  (greedy: keep or discard; tree: add a branch, a strategy picks next) │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  the final check on the locked-away data, once  →  verdict + a written report
```

The AI agent is OpenAI's Codex CLI (GPT-6), always inside a container. The runner is about 1,500 lines of Python
that owns the loop, the statistics and the record-keeping.

## What it has found so far

| Question | Answer | What it means |
|---|---|---|
| *Lab self-test:* can arlab find a real improvement when one exists? (training a small GPT) | **Yes.** It kept 1 of 28 changes (a smaller optimizer batch) and confirmed it on hidden data | The machinery works end to end. The change is a known trick, not a discovery. |
| *Lab self-test:* does arlab say "nothing" when there's no room to improve? (digit recognition that is already 96% accurate) | **Yes, a null.** | The lab doesn't invent wins. |
| Can smarter memory bookkeeping help a small model answer questions about very long chat histories? | **Not found at this scale.** 25 designs; none gained the 8 points we were looking for. | Clever retrieval around a small model hit a ceiling here. |
| Can a small model "think longer" internally by looping its middle layers, instead of writing its reasoning out? | **Not found at this scale.** 20 loop designs on a 0.6B model; any real gain is under 2.5 points (the bar was 3). | Bolting loops onto an already-trained small model didn't help. Training with loops from scratch is the next test. |
| Can better agent scaffolding (planning, self-checks, tools, context summaries) make a small 4B model a better coder? | **Not found at this scale.** After fixing the lab's timing problem, 12 scaffold ideas all landed within noise (12.5–22.5% vs a 15–25% baseline). | This test can only detect big wins (+20 points); nothing that large exists among simple scaffold tweaks for this model. |
| *Lab self-test:* can the lab learn a better **search strategy** by studying its own past searches? (tree search on a toy problem, 336 attempts) | **Not shown yet.** Learned strategies found slightly less while using a fifth of the attempts; run-to-run noise was as large as the difference. | The machinery works, including the check that rejects strategies that only fit old searches. The real test is on a harder problem (next: small-GPT training). |
| Can tree search improve a small language model's training *beyond* what the step-by-step lab already found? And can the lab learn a better search strategy? | **Yes to the first, not shown for the second.** All 4 search rounds beat the earlier best on hidden text, by 0.0125–0.0177 bits per byte (the bar was 0.010). The greedy strategy did best. The *learned* strategy matched the others using 4 attempts instead of 32: it learned to stop once it found a clear win. | The winners grew the model within the allowed 30% extra training time, so these are data-efficiency gains. "Learned search saves effort" matches what the Dream-RSI paper really claims, but we have one round per strategy and the rounds differ from each other by about the same amount, so it needs repeats. **Follow-up with no extra training time allowed: nothing found** in 5 rounds (best +0.0013, bar 0.010). So the wins came from bigger models, not cleverer training. [details](docs/TREE-SEARCH.md#t2-nanochat-lite-rooted-at-m3b0005-2026-09-30) |
| Does *which* earlier instructions you remind a model of matter? ("stencil": a 4B model in long coding chats, where old instructions fall out of its memory) | **Not found at this scale.** 15 selection rules; none gained the 5 points we were looking for (best +1.7). Even the default reminder scored about the same as no reminder, although an "oracle" that knows the right instructions gains ~27 points. | Picking which forgotten instructions to repeat, without knowing which matter, did not help. The headroom is real but no simple rule found it. |
| Can a model learn a long document into its weights at test time, instead of reading it in context? ("test-time training", 1.7B model, 8K-token documents) | **Not found at this scale.** 15 training recipes (which layers, learning rates, which text spans); none beat plain reading, and the default was slightly worse (−2.4 ± 2.7 points on hidden data). | With a few seconds of training per document, reading beats learning here. Published gains used bigger models and more careful setups. |
| Can an agent dropped into an unfamiliar environment learn it into its own weights, like a driver learning a new city? | **Yes in the test worlds, but it doesn't carry over.** A small 4B model explored a made-up computer environment, then trained itself on what it saw. Afterwards it finished 80% of new tasks there with no notes in view, versus 14% for the starting method and 60% for keeping the notes in view. An independent AI audit found no cheating. **But** on a follow-up set of worlds with new kinds of actions and reworded messages, the same method scored 7%, *below* not learning at all (8%). Keeping the notes in view still scored 56% there, and the starting method 15%. | The winning recipe works by recognizing the exact wording of the world's messages and task requests, which the rules allowed. Change the wording and it breaks. So this is a result about *these* worlds, not a general way to learn new environments. It also cost about 6 points on grade-school math. The next campaign scores on reworded worlds, so only general methods can win. [audit](ideas/plastic-agent/AUDIT-v1.md) |
| Can a small model answer math questions more accurately on the same token budget, just by choosing *when* to sample another answer, when to stop, and how to vote? ("test-time-compute controller", Qwen3-0.6B, grade-school math, 1,000 tokens per question on average) | **Yes, a small win.** On hidden questions, accuracy rose from about 72% to about 75% (+3.2 ± 0.5 points; the bar was 3). The lab kept 3 of 51 changes: stop as soon as 3 answers agree, break ties by the model's confidence, and a stop-or-continue rule learned from training questions that saves tokens on easy questions and spends them on hard ones. | Half the gain comes from the simplest idea, stopping at 3 agreeing answers. The result only just clears the bar, and it is for one small model whose answers were sampled once and cached. |
| Can a small model trained from scratch learn to do *more* reasoning steps in its head than it was trained on, if we give it a design that "thinks" internally (loops, recurrent memory)? ("latent-arch", 27M-parameter models, 11 minutes of training) | **Couldn't be tested at this scale.** Models learned to track a shuffle for 2–3 steps, but on chains longer than any seen in training every design guessed at chance (20%), including a "positive control" design that published work says can do this. | Within an 11-minute training budget, no design gets past chance on chains longer than it trained on, so the lab's search would have nothing to climb. We stopped rather than keep tweaking until something moved. The task, the tests and the controls are kept for a bigger budget. |

Full per-run reports live in `~/arlab-runs/<pack>/<tag>/report.md`. `DECISIONS.md` records every judgment call with
its reason.

## Research directions

Each direction is a folder in `ideas/` with a plain-language `IDEA.md`, and sometimes a literature review in
`RESEARCH.md`.

- **Focus vs. memory**: [stencil-focus](ideas/stencil-focus/IDEA.md). Based on a neuroscience account of how the
  brain picks which stored patterns to read out.
- **Learning at test time**: [ttt-context](ideas/ttt-context/IDEA.md) and
  [plastic-agent](ideas/plastic-agent/IDEA.md) ([research](ideas/plastic-agent/RESEARCH.md),
  [deep dive](ideas/plastic-agent/TTT-DEEP-DIVE.md)). Models that update themselves from what they just saw.
- **Thinking in latent space**: [looped-latent](ideas/looped-latent/IDEA.md)
  ([research](ideas/looped-latent/RESEARCH.md)). Reasoning in hidden states instead of written tokens.
- **Memory built into the architecture**: [research](ideas/memory-architecture/RESEARCH.md);
  [memory-longmemeval](ideas/memory-longmemeval/IDEA.md) was the first test.
- **Small-model coding agents**: [agentic-coding-small](ideas/agentic-coding-small/IDEA.md).
- **Planned for tree search** (see [docs/TREE-SEARCH.md](docs/TREE-SEARCH.md)):
  - choosing pretraining data;
  - GPU kernels, only with a harness hardened against reward hacking.

## The rules it never bends

- Frozen graders; the agent never sees them or the hidden data.
- The final check on hidden data happens once per run and is never re-rolled.
- The bar for "meaningful" is set before any results exist.
- Nulls and "inconclusive" are reported as plainly as wins.
- On the machine: no admin rights, never touches anything it didn't create, never publishes anything, and the only
  AI agent it calls is Codex in a container (see `CLAUDE.md`).

---

## For developers

```bash
cd ~/arlab && uv sync                                  # creates .venv with the `arlab` CLI
docker build -t arlab-agent:0.159.2 -f docker/Dockerfile.agent docker   # the Codex agent image
mkdir -p ~/.cache/arlab/codex-home && CODEX_HOME=~/.cache/arlab/codex-home codex login --device-auth

arlab new my-idea --idea ~/ideas/my-idea.md   # scaffold ideas/my-idea/ from templates/pack/
arlab check ideas/my-idea                     # schema, image, data prep, pack tests + one baseline run
arlab run ideas/my-idea --tag t1 --detach     # greedy campaign as a systemd --user unit (resumable, kill -9 safe)
arlab tree run ideas/my-idea --tag tr1 --budget 32 --workers 4 --detach   # tree search instead
arlab status my-idea --tag t1                 # progress; `arlab stop` / `arlab report` also exist
```

A greedy campaign goes CHECK → PREPARE → TESTS → SEAL → PROBE → CALIBRATE → LOOP → FINALIZE → REPORT. SEAL
snapshots the pack, so editing the repo never affects a running campaign. CALIBRATE measures seed noise and checks
statistical power. FINALIZE runs the one hidden check. The rules are in `PLAN.md` §3.5.

```
arlab/            runner: cli, pack, campaign, experiment, execute, agent, stats, guards, record, report
arlab/tree/       tree search: model, replay, policy (sandbox + built-ins), online, dream, report
arlab/lib/        helpers packs import (token-budget loop, LM scorer, budgeted model client, clean-room tests)
ideas/<name>/     one pack per idea: IDEA.md, pack.yaml, program.md, surface/, frozen/{run,eval,prepare}, tests/
tests/            unit tests + CPU fixture campaigns
```

- Docs: [PLAN.md](PLAN.md) (spec), [docs/PACK-AUTHORING.md](docs/PACK-AUTHORING.md),
  [docs/ORCHESTRATOR.md](docs/ORCHESTRATOR.md), [docs/TREE-SEARCH.md](docs/TREE-SEARCH.md),
  [docs/spark-notes.md](docs/spark-notes.md), [docs/retro.md](docs/retro.md).
- Tests: `make test` (fast); `make accept-M0 … accept-M6` and `make accept-T1` (exit code = verdict).

License: MIT. `ideas/nanochat-lite/surface/train.py` is ported from karpathy/autoresearch (MIT).
