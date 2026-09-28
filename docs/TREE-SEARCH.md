# Tree search: searching wide, and learning how to search (`arlab tree`)

arlab's original loop is **greedy**. It keeps one "incumbent", asks the agent for one change, keeps the change if it
beats the noise, and repeats. `arlab tree` adds a second way to run a pack. The design follows Google DeepMind's
**Dream-RSI** (arXiv 2609.14858, Sept 2026), with guards added for what that paper leaves untested.

## The idea in three steps

1. **Explore as a tree.** Every attempt is a node: a code snapshot plus one screen score. An attempt can start from
   the original code (a new direction, "pick the root") or from any earlier attempt that has no children yet (a
   refinement, "pick a leaf"). Many directions stay alive at once, so one early lucky result does not lock the search
   in.
2. **Record, then replay.** A finished tree is a free simulator. An alternative strategy can be "re-run" on it: it
   picks nodes in its own order and stops when it wants, and it sees only outcomes that really happened.
3. **Dream.** Codex reads the replays and rewrites the **exploration policy**, a small Python program that decides
   what to expand next and when to stop. This happens M times, and the best rewrite by replay score is deployed in the
   next real round. The model's weights never change. What improves is the lab's search strategy.

## What the paper shows, and what we add

**What it shows.** Dream-RSI beat a fixed parallel-refinement policy at the same budget, but only modestly: 1.22×
faster Lasso using 1.74× fewer calls, and 1.8–2.4× on GPU kernels. Its "162× fewer calls" figure is measured against
SimpleTES's 51,200-call budget.

**What it leaves out.**
- No seeds or error bars.
- No held-out trees.
- No check that replay rankings predict real (online) rankings.
- No released code.
- The unofficial reimplementation (open-dream-rsi) had a replay bug that leaked the scores of nodes the policy never
  reached.

**What arlab adds.**

| Guard | How |
|---|---|
| No leaks in replay | Choosing a node reveals its *next recorded child*, in recorded order. An exhausted branch returns nothing. Revealed nodes are relabelled `r0001…` in reveal order, and the policy sees only `id, parent, depth, order, status, score, cost_s` of revealed nodes (no agent-written tags or text), plus the run's public budget. Tests: `tests/test_tree.py::test_no_leak_of_unrevealed_nodes`. |
| Held-out trees | A dreamed policy is accepted only if its mean replay V on trees it never saw is ≥ the current policy's. With no held-out tree the report says **UNGUARDED**. |
| Replay ≠ reality | Every dreamed online round is paired with a fixed-policy control round on the same pack and budget, and we report replay V against online V. |
| Selection bias | Nodes are scored on one validation seed, so the top node is optimistic (winner's curse). Finalize is pre-registered: take the top 3 by screen score, confirm them on fresh seeds, and choose the best confirm d > 0. **One** holdout run follows, with the usual verdict rules. |
| Untrusted policy code | Policies run in `docker --network none --read-only --memory 2g --cap-drop ALL`, speaking JSON lines. A 20 s limit per decision. |
| No steering | The policy only chooses *which* node to expand. It never adds hints to the agent's prompt (hints hurt in Dream-RSI Fig. 5). |

## Objective

Replay and online use the same objective:

    V = max(0, best score − root score) − β · N / N_ref        (β = 0.5 by default; reports also show β = 0 and β = 1)

- `score` is the improvement over the pack baseline in **MES units**: 1.0 means one minimum effect of interest.
- `N` is the number of nodes spent.
- `N_ref` is the budget: the recorded tree's size in replay, `--budget` online.

## Commands

```bash
# grow a tree (resumable; kill -9 safe; GPU packs hold the arlab GPU lock for the whole run)
arlab tree run ideas/nanochat-lite --tag tr1 --policy parallel_refine --budget 32 --workers 4 --detach
#   --root-from nanochat-lite/m3b:0012   start from an earlier keep; that code becomes this run's baseline
#   --calib-from nanochat-lite/tr0       reuse calibration when pack, data, image and root all match
arlab tree status nanochat-lite/tr1
arlab stop nanochat-lite --tag tr1              # the STOP file works for trees too
# offline
arlab tree import nanochat-lite/m3b             # a finished greedy campaign as a (star-shaped) tree
arlab tree replay --policy parallel_refine greedy_best_leaf ucb_chains --trees nanochat-lite/tr1 looped-latent/v0
arlab tree dream --policy parallel_refine --train _fixture/r1 --heldout _fixture/r0 --name fx-d1 -m 8
```

Built-in policies (`arlab/tree/policies/`):
- `parallel_refine`: W chains refined in parallel. This is the fixed control, as in the paper.
- `greedy_best_leaf`: the whole batch refines the best leaf. This is the tree version of the greedy runner.
- `ucb_chains`: a small bandit over chains.

Dreamed policies go to `~/arlab-runs/_tree/policies/<name>.py`. Dream reports go to
`~/arlab-runs/_tree/dream/<name>/`.

## What a node is

A node runs the same operator as a greedy experiment: one Codex proposal (plus one fix call if it crashes), applied
to the **parent's** code, then one screen trial with guards. The node's `history.md` shows:
- its own chain, from the root to its parent;
- the 5 best nodes anywhere in the tree;
- the other recent attempts;
- tag counts.

`notes.md` is per chain: a child inherits its parent's notes. Every node commit is tagged `node/<id>` in the run's
`work/` repository, so no attempt is ever garbage-collected.

Files for each run live in `~/arlab-runs/<pack>/<tag>/`:
- `tree.json`: the tree, the policy, the budget and every batch's selection;
- `nodes/<id>/`: the view, the agent output, `diff.patch` and the trials;
- `policy.py`: a frozen copy of the policy;
- `report.md`: the verdict, the pre-registered choice, the tree, the anytime curve and cost.

## Plan (stages)

| Stage | What | Where |
|---|---|---|
| T0 | Build, tests (`make accept-T1`), import greedy campaigns, replay the built-ins | CPU |
| T1 | `_fixture` shakedown with real Codex: π₀ then 4 dreamed rounds with paired controls, W 8 × depth 6 (about 450 calls). First replay-vs-online correlation | CPU, alongside the GPU queue |
| T2 | `nanochat-lite` rooted at m3b's keep: 4 chains × depth 8 per round; π₀, 4 dreamed rounds, a final paired control (about 18 GPU hours) | GPU |
| T3 | Second chances: `looped-latent`, `stencil-focus` (tree vs greedy on the same pack), then TTT and agentic if their greedy runs show headroom | GPU |
| T4 | New packs: `ttc-controller` (test-time-compute controller on cached reasoning traces), `latent-arch` (from-scratch tiny GPTs with loops, hybrid linear attention, recursion, continuous thought; depth-generalization split), `data-select` (the program choosing pretraining data), `gpu-kernels` (last, and only with a hardened anti-hacking harness) | GPU |

## Known limits

- **Recorded trees are biased by the recording policy.** A replayed policy can only find what that policy happened
  to try. This is why dreaming needs held-out trees and paired online rounds, and why trees imported from greedy
  campaigns (stars and chains) are smoke-test data only.
- **Replay has no notion of "untried".** A leaf with no recorded child returns nothing, and three empty batches in
  a row end a replay.
- **A resumed partial batch** (runner killed mid-batch) lets the policy re-decide from the saved tree; the lost
  selections are not replayed.
- **A resumed partial batch** (runner killed mid-batch) lets the policy re-decide from the saved tree; the lost
  selections are not replayed.
- **Screen scores are single-seed.** The tree's best node is optimistic until confirm and holdout.
