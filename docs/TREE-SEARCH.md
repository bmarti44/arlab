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

## Results

### T1: shakedown on `_fixture` with real Codex (2026-09-28)

Setup: W = 8, 48 nodes per round.
- r0: π₀ (`parallel_refine`).
- Rounds 1–3: a dreamed policy (rK) against a paired fixed control (cK).
- 336 nodes, about 345 Codex calls, 2 h 10 min.

Every round's own finalize gave `supported`. That's expected: this toy problem saturates, and the first batch
already reaches about 6.8 of the ~7.1 MES found.

| Round | Policy | Online V (β = 0.5) | Best score (β = 0) | Nodes used |
|---|---|---|---|---|
| r0 | parallel_refine | 6.583 | 7.083 | 48 |
| r1 / c1 | fx-d1 / control | **6.823** / 6.683 | 6.917 / **7.183** | 9 / 48 |
| r2 / c2 | fx-d1 / control | **6.823** / 6.583 | 6.917 / **7.083** | 9 / 48 |
| r3 / c3 | fx-d3 / control | 6.723 / **6.900** | 6.817 / **7.400** | 9 / 48 |

**What we learned**
1. **Dreamed policies learned to stop early.** They found 0.2–0.6 MES less than the control, using about 1/5 of the
   nodes. At β = 0.5 they won 2 of 3 paired rounds. At β = 0 (best found) the control won all 3.
2. **The noise is as large as the effect.** The *same* fixed policy scored 6.58–6.90 across four rounds (Codex is
   stochastic). That spread is as big as every dreamed-vs-fixed difference, so there is no claim either way at
   this sample size.
3. **Replay did not predict the online ranking.** On the control trees, replay ranked fx-d3 (6.97) above fx-d1
   (6.71). Online the order was fx-d1 (6.82) above fx-d3 (6.72).
4. **Trees from narrow policies are poor replay data.** The dreamed runs recorded only about 9 nodes each, so
   every policy stalls quickly on them.
5. **Simple untuned policies did as well as dreamed ones in replay.** `greedy_best_leaf` scored 6.90 and
   `ucb_chains` 6.88 on the wide trees, against fx-d3's 6.87. This matches AIRA's finding that the search
   algorithm matters less than the operators.
6. **The held-out guard worked.** fx-d2 improved on its training trees (6.81 → 6.84) but lost on held-out r0
   (6.74 < 6.82) and was rejected. fx-d1 was accepted UNGUARDED, since no held-out tree existed yet; it
   overfit its single training tree by shrinking the opening breadth one step at a time.
7. **Operator fix.** Simultaneous siblings with identical views all proposed the same change. Numbering them
   ("your #k idea") raised the first batch from 1 distinct idea to 5 of 8.

**Consequences for T2 (nanochat, GPU)**
- Every dreamed round needs its paired control, and several control rounds are needed to measure between-round
  spread before any "dreamed beats fixed" claim.
- Report β = 0 (best found) next to β = 0.5, because stopping early is easy to reward and hard to trust.
- Include `greedy_best_leaf` and `ucb_chains` as extra online controls.
- Dream only on wide trees.

### T2: nanochat-lite, rooted at m3b:0005 (2026-09-30)
Pack nanochat-lite (val_bpb at a fixed token budget, MES 0.01, train_s ≤ 1.3× baseline), root = m3b's supported keep.
σ = 0.0029, 5 holdout seeds (widened from 3 after a load-noisy calibration, see DECISIONS). Proposer gpt-6-sol, pinned
for the whole comparison. Dream step gpt-6.1-sol, M = 8, train a0, held-out a1.

| round | policy | nodes | chosen | holdout d (± 2·SE) | params / train_s |
|---|---|---|---|---|---|
| a0 | parallel_refine (W4) | 32 | n0024: 8 layers, MLP 3×, wd 0.1, matrix lr 0.05 | 0.0127 ± 0.0037 | 30.7M / 1.19× |
| a1 | parallel_refine (W4) | 32 | n0026: 7 layers, MLP 5×, wd 0.1 | 0.0165 ± 0.0037 | 42.8M / 1.26× |
| b1 | dreamed n-d1 | **4** (policy stop) | n0001: 7 layers | 0.0125 ± 0.0037 | 31.3M / 1.14× |
| g1 | greedy_best_leaf | 32 | n0025: 7 layers, wd 0.05, MLP 5×, VE gate 1.5, unembed lr 0.003 | **0.0177 ± 0.0037** | 33.3M / 1.23× |

All four rounds are *supported*, with every holdout seed positive.

**The pack's time slack is the main lever.** Every winner adds depth or width and uses 14–26% more training time,
inside the 1.3× guard. These are fixed-token (data-efficiency) gains, not compute-matched ones. A compute-matched
guard (train_s ≤ 1.05× or FLOP-matched) is the fairer follow-up.

**Dreaming:** n-d1 was accepted, with held-out replay V 1.240 against 1.200 for parallel_refine and train V 1.267
against 0.975. What it learned is to stop once a ≥ 1-MES gain appears, and otherwise refine the top 2 leaves one at a
time. Online it stopped after its first batch, which was the same 4 root children a fixed policy draws, so stopping is
the only learned behaviour exercised. It matched a0 at 1/8 of the nodes and fell 0.004 short of a1, about 1.5 SE of
the difference. The early stop benefited from a lucky first batch: depth 7 came up in batch 1, while a0 found it only
in batch 4.

**Greedy vs parallel:** greedy's deep chain (depth 6) found the best node. Same-policy spread (a0 vs a1) is 0.004,
though, and greedy − a1 is 0.001, so policies can't be ranked from one round each.

**Verdict on the research question:** as in T1, no "dreamed beats fixed" claim. The learned policy saves nodes, which
matches Dream-RSI's actual claim (fewer calls for similar quality), but n = 1 per arm. Next, if repeated:
- 3+ rounds per policy, now cheap because a dreamed round took about 80 min including finalize;
- a compute-matched guard;
- dreaming on all four trees.

### T2b: nanochat-lite, compute-matched replication (2026-10-01)
The same root (m3b:0005), but the training-time guard is **train_s ≤ 1.05×** (compute-matched), with 5 holdout seeds and
proposer gpt-6.1-sol. Rounds: a0 → a1 (parallel_refine) → dream n2-d1 (train a0, held-out a1; accepted) → b1, b2
(dreamed) → g1 (greedy). σ = 0.0025, 2·SE = 0.0031.

| round | policy | nodes | holdout d (± 2·SE) | verdict |
|---|---|---|---|---|
| c-a0 | parallel_refine (W4) | 32 | 0.0013 ± 0.0031 | not found at this scale |
| c-a1 | parallel_refine (W4) | 32 | 0.0008 ± 0.0031 | not found at this scale |
| c-b1 | dreamed n2-d1 | 5 (policy stop) | 0.0012 ± 0.0031 | inconclusive (stopped early) |
| c-b2 | dreamed n2-d1 | 5 (policy stop) | 0.0010 ± 0.0031 | inconclusive (stopped early) |
| c-g1 | greedy_best_leaf | 32 | 0.0005 ± 0.0031 | not found at this scale |

**With the time slack removed, nothing reaches MES 0.01.** No round got near it: all five estimates lie in
0.0005–0.0013 bpb. This confirms that T2's wins came from bigger models inside the 1.3× allowance, not from better
training recipes at equal compute. The m3b incumbent is near a local optimum for this budget and this proposer.

The dreamed policy again stopped after its first batch plus one node. With nothing to find, early stopping costs
nothing in quality (its d is within noise of the 32-node rounds) and saves about 85% of the nodes. That is the
"cheaper at similar quality" claim, but in a regime where every policy finds nothing, so it says little about
search quality. The powered test of that question is P3 ([DREAM-PROTOCOL.md](DREAM-PROTOCOL.md)).

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
