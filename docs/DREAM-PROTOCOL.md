# Pre-registered protocol: does a dreamed search policy beat fixed policies? (P3, registered 2026-10-01)

Written before any P3 data exists. Design source: gpt-6.1-sol's researched recommendation (2026-10-01, saved as
`docs/DREAM-PROTOCOL-sol.md`). The T1/T2 results in [TREE-SEARCH.md](TREE-SEARCH.md) showed why a different design is
needed: nanochat rounds cost 2.5–4 GPU hours, and the spread between rounds of the same policy (≈ 0.4 MES) is as
large as the differences between policies. A powered test there would take hundreds of GPU hours.

## Claim
For this frozen dreamed policy, on the ttc-controller pack with its baseline root and at 12 attempted nodes, W = 4:
the expected holdout improvement of the node the policy *selects* (top 3 by screen → confirm → best confirm d > 0, or
the root if none) beats **both** fixed policies by more than **0.2 MES**, at no more Codex calls.

The result is conditional on this pack and cache. Transfer to other packs is not claimed.

## Arms (one tree per arm per block)
| arm | policy |
|---|---|
| D | the dreamed policy `p3-dream` (frozen after the gate) |
| P | `parallel_refine` (Dream-RSI's fixed control) |
| G | `greedy_best_leaf` (arlab's greedy runner) |
| S | `stop_first_win`: parallel_refine that stops at the first node with score ≥ 1 MES (secondary). It isolates the one behaviour T2's dreamed policy showed. |

## Procedure
1. **Development:** six broad parallel_refine trees, `p3-d0…d5`, 24 nodes each, W = 4. d0 calibrates and the rest
   reuse its calibration.
2. **Dream:** `p3-dream` trains on d0–d3, uses d4 and d5 as the held-out gate, M = 8 revisions (gpt-6.1-sol), replay
   horizon 12 nodes. The gate is used once.
   - If it rejects, the protocol records "no dreamed policy passed the gate" and stops. A rejection is a result, not
     a reason to retry with tweaks.
3. **Blocks b = 1…48**, in this order:
   1. A shared opening `p3-bNN-o`: parallel_refine with budget 4, i.e. the first batch of 4 root children.
   2. Arms D, P, G and S, each a tree with budget 12 (the 4 opening nodes are charged to every arm) and
      `--opening-from p3-bNN-o`. The arms run concurrently, with no memory across rounds.
   3. `arlab tree block-eval`: each arm's chosen node and the root on 5 fresh holdout seeds,
      20000 + 10·b + {0…4}, shared by all arms.
   4. Pacing: one block starts at most every 110 min, which keeps the protocol near ~450 Codex calls/day.
4. **Analysis:** `arlab tree protocol` over all 48 `block.json`. There is no early stopping and no interim decisions.

## Primary metric
q = mean over the block's 5 seeds of (selected − root) / MES on the holdout split. If the selected node fails or
violates a guard on a seed, the root's value is used for that seed.

## Decision rule (one-sided α = 0.025 per comparator; t intervals over blocks; bootstrap bounds as sensitivity)
- **Proven:** for both X ∈ {P, G}, the lower bound of mean(q_D − q_X) > 0.2 **and** the upper bound of
  mean(calls_D − calls_X) ≤ 0.
- **Denied:** for either X ∈ {P, G}, the upper bound of mean(q_D − q_X) < 0.2.
- **Inconclusive:** otherwise.

Also reported:
- the D − S contrast;
- nodes used;
- each arm's mean q;
- the stop reasons;
- replay-predicted vs online ranks on the 48 blocks (diagnostic only).

## Cost (planned)
- Codex calls: ≈ 144 development + 8 dream + ≈ 1,600 block calls.
- No GPU: ttc trials are cached and run on CPU.
- About 4 days of wall clock at the pacing above.

## Not changed after registration
- the arms, budgets, W, seeds, margin, α and decision rule;
- the pack (ttc-controller at its current seal, B = 1000);
- the proposer (gpt-6.1-sol, effort high).

Any deviation is logged in DECISIONS.md, with the reason, before the analysis is run.
