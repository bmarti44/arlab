# Task: improve an exploration policy for a discovery-tree search

You are improving the *search strategy* of an automated research lab, not solving a research problem yourself.

In each research run, a coding agent grows a tree of attempts. Every node is one attempted code change, built on its
parent node's code and scored once on a noisy validation run. The **policy** (`policy.py` in this directory) decides
which nodes get expanded next and when to stop. Your job: edit `policy.py` so that, on recorded trees, it finds the
best attempts sooner and wastes fewer attempts.

## The policy contract (`API.md` has the details)
- `select(nodes, budget_left, W) -> list[str]`: return up to W node ids to expand, all from A(T) = {"root"} ∪ leaves.
  Repeats are allowed (several children of one node in one batch). Return `[]` to stop the search.
- Choosing "root" starts a new, independent direction; choosing a leaf refines that attempt.
- `nodes` is the revealed tree only: `id, parent, depth, order, status, score, cost_s, children, leaf`.
  `score` is the improvement over the baseline in units of the minimum effect of interest (MES), measured on one seed,
  with noise. It is `None` when the attempt crashed, broke a guard, and so on.
- The policy must be a **pure function** of its arguments: no files, no network, no global state across calls,
  standard library only, and at most a few seconds per call. It runs in a sandbox with no network.

## How it is scored (replay)
Each recorded tree is replayed. Choosing a node reveals its next *recorded* child. If no recorded child is left, the
policy gets nothing (and three empty batches in a row end the replay). So a policy can only discover what the
recording contains, in the order it was recorded.

Objective per tree: V = max(0, best revealed score − root score) − 0.5 · N / N_ref, where N is the number of nodes
revealed and N_ref is the run's node budget (`budget_left` at the first call). The mean V over the training trees decides. Your version is accepted
only if it also does not lose on **held-out trees you cannot see**. Do not special-case the trees shown to you (ids,
depths, orders): generalizable rules only. Being overfitted to these trees is failure.

## Files
- `policy.py`: the current best policy. Edit this file.
- `API.md`: the contract.
- `trees/*.md`: each training tree in full (structure, statuses, scores, tags, costs).
- `replay.md`: how the current policy did on each training tree (V, N, stop reason, the batches it selected, and the
  best-so-far curve).
- `attempts.md`: earlier revisions in this dreaming round, with their training V. Learn from them.

Make one coherent revision. Keep it short and readable, and state the idea in a comment at the top. Then emit the JSON
object: action "edit", a one-line description, a hypothesis_tag (for example `early-stop`, `width`, `bandit`,
`prune-failures`), and constraint_learned null.
