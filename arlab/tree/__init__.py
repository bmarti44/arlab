"""Discovery-tree search (Dream-RSI style, arXiv 2609.14858): an orchestration layer over the greedy runner.

model   - tree.json (nodes = attempts), the public view a policy may see, the frontier A(T) = {root} ∪ leaves
replay  - the trusted replay simulator (reveals only recorded outcomes, in recorded order) and the objective V
policy  - the sandboxed policy process (docker, no network, JSON lines) and the built-in policies
online  - TreeCampaign: grows a real tree with Codex + the frozen evaluator, then finalizes with one holdout
dream   - Codex rewrites the policy offline; accepted only if held-out-tree replay V does not drop
"""
