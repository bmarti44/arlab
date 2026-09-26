# nanochat-lite — the lab's calibration pack

**Hypothesis.** Within a fixed token budget, changes to a small GPT's architecture, optimizer and schedule
(the karpathy/autoresearch search space) lower validation bits-per-byte on climbmix text.

**Why.** It is the reference autoresearch task (karpathy/autoresearch, GB10 numbers from mazar/autoresearch-spark),
so it calibrates arlab's speed, noise floor and honesty on this machine.

**Setup.** Depth-6, width-384 GPT (26 M params), vocab 8192 BPE, seq len 1024, MuonAdamW. The frozen harness
feeds random 64×1024 windows of the training stream until the token budget is spent. The frozen scorer computes
cross-entropy from the model's logits on 2048 fixed rows (validation and holdout are disjoint halves of the
pinned upstream validation shard) and checks causality.

**Budget and MES.** Token budget and MES were set by a pilot calibration (tag `pilot`), which PLAN.md allows for
this lab-calibration pack only; see DECISIONS.md. They are fixed for every later tag.
