# Prior art survey (2026-09-26)

Reference only. Not a spec. PLAN.md wins on any conflict. Licensing: vendor only MIT/Apache code with attribution; EurekAgent is AGPL (ideas only); mazar/autoresearch-spark has no license file on GitHub (README says MIT) — ideas and numbers only.

## 2. What exists (survey, September 2026)

### 2.1 The thing being replaced: CoreWeave / W&B ARIA
- Announced 2026-06-29, public preview. "AI Research & Iteration Agent" built on W&B Weave; a coding agent that reads your W&B experiments, builds live visualizations, forms hypotheses, writes configs, launches experiments through W&B Launch on your compute, evaluates results, and proposes next steps; roadmap is "deeper autonomous research". It is closed, hosted, and tied to W&B's tracking + Launch.
- **What to copy:** hypothesis → config → launch → evaluate → next, with the project's full experiment history as context. **What we do differently:** local, open, evaluator-sealed, and with statistical acceptance rather than "the agent read the chart".

### 2.2 The contract everyone uses: karpathy/autoresearch (MIT)
- Three files: `prepare.py` (frozen constants, data prep, tokenizer, dataloader, `evaluate_bpb` — never modified), `train.py` (the only file the agent edits), `program.md` (human-edited instructions).
- Fixed **5-minute wall-clock training budget**; metric `val_bpb` (lower is better, vocab-independent); ~12 experiments/hour.
- Loop (from `program.md`): branch `autoresearch/<tag>` → edit `train.py` → commit → `uv run train.py > run.log 2>&1` → `grep "^val_bpb:"` → log to `results.tsv` (`commit  val_bpb  memory_gb  status  description`; status ∈ keep/discard/crash) → keep (advance) if improved else `git reset` → repeat, **never stop to ask**; 10-minute hard timeout; crashes: fix if trivial else log `crash` and move on; simplicity criterion (a tiny gain that adds ugly complexity is not worth it; a simplification at equal metric is a win).
- Karpathy's own results: ~700 experiments over two days, ~20 real improvements, ~10–11% speedup on already-tuned code, and a real bug found (missing scalar in QK-norm).
- Weaknesses everyone hit: the agent runs the loop in-band (it can skip steps, mis-grep, or edit what it shouldn't), no noise floor, single greedy incumbent, wall-clock budgets are platform- and contention-sensitive, Codex has no native loop mode (`codex exec` runs once and exits).

### 2.3 DGX Spark ports (confirm the platform works)
| Fork | What it adds | Use for us |
|---|---|---|
| `mazar/autoresearch-spark` (MIT) | Working GB10 port, `train.py` hyperparameters pre-tuned for the 5-minute budget (README quotes ≈1.17); a published run in discussion #225: depth 6, n_embd 384, 26.3M params, **val_bpb 1.1495 in 300 s**, 279k tok/s, MFU 15%, peak 13.7 GB | **Calibration numbers + seed of the toy pack** |
| `Entrpi/autoresearch-everywhere` | Cross-platform; validated GB10 lane with a FlashAttention-4 runtime image (`vllm-node-tf5-fa4:sm120`), `calibrate.py` one-button hardware calibration; val_bpb starts ~1.16 on GB10 | FA4 container recipe; calibration idea |
| `apkovacs/autoresearch-dgx-local` | Three agent modes: *hypothesis generator* (LLM emits structured JSON edits; script executes), *guarded* Claude Code (allowlists, token caps), *minimal* | The **structured-JSON-edit backend** for weak local models |
| `David-Barnes-Data-Imaginations/autoresearch-DGX-Spark` | Docker + unified-memory tweaks (pinned memory, OOM protection, ARM64), long-horizon research program (recursive-depth transformer, TTT) | Docker/UMA notes |
| `mmonad/autoresearch-dgx-spark` (#207) | Straight GB10 port | Cross-check |
| `SuperElectron/dgx-spark-tuner` | Autoresearch over **vLLM serving configs** on GB10 (sparkrun, llama-benchy), append-only ledger, mutation schema, guardrails | Shows the loop generalizes to non-training packs on this box |

### 2.4 Generalized "any metric" harnesses (the closest OSS analogues to ARIA's loop)
| Project | Language / license | Enforcement | Backends | Notes |
|---|---|---|---|---|
| **VectorInstitute/helix** (`pip install helices`) | Python, Apache-2.0, v0.1.5, CI+tests, 20★ | `helix.yaml` declares `scope.editable/readonly`, `metrics.primary` (+direction), `evaluate.command`, timeout, regex patterns; `experiments.tsv` ledger; `helix run/status/init` | Claude Code (default), Gemini CLI, custom `AgentBackend` protocol | Cleanest declarative schema; small community; no Codex/Ollama backend shipped |
| **THU-Team-Eureka/EurekAgent** (arXiv 2606.13662) | Python, **AGPL-3.0**, 83★ | Agent container + **separate grader container**; hidden `evaluate.py` mounted read-only only in the grader; `grade_submission()`/`is_better()` contract; propose→implement→evaluate loops; budgets; resumable; web monitor | Claude Code (any Anthropic-compatible endpoint, e.g. GLM); Codex adapter "welcome" | Best evaluator-isolation design; SOTA on circle packing, TriMul kernel, 85.7% medal rate on an MLE-bench subset for <$17/run |
| `wbbradley/autorize` | Rust CLI | Runs *any* agent CLI in sandboxed git worktrees against any scoring command; keep/discard until a deadline | any CLI | Out-of-band by construction |
| `mandar-karhade/scalar-loop` | Python CLI | Seals harness files, enforces repo scope, keeps/reverts only edits that pass metric **and guard commands** | agent-agnostic | "Guard commands" idea (tests, size, latency caps) |
| `199-biotechnologies/autoresearch-cli` | Rust CLI | Scaffolds config, validates eval commands, JSONL results, installs slash-command skills into multiple agents | multi-agent | Scaffolder idea |
| `nikhaldi/autoresearch-lab` | Python | Docker sandbox for the pipeline, host-side git commit/revert | black-box | Host-owns-git pattern |
| `aelaguiz/codex-autoresearcher` | Python | Separate **worker and judge** Codex processes, static `evaluate.sh`, schema-validated keep/restore verdicts, attempt forensics | Codex | Judge-separation pattern |
| `SarahXC/codex-autoresearch-harness` | bash | `codex exec` inside a loop, **one experiment per call**, A/B two models from same baseline | Codex | Cerebras' harness; proven |
| `davebcn87/pi-autoresearch` (+ OpenClaw/OpenCode/Cursor/Claude-Code ports) | pi extension | Metric-driven loop for any target, live dashboard; Shopify used it on 40+ metrics | pi | Dashboard/UX reference |
| `hugoferreira/autoresearch` | — | Falsifiable hypotheses, isolated worktrees, instrument-backed observations, gate review, reusable lessons | — | Hypothesis-first framing |
| `rbudnar/open-autoresearch` (AutoResearch++ v0.6) | protocol + templates, MIT | Explicit split of **in-band-advisory vs out-of-band-enforced** controls; frozen splits with content-hash `MANIFEST.json`; behavioral-equivalence fixtures; adoption levels (Level-1 branch winner → Level-3 promoted with ablation + skeptic + non-agent verifier); `not_deployable` markers | any | Governance vocabulary we adopt |
| `Human-Agent-Society/CORAL` (arXiv 2604.01658) | — | Claude Code / Codex / OpenCode workers in isolated worktrees, `coral eval`, shared notes/skills | multi | Multi-agent later |

Also relevant: `dean0x/autolab` (`autojudge` statistical keep/discard, `autosteer`, `autoevolve` branch competitions), `ErikDeBruijn/autoresearcher2` (Bayesian selection + persistent memory, runs on Blackwell RTX PRO 6000), `hgarud/autoresearch` (MAP-Elites evolutionary DB), `tonitangpotato/autoresearch-engram` (recall/reflection memory across runs), `robzolkos/pi-lifeline` (small models escalate to a stronger advisor on plateaus), `epappas/autoresearch-rl` (grid/random/LLM/hybrid experiment *policies* over a frozen `prepare.py`), `blog.skypilot.co` "research-driven agents" (literature phase before coding), Paper Lantern (paper-MCP inside the loop → 3.2% lower val loss over 100 experiments), `SohniSwatantra/autoresearch-local-llm` (local Qwen instead of Claude Code), `burtenshaw/multiautoresearch` (planner/researcher/reviewer/memory-keeper sub-agents across Claude Code, Codex, OpenCode, pi).

### 2.5 Research-grade systems (for ideas, not as the base)
- **Arbor** (RUC-NLPIR, arXiv 2606.11926): hypothesis tree, keeps only improvements that survive held-out evaluation; claims 2.5× over Claude Code/Codex at equal compute.
- **GEAR: Genetic AutoResearch** (arXiv 2605.13874): bounded frontier of code variants instead of a single incumbent; argues greedy keep/discard prematurely discards complementary local optima.
- **Recovering Wasted Compute in Autoresearch Agents** (COLM 2026, arXiv 2608.10424): agents (1) re-solve the same bugs repeatedly, (2) under-tune hyperparameters even with budget left, (3) don't explore, (4) analyze data without acting on it. Fix that worked: a **global debug consultant** sharing discovered runtime constraints across branches, plus control-level changes.
- **NanoGPT-Bench** (IntologyAI): current coding agents recover <10% of the human-achievable speedup and mostly tune hyperparameters. `ferreirafabio/autoresearch-automl`: over a 24-h budget on Karpathy's task, code-editing autoresearch is competitive but **fixed-space classical HPO still wins**. ⇒ include a classical HPO policy; don't expect miracles from the LLM alone.
- **Cerebras, "How to stop your autoresearch loop from cheating"** (71 experiments): tightly scoped loops with a strict gate surface real findings; loose objectives drift within hours (the agent "abandoned our experiment and started its own"); proposal quality dominates cost (GPT-5.4 accepted 67% of proposals vs Codex-Spark 17%; rejected proposals still burn full training runs); one experiment per `codex exec` call prevents context overflow and gives clean crash recovery; most engineering time went to sandbox/GPU/env friction, not research.
- **EurekAgent's** thesis: "agent environment engineering is all you need" — resources, constraints, artifacts, budgets, human interfaces around off-the-shelf CLI agents.
- Sakana AI-Scientist(-v2), AIDE (WecoAI), Agent Laboratory, Curie, MLR-Copilot, AutoResearchClaw, Sibyl: idea→paper pipelines; heavier, cloud-oriented; not the base for a single-box lab.

### 2.6 Lessons the ecosystem agrees on (design inputs)
1. One editable surface, one frozen evaluator, one scalar primary metric (+ guard metrics). The loop generalizes to anything with a number.
2. The runner owns git + eval + ledger; the agent only proposes and edits. In-band instructions ("never edit prepare.py") are advisory; enforce with mounts/permissions/diff checks.
3. One experiment per agent call, fresh context each call; all durable state lives in files (ledger, notes, constraints).
4. Measure the noise floor first (repeat the baseline with different seeds); a "win" inside the noise floor is not a win. Confirm wins with paired seeds; sealed holdout for promotion.
5. Fixed budgets make runs comparable; on shared/throttling hardware prefer **step/token budgets with a throughput guard** over pure wall-clock.
6. Record runtime constraints once (OOM at batch>X, kernel Y unsupported on sm_121) and inject into every later call.
7. Keep a frontier (not only an incumbent) once the greedy phase plateaus; add classical HPO for the numeric knobs.
8. Escalate to a stronger model only on plateau/crash loops; route cheap steps to local models.
9. Evaluator hardening: think like the adversary (score tampering, tolerance abuse, hidden-test leakage, reading answers from the input, gaming mutable metrics — e.g. the tennis-XGBoost case where the agent gamed ROC-AUC until the evaluator was hardened).
10. Environment friction is the real bottleneck: pin containers, pre-download data/weights, warm caches, and write the debug ledger.

### 2.7 Decision: what to build on, and why not just adopt X
- **Not helix as-is:** right schema, but agent-runs-the-loop model, no Codex/Ollama backends, tiny community — we'd be forking within a week. **Do** vendor its `helix.yaml` shape (Apache-2.0) as the basis of `pack.yaml` and keep the door open to contributing an `arlab` backend upstream.
- **Not EurekAgent as-is:** AGPL (fine for personal research, but copying code into an MIT/Apache repo is a licensing mess), Claude-Code-only, Node 22 + two Docker containers, propose/implement sessions of 20–120 min — heavier and slower than a 5-minute loop. **Do** copy the design: hidden evaluator in a grader container, `grade_submission`/`is_better` contract, `INSTRUCTION.md`/`SUBMISSION_FORMAT.md` split, resumability, defensive-evaluator checklist.
- **Not a bash harness:** proven but not reproducible/inspectable enough for multi-day campaigns.
- **Therefore:** build `arlab` — small, typed Python, `uv`-managed, fully tested on the Spark, with the runner as the source of truth. The agent that executes this plan should read the loop code of helix, codex-autoresearcher, codex-autoresearch-harness and autolab before writing ours (≈2 hours), and may vendor MIT/Apache snippets with attribution. Licensing rule: never copy AGPL code (EurekAgent); ideas only.
- **Allowed shortcut:** if, after reading `VectorInstitute/helix` (Apache-2.0, ~115 KB sdist), the executor judges its `pack`/ledger/git code solid, it may **fork helix as the seed of `arlab`** and add the runner-owned loop, grader isolation, stats, Codex/Ollama backends on top. Either way the CLI surface, pack contract, ledger columns and acceptance rules in §3 are the spec; record the choice in `DECISIONS.md`.


## 10. Sources (verified 2026-09-26)
- CoreWeave ARIA launch — https://www.coreweave.com/news/coreweave-aria-launches-as-an-ai-research-and-iteration-agent-with-autonomous-research-and-collaborative-intelligence ; W&B report — https://wandb.ai/wandb/aria/reports/Introducing-CoreWeave-ARIA-AI-Research-and-Iteration-Agent--VmlldzoxNzM1MzA4Mg
- karpathy/autoresearch (README, program.md, notable forks) — https://github.com/karpathy/autoresearch ; fork index — https://github.com/karpathy/autoresearch/discussions/225
- DGX Spark ports — https://github.com/mazar/autoresearch-spark · https://github.com/Entrpi/autoresearch-everywhere · https://deepwiki.com/apkovacs/autoresearch-dgx-local · https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark · https://github.com/SuperElectron/dgx-spark-tuner
- Curated ecosystem lists — https://github.com/yibie/awesome-autoresearch · https://github.com/WecoAI/awesome-autoresearch · https://github.com/webfuse-com/awesome-autoresearch
- helix — https://github.com/VectorInstitute/helix (PyPI `helices`)
- EurekAgent — https://github.com/THU-Team-Eureka/EurekAgent (arXiv 2606.13662)
- open-autoresearch / AutoResearch++ — https://github.com/rbudnar/open-autoresearch
- Cerebras: how to stop your autoresearch loop from cheating — https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating ; harness — https://github.com/SarahXC/codex-autoresearch-harness
- Recovering Wasted Compute in Autoresearch Agents (COLM 2026) — https://arxiv.org/abs/2608.10424
- Shopify on generalized autoresearch — https://shopify.engineering/autoresearch
- NVIDIA DGX Spark playbooks — https://github.com/NVIDIA/dgx-spark-playbooks (CLI coding agents w/ Ollama: `nvidia/cli-coding-agent`; PyTorch fine-tune: `nvidia/pytorch-fine-tune`; OpenShell: `nvidia/openshell`; Unsloth, vLLM, SGLang, NeMo)
- DGX Spark product page — https://www.nvidia.com/en-us/products/workstations/dgx-spark/ ; bandwidth/thermal reviews — https://sayob.com/blog/dgx-spark-memory-bandwidth-explained/ · https://toolhalla.ai/blog/nvidia-dgx-spark-complete-guide-2026 · https://kaitchup.substack.com/p/dgx-spark-use-it-for-fine-tuning
- Ollama cloud models — https://ollama.com/blog/cloud-models ; model rotation notes — https://github.com/RedBeret/awesome-ollama-models ; Codex + open models — https://explainx.ai/blog/codex-open-source-models-ollama-oss-mode-2026
- Codex CLI non-interactive reference — https://developers.openai.com/codex/noninteractive · https://developers.openai.com/codex/cli/reference
- stencil-llm — https://github.com/bmarti44/stencil-llm (README, GPT2-PLAN.md)

