# fastweight-memory — can a frozen small LM learn and REVISE a tool world through forward-only weight writes?

Owner direction (2026-10-02): "find new things, like having the weights of the agent change automatically / on the fly
without a training run." Design source: gpt-6.1-sol deep research, `docs/RESEARCH-fast-weights-sol.md` (option A,
chosen by the owner). Closest prior work (must not be re-claimed): fast weights / DeltaNet / DeltaProduct, TTT, Titans,
GenerativeAdapter, StreamAdapter, Doc-to-LoRA, and especially **δ-mem** (2026: frozen Qwen + gated delta-rule state ->
dynamic low-rank attention corrections). "Streamed delta-rule LoRA on a frozen Qwen" is NOT new. What is open, and what
this pack measures: whether such writes learn **actionable** tool semantics (executed calls, not QA), **revise** them
after an unannounced change from a few observations while **preserving** unchanged tools, and **transfer to operation
types the writer never saw in training**, under twin controls and active-memory forgetting guards.

## Hypothesis (pre-registered)
After ONE-TIME training of a small memory interface (shared across worlds), a frozen Qwen3-0.6B agent that streams a
FauxOS exploration log through forward-only state writes (no optimizer, no backward pass, no transcript kept), and then
6 observations of 2 unannouncedly changed tools, executes held-out one-call tasks better than the matched δ-mem-style
baseline writer by >= MES, with the gain present on reserved (never-trained) operation types, coming from the right
world's observations (twin control), and without damaging GSM8K or unrelated-world behaviour while the memory is active.
"No per-world training run" is the claim; one-time preparatory training of the interface is allowed and charged.

## What is fixed (frozen) vs editable (surface)
- **Backbone:** Qwen3-0.6B (local snapshot, bf16, thinking off), frozen everywhere. Fallback to Qwen3-1.7B ONLY if the
  pre-registered capability controls (below) fail on 0.6B; decided in the pilot, before sealing.
- **Read path (frozen, evaluator-owned):** at up to 4 surface-chosen sites (layer index x projection in
  {q,k,v,o,gate,up,down}), the effective weight is W0 + U_s S_s V_s^T, where U_s, V_s are trained interface parameters
  and S_s is the per-world fast state. The evaluator applies the correction itself (the surface never touches the
  model at evaluation time); a site with S = 0 is exactly the base model.
- **Fast state (frozen accounting):** all S_s together <= 64 KiB in fp32 (e.g. 4 x 64x64), plus nothing else that
  persists between records: no token buffers, no KV cache of earlier records, no Python state. The evaluator holds
  the state tensors and passes them in/out of the writer; anything else the writer returns is rejected.
- **Writer (surface):** `write(state, record) -> state`, called once per record, in order, with only the CURRENT record:
  its token ids and the frozen backbone's hidden states for it at surface-chosen layers (computed by the evaluator
  with the BASE model, memory off), plus the current state. Forward-only (evaluator runs it under torch.no_grad in a
  Landlock sandbox worker, like latent-arch's worker.py); <= 5M interface parameters total (writer + all U, V).
- **Interface training (surface `train.py`, RUN):** a wall-clock budget (calibrated in the pilot; target ~40 min on an
  idle GB10) inside the frozen harness, on TRAINING worlds only. The harness supplies episodes (record features cached
  once per training world, task prompts + gold call tokens, revision episodes) and a frozen-model handle to run task
  prompts with the U S V^T corrections (differentiable w.r.t. interface params and through the write chain). RUN never
  mounts validation/holdout data; the surface sandbox can read only /work.
- **Baseline surface:** a δ-mem-style gated delta-rule writer (keys/values/beta from mean-pooled record features, rank
  r state per site, S <- a*S + beta*(v - S k) k^T) with a plain training recipe. This IS the "matched existing
  forward-only memory baseline" the MES is measured against (arlab's verdict compares incumbent vs baseline).

## Worlds and the revision protocol
- Reuse plastic-agent v2's FauxOS "fam" generator (copied into this pack's frozen/prepare; plastic-agent untouched):
  pseudo-word tool names independent of semantics, per-world wording families, balanced typed tasks, no-lookup rule
  (no question answerable from a logged call with an unchanged value), twins with the same names/family.
- **Three disjoint wording banks:** C = interface training, A = validation, B = holdout (add bank C to wording.py; no
  string shared between banks). **Operation types:** training worlds use only the familiar types; validation adds the
  dev-only types; holdout adds the reserved types (bump, pin, count_tag, oldest_at). Reserved types must not occur in
  any training example, target or validation world (tests check actual exposure, not file names).
- **Episode:** (1) stream the 120-record scripted exploration; (2) unannounced change of 2 tools (new op semantics,
  argument order or unit mapping; drawn so the change is identifiable from 3 successful observations each);
  (3) stream 6 post-change records (3 per changed tool) — unaffected tools get no extra records; (4) evaluate
  held-out one-call tasks from the post-change state, with the final memory CLONED per task (tasks never write).
  Sham-change worlds (step 2 skipped, 6 extra records of random tools) are included as a stratum.
- **Task strata (balanced, reported separately):** changed tools / unchanged tools x familiar / dev or reserved ops.
  Diagnostic curves at 1, 3, 6, 12 post-change observations (6 is primary).
- **Sizes (pilot re-checks power):** training 512 worlds (~30k tasks); validation 8 worlds x 60 tasks; holdout
  64 worlds x 60 tasks = 3840 items, 3 holdout seeds (interface re-trained per seed).

## Metric, MES, guards
- **Primary `success`:** world-macro-averaged exact executed success (one call, one retry after an error, plastic-agent
  scoring), after 6 post-change observations. **MES = 0.06** (vs the baseline surface; fixed now).
- Reported every run (evaluator computes all from the same trained interface): `nowrite_success` (S = 0),
  `twin_success` (state written from the twin's stream, scored on the real world), `s_changed`, `s_unchanged`,
  `s_familiar`, `s_dev` / `s_reserved`, `pre_change_success` (same tasks before the revision records), curves.
- **Guards** (thresholds set from the baseline's 5 calibration seeds in the pilot, before sealing, as in plastic v2):
  - `gsm8k_drop` <= 0.02 (+ calibrated noise margin): 400 GSM8K problems WITH a world's memory active vs the base model.
  - `guardworld_drop` <= 0.02 (+ margin): another FauxOS world's tasks with ITS transcript in context, memory of an
    unrelated world active, vs the base model.
  - `retention_drop` <= 0.02 (+ margin): unchanged-tool success before vs after the revision records.
  - `world_specific_share` >= 0.5: (success - twin_success) / (success - nowrite_success).
  - `state_bytes` <= 65536, `interface_params_m` <= 5, `peak_mem_gb` <= 60, train budget (harness-owned).
- **Pre-registered claim conditions** (orchestrator-checked on the holdout result files, as plastic v2):
  reserved-op gain over the baseline > 0 for an unqualified claim (else "familiar operation types only");
  report changed-tool vs unchanged-tool gains separately.

## References (frozen arms, reported only)
`none` (base model, no memory), `icl` (exploration + revision records in context: the accuracy ceiling reference;
shared-prefix caching, so no repeated-prefill penalty), `retrieval` (byte-matched: keep up to 64 KiB of the raw log
text — the most recent records per tool — in context; tests whether weights beat cheap retained text), `additive`
(plain linear-attention fast weights S <- S + v k^T, same interface). Orchestrator-run after the campaign: the best
guard-passing plastic-agent consolidation recipe on 0.6B with a 160 s per-world budget (non-inferiority margin 5 pts),
and one-gradient-step-per-record (separates "no separate job" from "backward-free").

## Positive controls (pilot, BEFORE sealing; stop rule: <= 8 GPU-h, else inconclusive)
1. Evidence sufficiency: `icl` beats `none` by >= 15 pts and reaches >= 25% on each primary validation stratum.
2. Task solvability: an oracle readable tool specification in context reaches >= 80%.
3. Interface capacity: optimizing the fast state S directly per world (gradient, oracle diagnostic — never an
   eligible method) on that world's development tasks, scored on different tasks of the same world, beats
   `nowrite_success` by >= 15 pts on 3 seeds. Answers latent-arch's open question "can this interface express a
   solution at all?" If 1-3 fail on 0.6B, retry once on 1.7B; if they fail there, stop (inconclusive, recorded).
Also: if the baseline δ-mem-style writer already meets the whole claim, report "existing method transfers" — no
novelty claim is manufactured.

## Search and stop rule
Greedy arlab campaign, gpt-6.1-sol, surface = `memory.py` (writer + interface definition) + `train.py` (recipe).
At most 16 experiments or 24 GPU search hours (whichever first), then FINALIZE once on the holdout (3 seeds).
Never reopen the holdout, change the MES or add configurations after seeing it. Hard cap for the whole pack
(pilot + calibration + search + holdout + orchestrator baselines): 96 GB10 GPU-hours.

## Known risks (from sol) and how they are handled
Unidentifiable changes (generator checks identifiability: the 3 observations must rule out the old semantics);
generic tool skill masquerading as adaptation (nowrite + twin + reserved ops); hidden transcript retention (state
accounting, writer gets only the current record, no persistent Python state: fresh worker per world); tasks writing
memory (clone per task); exact-answer lookup (v2 no-lookup rule); indiscriminate reset (retention guard);
adapter damage (GSM8K and guard world with memory ACTIVE); weak baselines (matched δ-mem baseline is the comparator;
ICL/retrieval/offline consolidation reported); floor effects (positive controls stop the pack).
