# Architecture-level memory for LMs: what the evidence and the forums say (brief)

Written 2026-09-27. Scope: memory built into the network (recurrent/linear state, test-time-trained weights, memory layers,
memory tokens, trained caches), not RAG or prompt-level memory. Numbers are as reported by the cited sources; I reproduced none.

**Access limits (read first).**
- **Reddit (r/MachineLearning, r/LocalLLaMA) was not reachable.** The search tool refuses reddit.com, so no Reddit sentiment
  below is first-hand, and I have not guessed at it.
- **OpenReview reviews were not reachable.** openreview.net and its API return a bot-challenge page. Peer-review status below
  comes from conference proceedings and poster pages, not from reading the reviews.
- **What forum signal I used:** Hacker News threads, LessWrong, researcher blogs and X posts quoted on the web. Discord
  (EleutherAI, GPU MODE) was out of reach. Everything marked *opinion* is forum opinion, not evidence.

## 1. Bottom line

1. **Hybrids hold up best.** Linear-attention/SSM layers interleaved with some full-attention layers have the strongest
   evidence: matched-data studies, dozens of open ablation models, and shipped open-weight models. The catch is that the
   hard "memory" (exact recall) still lives in the attention layers' KV cache, not in the recurrent state.
2. **Sparse parametric memory is the most rigorous "new memory" result.** Memory layers, product keys and Engram have
   matched-FLOP, open-code evidence at 8B–27B, but they store *training knowledge*. They do not hold a conversation.
3. **Test-time-trained memory is split.**
   - **Titans / ATLAS / Nested Learning (HOPE):** the most hyped approaches, yet they have no official code, small-scale
     results, and mixed independent reimplementations.
   - **TTT-E2E, qTTT and Cartridges:** these have code or honest negative results, and are the ones worth testing.
4. **Where evidence and enthusiasm disagree most:**
   - Titans/HOPE: enthusiasm ≫ evidence.
   - Infini-attention: the replication failed.
   - Memory layers and sparse memory finetuning: evidence > enthusiasm.
   - Hybrids: roughly aligned, with a documented dissent from MiniMax.

## 2. Ranked table (by evidence strength; "Forum" = community enthusiasm, H/M/L, opinion)

| # | Method | Core mechanism | Best evidence | Replications / critiques | Code / weights | Evidence | Forum |
|---|---|---|---|---|---|---|---|
| 1 | **Hybrid linear-attn/SSM + full attn** ([Gated DeltaNet](https://arxiv.org/abs/2412.06464) ICLR'25; Qwen3-Next/[Qwen3.5](https://huggingface.co/blog/mlabonne/qwen35); [Kimi Linear](https://arxiv.org/abs/2510.26692); [Jamba](https://arxiv.org/abs/2403.19887)) | fixed-size matrix state updated by a gated delta rule (online regression) in ~3/4 of layers; full attention in the rest | NVIDIA 8B Mamba-2-Hybrid vs Transformer on the same 3.5T tokens: hybrid ≥ Transformer on all 12 short tasks and on long context ([2406.07887](https://arxiv.org/abs/2406.07887)). 72 open models: recall rises with more full-attention layers, best ratio 3:1–6:1 ([2507.06457](https://arxiv.org/abs/2507.06457)). Kimi Linear beats full MLA under an identical recipe (RULER 84.3 vs 81.3) ([2510.26692](https://arxiv.org/abs/2510.26692)) | MiniMax dropped hybrid and SWA for M2: "clear deficits in complex, multi-hop reasoning" at scale, and proxy metrics were unreliable ([MiniMax blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)). Split-prefill on **Qwen3.5-4B**: retrieval survives in the attention KV (64–98%) and collapses to 0 with only the recurrent state, which instead carries language/persona ([2609.04434](https://arxiv.org/html/2609.04434v1)). CoT-SFT can wreck hybrid NIAH (67.2→9.4%) ([2606.11052](https://arxiv.org/abs/2606.11052)) | yes, many open weights (Qwen3.5 cached locally) | **High** | H. HN: "linear attentions kind of suck… but the efficiency is very attractive" ([HN](https://news.ycombinator.com/item?id=45766937)) |
| 2 | **Memory layers / product keys / hashed lookup** ([Memory Layers at Scale](https://arxiv.org/abs/2412.09764) ICML'25; [UltraMem](https://arxiv.org/abs/2411.12364) ICLR'25; [Engram](https://arxiv.org/abs/2601.07372)) | huge sparse key→value table (product-key top-k, or hashed N-grams) replacing or augmenting FFN capacity at ~constant FLOPs | Meta: beats dense models with >2× compute and MoE matched on compute and params; factual QA +100%; up to 128B memory params, 1T tokens ([ICML](https://icml.cc/virtual/2025/poster/46172)). Engram, iso-param/iso-FLOP at 27B: MMLU +3.4, BBH +5.0, Multi-Query NIAH 84.2→97.0 ([abs](https://arxiv.org/abs/2601.07372v1)) | Independent small-scale Engram study: collision-free lookup doesn't help; gating is the bottleneck ([2601.16531](https://arxiv.org/abs/2601.16531)). N-gram-embedding gains shrink past ~20 layers ([2606.08347](https://arxiv.org/pdf/2606.08347)). A secondary source says Engram did *not* ship in DeepSeek V4 ([Kili](https://kili-technology.com/blog/data-story-deepseek-v4)) | yes ([memory](https://github.com/facebookresearch/memory), [Engram](https://github.com/deepseek-ai/Engram)) | **High** (at scale, same-lab) | M. X hype for Engram ([thread](https://x.com/scaling01/status/2010748516788777445)); HN: "too convoluted without improvements that could justify the complexity" ([HN](https://news.ycombinator.com/item?id=46592363)) |
| 3 | **Sparse memory finetuning** (continual learning as memory) ([Lin+ 2025](https://arxiv.org/abs/2510.15103)) | update only the memory-layer slots that a new fact activates unusually often (TF-IDF vs pretraining) | NQ F1 drop after learning new facts: full FT −89%, LoRA −71%, sparse memory −11% ([abs](https://arxiv.org/abs/2510.15103)) | Independent Qwen2.5-0.5B retrofit: forgetting within ~1 pt ✔, but only +2.5 pp task gain vs larger LoRA/FT gains ([2605.03229](https://arxiv.org/abs/2605.03229), [2604.05248](https://arxiv.org/abs/2604.05248)) | partial | **Med** | M–H in the continual-learning discourse ([Lin on X](https://x.com/realJessyLin/status/1980662516285075762)) |
| 4 | **Test-time training on a standard LM** ([TTT-E2E](https://arxiv.org/abs/2512.23675); [qTTT](https://arxiv.org/abs/2512.13898); [Cartridges](https://arxiv.org/abs/2506.06266) ICLR'26) | gradient steps on the context at inference: next-token loss into SWA-transformer weights (E2E), W_Q only with frozen KV (qTTT), or a trained compact KV via self-study (Cartridges) | TTT-E2E 3B/164B tok: loss scales with context like full attention, while Mamba 2 and GDN do not; 2.7× faster at 128K. qTTT on Qwen3 1.7–8B: up to +12.6 pp LongBench-v2, +14.1 ZeroScrolls at FLOPs matched to thinking tokens. Cartridges: 38.6× less memory, 26.4× throughput | **TTT-E2E's own table: passkey at 128K is 0.06 vs 0.99 for full attention** ([html](https://arxiv.org/html/2512.23675v1)); training 3.4× slower at 8K. S-TTT: TTT on *random* spans **hurts** LongBench-v2; only good spans help ([2607.09415](https://arxiv.org/abs/2607.09415)). Cartridges needs per-corpus training; rebuilds cost "a large fraction of full preparation" ([2608.30647](https://arxiv.org/abs/2608.30647)); multi-cartridge scaling fixes ([2606.04557](https://arxiv.org/abs/2606.04557)) | E2E [JAX](https://github.com/test-time-training/e2e); [Cartridges](https://github.com/HazyResearch/cartridges); qTTT: none found | **Med** | M. Little forum signal ([HN E2E](https://news.ycombinator.com/item?id=47106538) has few comments) |
| 5 | **Recurrent-memory transformers** ([RMT](https://arxiv.org/abs/2207.06881) NeurIPS'22; [ARMT](https://arxiv.org/abs/2407.04841); [GradMem](https://arxiv.org/abs/2603.13875) ICML'26) | segment-level recurrence through memory tokens; ARMT adds associative (delta-rule) memory; GradMem writes memory tokens by test-time gradient descent | BABILong (NeurIPS'24 D&B): fine-tuned RMT/ARMT lead, and LLMs use 10–20% of their context ([2406.10149](https://arxiv.org/abs/2406.10149)); ARMT 79.9% single-fact QA at 50M tokens | Gains come after **fine-tuning on the task**. Almost all evidence is from one group (Burtsev/Kuratov). 2026 ARMT retrofit: 30% fewer FLOPs ([2607.11614](https://arxiv.org/abs/2607.11614)) | yes ([ARMT](https://github.com/RodkinIvan/associative-recurrent-memory-transformer)) | **Med** | L |
| 6 | **Pure linear RNN / SSM** ([Mamba-2](https://arxiv.org/abs/2405.21060), [Mamba-3](https://arxiv.org/abs/2603.15569) ICLR'26 oral, [RWKV-7](https://arxiv.org/abs/2503.14456), [DeltaProduct](https://arxiv.org/abs/2502.10297) NeurIPS'25) | constant-size state; richer transitions (complex, Householder products) for state tracking | Mamba-3 +0.6–1.8 pts over GDN at 1.5B; RWKV-7 is 3B SOTA-class on fewer tokens | Recall/copying gap: associative recall explains >82% of the perplexity gap ([Zoology](https://arxiv.org/abs/2312.04927)); SSMs are worse at copying ([2402.01032](https://arxiv.org/abs/2402.01032)); state collapse beyond training length ([Stuffed Mamba](https://arxiv.org/abs/2410.07145)) | yes | **Med** (as memory: weak) | M |
| 7 | **Titans / ATLAS / MIRAS / Nested Learning (HOPE)** ([Titans](https://arxiv.org/abs/2501.00663) NeurIPS'25; [ATLAS](https://arxiv.org/abs/2505.23735); [NL](https://arxiv.org/abs/2512.24695) NeurIPS'25) | deep MLP memory updated by a gradient ("surprise") with momentum and decay at test time; HOPE adds multi-frequency "continuum memory" and self-modification | Titans beats Transformer++, Mamba-2 and GDN at 170M–760M, 15–30B tokens; >2M-token NIAH ([html](https://arxiv.org/html/2501.00663v1)); ATLAS +80% at 10M BABILong | **BABILong "beats GPT-4" compares fine-tuned Titans to few-shot GPT-4** (paper's own setup). Reimplementation: neural memory helps vs attention-only, but Titans "does not always outperform established baselines due to chunking"; no official code ([2510.09551](https://arxiv.org/html/2510.09551v1)). TNT: <5–10% FLOP utilization, over-specialized to training chunk size ([2511.07343](https://arxiv.org/html/2511.07343v1)). HOPE reproductions are mechanism-level only, no paper-scale results ([repo](https://github.com/kmccleary3301/nested_learning)) | **no official code**; unofficial [lucidrains](https://github.com/lucidrains/titans-pytorch) | **Low–Med** | **H (highest)**. Viral ([Google blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/), [LW](https://www.lesswrong.com/posts/Lby4gMvKcLPoozHfg/are-we-in-a-continual-learning-overhang-1)); HN: "11 months and you can't download a Titans-architecture model code or weights anywhere" ([HN](https://news.ycombinator.com/item?id=46181231)) |
| 8 | **Large-chunk TTT layers** ([TTT layers](https://arxiv.org/abs/2407.04620); [LaCT](https://arxiv.org/abs/2505.23884) ICLR'26) | nonlinear fast weights updated once per 2K–1M-token chunk, with window attention inside the chunk | state up to 40% of params; up to 70% GPU utilization; 14B video model, 1M-token novel-view synthesis | strongest evidence is in vision/video; LM evidence is thinner | yes ([project](https://tianyuanzhang.com/projects/ttt-done-right/)) | **Med** | L–M |
| 9 | **Memory Mosaics** ([v1](https://arxiv.org/abs/2405.06394) ICLR'25; [v2 at scale](https://arxiv.org/abs/2507.03285) NeurIPS'25) | networks of kernel-regression associative memories ("predictive disentanglement") | v2 at 10B/1T tokens: matches transformers on training knowledge, beats them on new-knowledge / in-context tasks | one group (Meta/NYU); no independent replication found | not verified | **Med–Low** | L |
| 10 | **Latent memory pools & side memories** ([MemoryLLM](https://arxiv.org/abs/2402.04624) ICML'24; [M+](https://proceedings.mlr.press/v267/wang25au.html) ICML'25; [Larimar](https://arxiv.org/abs/2403.11901) ICML'24; [LM2](https://arxiv.org/abs/2502.06049)) | per-layer latent memory tokens (M+ adds a co-trained retriever); Kanerva-style episodic memory for editing; cross-attended memory bank | M+ retention <20k → >160k tokens; Larimar edits 4–10× faster at comparable accuracy | LM2 critique (automated referee, not OpenReview): no baseline isolates the memory module; headline +37.1%/+86.3% not reproducible from Table 1 ([Pith](https://pith.science/paper/2502.06049)). HN: "poorly written and full of errors" (opinion) ([HN](https://news.ycombinator.com/item?id=43042753)) | MemoryLLM/Larimar yes; LM2 unclear | **Low–Med** | L |
| 11 | **Infini-attention** ([2404.07143](https://arxiv.org/abs/2404.07143)) | linear-attention compressive memory per segment, gated into local attention | original: 1M passkey, 500K summarization | **HF replication failed**: "performance gets worse as we increase the number of times we compress the memory"; needle in segment 1 fails; recommends YaRN/ring attention instead ([HF blog](https://huggingface.co/blog/infini-attention)). A 300M pretraining study is positive vs baseline but degrades with repeated compression ([2512.23862](https://arxiv.org/abs/2512.23862)) | no official code | **Low** | L now (was H in 2024) |
| 12 | **Compressive / Memorizing Transformers** ([1911.05507](https://arxiv.org/abs/1911.05507), [2203.08913](https://arxiv.org/abs/2203.08913)) | compressed old activations / kNN lookup over a cache of past KVs | ICLR'20/'22 LM perplexity gains | largely superseded by long-context attention and KV compaction (e.g. [Still](https://arxiv.org/abs/2606.07878)) | yes (unofficial) | Med (historical) | L |

Also noted, newer than September 2025 and single-source for now:
- Fast-weight PKM, a TTT-updated product-key memory; trained at 4K, generalizes to 128K ([2601.00671](https://arxiv.org/abs/2601.00671), Llion Jones).
- An associative-recall curriculum that takes fixed-state recurrences from 0.021 to 1.000 success ([2609.16183](https://arxiv.org/abs/2609.16183)).
- Finding that hybrid designs mostly change how *fast* long-context skill emerges ([2606.15378](https://arxiv.org/abs/2606.15378)).

## 3. Evidence vs. enthusiasm: where they disagree

- **Titans / HOPE: enthusiasm far exceeds evidence.**
  - Peer-reviewed at NeurIPS, but ≤760M params and ≤30B tokens.
  - No official code; the fine-tuned-vs-few-shot BABILong comparison; one mixed reimplementation.
  - The LessWrong "continual learning overhang" post treats Titans/HOPE as the path, and a commenter answers that the
    bottleneck is data "that looks like continual learning" (opinion) ([LW](https://www.lesswrong.com/posts/Lby4gMvKcLPoozHfg/are-we-in-a-continual-learning-overhang-1)).
  - Bing Liu on these systems: "they're probably just making the model forget a little bit less" (opinion)
    ([Transformer](https://www.transformernews.ai/p/teaching-ai-to-continual-learning)).
- **Memory layers and sparse memory finetuning: evidence exceeds enthusiasm.** Matched-FLOP scaling to 128B memory params
  with code, plus an independent low-forgetting replication, yet little forum excitement.
- **Hybrids: roughly aligned.** Both are high, but the "memory" credited to linear layers is mostly done by the attention
  layers (split-prefill), and MiniMax's reversal is the best-documented dissent.
- **Infini-attention: hype, then a failed replication.** Enthusiasm has collapsed, correctly.
- **TTT-E2E: a rigorous paper that reports its own failure.** It is the most honest paper in the cluster: it reports its own
  0.06 passkey score.

## 4. What the discussion gets wrong or over-hypes

1. **Passkey/NIAH treated as "memory."** Retrieving a planted string is the one thing compression-based memories are worst
   at, and full attention is best at (TTT-E2E 0.06 vs 0.99; split-prefill: recurrent state 0% on lookup). NIAH wins from
   memory modules usually come with fine-tuning on the probe. BABILong's value is that it shows LLMs use only 10–20% of
   context ([2406.10149](https://arxiv.org/abs/2406.10149)), not that fine-tuned small memory models "beat GPT-4".
2. **Unmatched comparisons.** Examples:
   - Titans (fine-tuned) vs GPT-4 (few-shot).
   - LM2 trained from scratch vs RMT on a LLaMA backbone.
   - Headline "beats Transformer++" results at 15–30B tokens, where recipe noise is large.
   - Training-FLOP penalties are rarely priced in: TTT-E2E is 3.4× slower at 8K; TNT reports <10% utilization.
3. **"Gemini/DeepSeek uses X."**
   - "Gemini 3 _is_ that architecture" (HN, opinion) has no supporting evidence ([HN](https://news.ycombinator.com/item?id=46181231)).
   - Engram was widely expected in DeepSeek V4 but reportedly is absent ([Kili](https://kili-technology.com/blog/data-story-deepseek-v4); secondary source).
4. **Continual learning conflated with long context.**
   - Titans, TTT-E2E and Cartridges compress *one context*.
   - Sparse memory finetuning and memory layers store *knowledge across time*.
   - These are different problems with different metrics (forgetting vs recall).
5. **"Linear layers are the memory" in hybrids.** In Qwen3.5 the facts live in the attention KV cache
   ([2609.04434](https://arxiv.org/html/2609.04434v1)). Removing attention to "add memory" trades exact recall for style and
   state. MiniMax found that proxy metrics hid this until scale ([blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)).
6. **Missing code is ignored in the hype.** The three most-hyped approaches (Titans, HOPE, Infini-attention) are the three
   without official code.

## 5. Testable on this machine

**Constraints.**
- Hardware: one GB10, sm_121, 119 GB unified memory ([docs/spark-notes.md](../../docs/spark-notes.md)); ~15–20 GPU-min per
  candidate; frozen metric, seeds, held-out split.
- **Model inventory correction:** per spark-notes, **Qwen3-4B and Qwen3-30B-A3B are refs-only stubs (no weights)**.
  Usable models: Qwen3-0.6B/1.7B and Qwen3.5-2B/4B.
- Architectural pretraining (Titans, hybrids, memory layers from scratch) cannot give a meaningful answer in 20 minutes.
  Both proposals therefore **retrofit memory into an existing checkpoint**. Both need PyTorch with gradients (the pytorch
  25.10 image plus `transformers`), not vLLM.
- **Prior result:** the prompt-level memory pack found nothing: 25 candidates, effect < 0.062 < MES 0.08
  ([README](../../README.md)).

### E1 (recommended): `ttt-memory`, query-only test-time training as context memory (Qwen3-1.7B)

- **Hypothesis:** at a fixed per-question compute budget, a few gradient steps on the context (W_Q only, KV frozen after one
  prefill, as in [qTTT](https://arxiv.org/html/2512.13898)) beat plain answering over the same context.
- **Data:** frozen, from PREPARE. Prefer a **generated RULER-style set**: multi-key KV tracking plus transaction-log state
  questions at ~16K tokens, **400 val / 400 held-out**, exact match. qTTT's largest gains were on synthetic retrieval-heavy
  tasks. A secondary check can reuse the pack's LongMemEval questions with a **~16K haystack** (evidence sessions plus
  same-haystack distractors, original order) under the existing scorer and 207/207 split. The full ~115K haystack would cost
  hours per pass.
- **Frozen:**
  - the model;
  - the context;
  - a per-question budget of ≤ 1 prefill + ≤ 32×256 TTT tokens (≈ qTTT's FLOP match to 8K thinking tokens);
  - greedy 16-token answers, no thinking;
  - a hard GPU-memory cap;
  - weights reset per question (a harness check: after reset, the model's logits equal the base).
- **Surface (the agent may change):**
  - which params adapt (W_Q, W_Q+W_K, LoRA on attention, norms);
  - span selection (random vs the model's own attention/loss-selected spans, per [S-TTT](https://arxiv.org/abs/2607.09415));
  - steps, LR, loss (next-token vs query-conditioned).
- **Baseline:** zero TTT steps.
- **Cost:** 16K prefill of a 1.7B model plus 32 short backward steps should be seconds per item. Calibrate it: if 400 items
  exceed ~10 min, cut to 12K tokens.
- **Realistic effect:**
  - Synthetic set: **+3 to +10 pp** plausible (the qTTT paper reports up to +12.6 on its best subsets, with larger models).
  - LongMemEval-16K: **0 to +4 pp**, below an 8-pp MES, so expect "not found" there.
  - S-TTT shows naive random-span TTT can be **negative**.
  - With n=400, MES ≈ 2.5·√(0.2/400) ≈ **0.056**.
  - My prior that a candidate clears it on the synthetic set: ~40%. On LongMemEval-16K: ~10%.

### E2: `memory-layer-cl`, sparse memory-layer retrofit for learning new facts (Qwen3-0.6B)

- **Hypothesis:** new facts written into a zero-initialized product-key memory layer, updating only the top-t slots
  ([Lin+](https://arxiv.org/abs/2510.15103)), learn as well as LoRA with much less forgetting.
- **Data:** frozen generator of ~1,000 fictional-entity facts. Train on 1–2 phrasings per fact; test on **held-out
  paraphrased questions**, split into validation and holdout.
- **Retention guard (frozen):**
  - held-out general-text perplexity within +2% of base;
  - accuracy on a fixed 300-item closed-book QA set the base model already answers, within −2 pp.
- **Metric:** new-fact accuracy, with the run invalid if the guard fails.
- **Baseline:** LoRA r=16 on MLPs, same steps and time.
- **Surface (the agent may change):**
  - memory size, top-k and insertion layer;
  - slot-selection rule (TF-IDF / KL);
  - LR, and value-only vs key+value updates.
- **Cost:** a few minutes of training, which fits.
- **Realistic effect:**
  - Retention should clearly beat LoRA; that is the robust finding, replicated at 0.5B
    ([2605.03229](https://arxiv.org/abs/2605.03229)).
  - **New-fact accuracy at matched steps will probably *trail* LoRA.** The independent replication got only +2.5 pp of
    learning.
  - Paraphrase generalization of injected facts is weak for every method.
  - So expect a win only on a forgetting-constrained metric. On raw new-fact accuracy, expect −10 to +5 pp.
  - Lin et al.'s large gaps used a model *pretrained* with memory layers; a cold retrofit is weaker.

**Cheap one-off diagnostic (not a campaign):** rerun [split-prefill](https://github.com/kirillTerra/split-prefill) on the
cached **Qwen3.5-2B/4B**. It takes minutes and tells us whether any "state-as-memory" idea on the local hybrids is worth a pack.
Per the paper, the answer for factual recall is probably no.
