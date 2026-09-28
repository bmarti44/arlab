# Test-time training deep dive: MindsAI, the ARC record, and what transfers to a plastic agent

Date: 2026-09-27. Companion to `RESEARCH.md` in this directory, which is the broad survey. This file covers test-time
training (TTT) only and does not repeat that survey's material on Cartridges, SDFT, forgetting, safety or the driving
analogy.

**Labels:**
- **[F]** fact from a primary source (paper, write-up, official report).
- **[I]** my inference.
- **[G]** gossip or opinion, attributed.
- "(PDF)" means I read the full text rather than the abstract.

**Sources I could not use:**
- Reddit and the ARC Prize Discord: nothing publicly quotable came up.
- Kaggle discussion pages rendered empty to the fetcher.
- X posts returned HTTP 402, so tweets are cited only through search snippets and articles that quote them.

## 1. MindsAI

### 1.1 Who they are [F]

The primary source for this section is the [2025 write-up](https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf) (PDF).

- **Jack Cole** is the team lead:
  - PhD clinical psychologist, still in part-time practice.
  - App developer: "Mind Games" has had more than 30M downloads.
  - About 3.5 years near full-time on ARC.
  - Contracted at Tufa Labs (Zurich) for about a year, until October 2025. Now works independently as Mindware Consulting.
- **Mohamed Osman** (Calgary, MSc in electrical engineering) co-developed TTT for ARC with Cole in 2022–23. He says he did
  "very little to no" work on the 2025 solution.
- **Michael Hodel** created RE-ARC and ARC-DSL. He contributed synthetic data in 2024, including the "ARC 1.5" set.
- **Dries Smit** and **Isaiah Pressman** (Tufa Labs) joined for 2025.
- Earlier collaborators ([Lab42 interview, Apr 2024](https://lab42.global/community-interview-jack-cole/)):
  - Phung Cheng Fei;
  - Matteo Batelic, who first proposed the test-time augmentation that became AIRV.

### 1.2 Timeline [F]

| When | Event |
|---|---|
| Nov 2022 | Cole proposes test-time fine-tuning (TTFT) on *retrieved similar tasks* ("ARTI" video, cited in the 2025 write-up). |
| 2023 | Key discovery: turn each demo pair into a test pair (leave-one-out). Ties for 1st in [ARCathon 2023](https://lab42.global/past-challenges/2023-arcathon/) at **30%** private, shared with Team SM. |
| Apr 2024 | 33% on Kaggle ([Lab42](https://lab42.global/community-interview-jack-cole/)). |
| Jun 2024 | [MLST episode](https://podcasts.apple.com/us/podcast/new-50-arc-result-and-current-winners-interviewed/id1510472996?i=1000659449138) with Cole, Osman and Hodel. The show notes claim 54% on the full public eval and that "60–70% with an ensemble" is possible. |
| ARC Prize 2024 | Private score goes from 33% to **55.5%**, the top score. **Not prize-eligible** because they did not open-source ([report](https://arxiv.org/abs/2412.04604)). The ARChitects won with 53.5%. The ARC Prize blog: "The prize incentive system for open sourcing is not quite compatible with these types of teams (evidenced by MindsAI choosing not to share this year)" ([blog](https://arcprize.org/blog/arc-prize-2024-winners-technical-report)). |
| Mar 2025 | [MLST with Osman](https://podcasts.apple.com/us/podcast/test-time-adaptation-the-key-to-reasoning-with-dl/id1510472996?i=1000700414100), billed as "Tufa Labs (formerly MindsAI)". |
| Jun 2025 | [Cole & Osman paper](https://arxiv.org/abs/2506.14276) (PDF, HTML). |
| Jul–Aug 2025 | Teammate Dries Smit's **StochasticGoose** (Tufa Labs; Cole as adviser) wins the ARC-AGI-3 preview at 12.58% ([ARC Prize](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)). |
| ARC Prize 2025 | **3rd place, 12.64%** on ARC-AGI-2 private ([results](https://arcprize.org/blog/arc-prize-2025-results-analysis)). The write-up says 15.42%, probably a different leaderboard split [I]. Everything was released under MIT: code, checkpoints, and the 100M-example "ARC-AGI Mega" corpus ([repo](https://github.com/jcole75/arc_2025_mindsai)). |
| Jun 2026 | Tufa Labs' "The Duck" wins ARC-AGI-3 Milestone #1 with an **in-context** LLM that plays by writing Python in a REPL, **not weight updates** ([ARC Prize](https://arcprize.org/blog/arc-prize-2026-milestone-1)). Whether Cole was involved is unknown. |

**Rumour to correct:** "Confluence Labs, 97.9% on ARC-AGI-2" (Feb 2026) is **not** Cole's lab.
- It was founded by Brent Burdick and uses LLM program synthesis, at about $12 per task on the *public* eval
  ([YC launch](https://www.ycombinator.com/launches/PWR-confluence-labs-an-ai-research-lab-focused-on-learning-efficiency)).
- Cole only tweeted "Wow! It's over for ARC-AGI-2" ([X](https://x.com/MindsAI_Jack/status/2026270468982882550)).
- Some search summaries wrongly attribute the lab to him.

### 1.3 What they disclosed about the method [F]

**Model** (paper, then write-up):
- Started with LongT5, chosen for its non-causal encoder.
- Moved to **Salesforce CodeT5-Large**, with the decoder pruned from 24 to 16 layers, giving **660M** parameters.
- Encoder–decoder models beat decoder-only models, and encoder depth matters most.

**Pretraining:**
- More than **100M examples, about 70M of them ARC-style**: RE-ARC, Hodel's ARC-1.5, PCFG string tasks, cellular automata,
  Reasoning Gym, and LLM-generated concepts from Andreas Köpf.
- Trained on TPU Research Cloud for **up to about 2.5 years**: v2-8 and v3-8, plus 7 days on a v4-64.
- Extra training objectives: T5 span corruption and reversal augmentations.
- Adding these augmentations lifted zero-shot by about 100% on a checkpoint that had plateaued for four months.

**TTT:**
- **Full fine-tuning**, not LoRA. Cole found LoRA "slightly less performant", and layer-wise tuning comparable but fiddly.
- Trains on **all test tasks at once**, about 45K examples, using the "task permutation" method: each demo pair is held out
  in turn as a pseudo-test.
- Augmentations: D4 geometry, colour permutation, mixup/combine, input–output swap, and BPE dropout (p=0.2 in TTT).

**AIRV** (augment, infer, reverse the augmentation, vote):
- About 10K augmented inferences per task, with board-level voting. Pixel-wise and logit voting were worse.
- Two checkpoints are self-ensembled.

**Ablations** (77M model, ARC-1.5 test set, 5 repeats):

| Condition | Gain |
|---|---|
| TTT alone, vs zero-shot | +430% |
| AIRV alone, vs zero-shot | +412% |
| Both | **+812%**, "almost precisely" additive |
| Seed self-ensembling | +6.2% compute-matched |

- With more TTT and AIRV items, the 77M model reaches 54% (zero-shot about 4%).
- The paper (LongT5): zero-shot 5% → AIRV 13% → TTFT+AIRV 39%.

**Non-additive:**
- Refinement, "TTRL" (DPO on beam pairs), ARC-2-targeted data, model merging and MC dropout each helped *instead of*
  TTT/AIRV, not on top of them.
- Cole: "ARC-AGI-2 proves to be at least somewhat adversarial to the approach."

**Compute:**
- The paper's setting: one P100 (16 GB) and a 2-hour limit.
- The 2024 Kaggle limit: P100, 12 hours, no internet ([report](https://arxiv.org/abs/2412.04604)).

### 1.4 Confirmed versus inferred

**Confirmed:**
- The 2023–25 scores and ranks.
- 2024 ineligibility because they did not open-source.
- The TTFT, AIRV and pretraining recipe as of 2025.

**Inferred:**
- The exact 2024 recipe that scored 55.5% was never released. The 2025 code is its successor, so treat it as a proxy. The
  paper's "58%" best private score presumably comes after the competition.
- Why they did not open-source in 2024: I found no primary statement. The ARC Prize remark about startups and the Tufa Labs
  affiliation point to commercial reasons.
- MindsAI's edge is best explained [I] by years of pretraining on tens of millions of ARC-format tasks, which makes the
  model *TTT-able*. Their own ablation says extensive ARC pretraining was "critical for TTFT performance".

## 1b. Scuttlebutt, opinion and skepticism (all [G] unless marked [F])

**Chollet, positive** (Dwarkesh, Jun 2024; [transcript](https://www.dwarkesh.com/p/francois-chollet)):
- "What Jack Cole is actually doing is that for every test problem, it's on-the-fly fine-tuning a version of the LLM for
  that task. That's really what's unlocking performance."
- "It's actually adding active inference to LLMs. That's working extremely well, actually."

**Chollet, qualified:**
- In the same interview he calls it "very shallow recombination" and wants a hybrid with discrete program search.
- On X (Nov 2024): TTFT and program synthesis "start from a preexisting bank of reusable functions" and recombine them, but
  at "different points" on the memorisation–recombination spectrum ([X](https://x.com/fchollet/status/1856071366996570350)).
- YC talk (2025): test-time adaptation is necessary but not sufficient. Transformers do "type 1" (value-centric)
  abstraction, not "type 2" (program-centric) ([transcript](https://singjupost.com/francois-chollet-how-we-get-to-agi-transcript/)).

**ARC Prize statements** [F]:
- 2024: "there does not exist any static inference-style transduction solution that scores above 10%" ([report](https://arxiv.org/abs/2412.04604)).
- ARC-AGI-2 was explicitly designed to be "less brute-forcible" ([paper](https://arxiv.org/abs/2505.11831)).
- 2025: warns of "knowledge overfitting". The underlying models know ARC well enough to infer its colour mappings from format
  alone ([report](https://arxiv.org/abs/2601.10904)).

**"It's just augmentation plus fine-tuning":**
- Partly supported by MindsAI's own numbers [F]: AIRV alone (+412%) is about as strong as TTT alone (+430%).
- The ARC Prize HRM analysis [F] found the architecture unimportant. What mattered was the refinement loop and training on
  the *evaluation tasks' demos*, a transductive TTT ("a kind of program synthesis substrate",
  [analysis](https://arcprize.org/blog/hrm-analysis); [Chollet](https://x.com/fchollet/status/1956442449922138336)).

**Overfitting to ARC:**
- HN, on the Akyürek paper ([thread](https://news.ycombinator.com/item?id=42108278)):
  - it is "a bunch of human programming work specifically tailored to the problem";
  - it would fail "if the test was presented on a hexagonal grid";
  - defenders point to the private hold-out set.
- LessWrong: "[ARC-AGI is a genuine AGI test but o3 cheated](https://www.lesswrong.com/posts/KHCyituifsHFbZoAC/arc-agi-is-a-genuine-agi-test-but-o3-cheated)"
  argues that training on ARC-like data turns the test into memorisation. That applies *a fortiori* to 70M ARC-style
  examples. Dave Orr's top reply: humans practise too.

**Compute:**
- Cheap at inference: 2 cents per task for the ARChitects ([PoE](https://arxiv.org/abs/2505.07859)).
- Huge upstream: MindsAI's roughly 2.5 years of TPU. NVARC's 4B full fine-tune took 4×8 H100 for 27 hours
  ([NVIDIA](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/)).

**Competitors:**
- Guillermo Barbadillo (2nd in 2024) calls his solution "an extension of the MindsAI approach", and says "LLMs need
  test-time fine-tuning to do new tasks. Few-shot prompting is not enough" ([write-up](https://ironbar.github.io/arc24/05_Solution_Summary/)).

**LessWrong, 2025** ([post](https://www.lesswrong.com/posts/tEZa7PouYatK78bbb/i-am-worried-about-near-term-non-llm-ai-developments)):
- Skeptical of HRM/TTT generality. Cole Wyeth: "Most things don't scale."
- gwern has long argued that "dynamic evaluation" (the older name for LM TTT) is underused for deployment reasons, not
  quality reasons ([gwern](https://gwern.net/doc/ai/nn/dynamic-evaluation/index)).

**My read [I]:** ARC TTT works because of three things together.
1. The test-time loss *is the task*. Leave-one-out demos are exact supervised examples in the test format.
2. Invertible symmetries both multiply the data and supply a vote.
3. The base was pre-trained on the task family.

Agents in new environments get (2) and (3) only if we build them, and (1) only through relabeling (§3).

## 2. The published TTT record

| Work | Task | Model | What is trained | Signal | Gain | Compute | Replicated? |
|---|---|---|---|---|---|---|---|
| [Sun+ 2020](https://arxiv.org/abs/1909.13231) | CIFAR-10-C, ImageNet-C | ResNet | shared encoder, per sample (and online) | rotation prediction | consistent error cuts under shift | ~1 step per sample | Yes (TTT-MAE and others) |
| [Hardt & Sun 2023](https://arxiv.org/abs/2305.18466) | Pile LM | GPT-2, GPT-Neo | full weights, 1 step | next-token on 20 retrieved neighbours | narrows GPT-2 → 10×-larger gap | 1 step per input + index | Extended (SIFT / active fine-tuning) |
| [MindsAI 2025](https://arxiv.org/abs/2506.14276) (PDF) | ARC-1 / ARC-1.5 | T5, 77M–660M | full weights, all tasks at once | leave-one-out demos + augmentation | 5→13→39% (paper); +430%, and +812% with AIRV | P100 2 h (paper); TPU-years pretraining | Yes, technique adopted by all 2024 leaders |
| [Akyürek+ 2024](https://arxiv.org/abs/2411.07279) | ARC public eval | Llama-3 8B (fine-tuned) | per-task LoRA r128 | leave-one-out ICL-format demos, geometric augmentation | 18.3→47.1%; **53.0%** with voting; 61.9% with BARC | ~12 h per 100 tasks on A100 | Yes (ARChitects, NVARC) |
| Akyürek+ (BBH) | BBH 10-shot | Llama-3 8B | per-task LoRA | leave-one-out over demos | 50.5→57.8 | ~15 min per task on A100 | Not independently |
| [ARChitects 2024](https://arxiv.org/abs/2505.07859) | ARC-1 | NeMo-Minitron 8B / Llama 3B | per-task LoRA r32, 64 steps | augmented demos | 18.3→44.5 (8B), 14.9→40.9 (3B); 71.6% full pipeline; Kaggle 53.5% | 51 s per task on a 4090 | Yes |
| [BARC, Li+ 2024](https://arxiv.org/abs/2411.02272) | ARC-1 | Llama-3.1 8B | LoRA (transduction) | demos + augmentation + rerank | 29.1→43%; induction ensemble 56.75% | 10–20K program samples | Yes |
| [Barbadillo](https://ironbar.github.io/arc24/05_Solution_Summary/) | ARC-1 Kaggle | Qwen2.5 0.5B | per-task full or LoRA, ~300 steps | n−1 demos | 11→33 tasks (3×); 40% private | Kaggle P100 | Kaggle-verified |
| [NVARC 2025](https://github.com/1ytic/NVARC) | ARC-AGI-2 | Qwen3 4B | per-puzzle LoRA r256 | augmented demos | **24.03%** private (1st); TTT extension 10→18% pass@1, second-hand via [Trelis](https://trelis.substack.com/p/nvarc-2025-arc-prize-winners) | ~$0.20 per task | Kaggle-verified |
| [ARChitects 2025](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) | ARC-AGI-2 | LLaDA 8B (masked diffusion) | LoRA r32, 128 steps per task | masked-output reconstruction | 16.53% private (2nd) | 16×H100 × 2 weeks dev | Kaggle-verified |
| [HRM analysis](https://arcprize.org/blog/hrm-analysis) | ARC-1 | 27M HRM | whole model, trained *with eval demos* | demos + ~300–1000 augmentations | 32% semi-private (41% claimed); eval-demos-only 31% | GPU-hours | ARC Prize reproduced |
| [TRM](https://arxiv.org/abs/2510.04871) | ARC-1 / ARC-2 | 7M recursive | same transductive setup | demos + augmentation | 45% / 8% (self-reported public); 1st paper award | ~GPU-days | Partly: [TTA of TRM](https://arxiv.org/abs/2511.02886) got 6.67% on ARC-2 semi-private |
| [CompressARC](https://arxiv.org/abs/2512.06104) | ARC-1 | 76K, **no pretraining** | all weights, per puzzle | MDL on the puzzle itself | ~20% eval | minutes per puzzle | 3rd paper award |
| [SOAR](https://arxiv.org/abs/2507.14172) | ARC-1 | Qwen2.5 7–72B | full fine-tune on target tasks | **hindsight-relabeled** search traces | 52% public eval | multi-GPU | 2nd paper award |
| [SEAL](https://arxiv.org/abs/2506.10943) (PDF via RESEARCH.md) | SQuAD no-context; ARC subset | Qwen2.5 7B; Llama-3.2 1B | LoRA on self-edits, RL outer loop | self-generated implications / augmentation configs | 32.7→47.0 (raw passage: 33.5); ARC 20→72.5% on **8 tasks** | 30–45 s per edit evaluation | No |
| [TTRL](https://arxiv.org/abs/2504.16084) | AIME, MATH | Qwen2.5-Math 7B and others | full RL | majority-vote reward | +211% relative pass@1 on AIME24 (abstract) | RL on the test set | Re-run widely; Qwen-Math caveat (RESEARCH.md) |
| [TTT-Discover](https://arxiv.org/abs/2601.16175) | single open problems, kernels | gpt-oss-120b | RL on one problem | task reward, entropic objective | new SOTA on several problems | large | New |
| [TTT layers](https://arxiv.org/abs/2407.04620) | LM up to 32K | 125M–1.3B, trained from scratch | hidden state = model | inner-loop self-supervised reconstruction | keeps improving past 16K, where Mamba stalls | pretraining | Yes; [reinterpreted as linear attention](https://arxiv.org/abs/2602.21204) (ICML 2026) |
| [TTT video](https://arxiv.org/abs/2504.05298) / [LaCT](https://arxiv.org/abs/2505.23884) | 1-min video; NVS, LM | 5B / 14B diffusion | TTT-MLP fast weights; large chunks | inner-loop reconstruction | +34 Elo over Mamba2 / DeltaNet | fine-tune of pretrained model | Partly |
| [Online TTT on video](https://arxiv.org/abs/2307.05014) | segmentation | MAE-based | encoder, updated continually | masked reconstruction | 2.2× / 1.5× vs fixed model; **online beats offline** ("locality") | per frame | — |
| [TTT-E2E](https://arxiv.org/abs/2512.23675) | long-context LM | 3B, 164B tokens | weights, next-token at test time | NTP with a **meta-learned** initialisation | scales like full attention; 2.7× faster at 128K | pretraining | New (Dec 2025) |
| [qTTT](https://arxiv.org/abs/2512.13898) | LongBench-v2, ZeroScrolls, synthetic | Qwen3 1.7B/4B/8B | **W_Q only**, KV frozen | NTP on 128-token context spans, 32 steps | +12.6 / +14.1 (4B) | FLOP-matched to thinking | [S-TTT](https://arxiv.org/abs/2607.09415): random spans *hurt*, oracle spans help; [EASE-TTT](https://arxiv.org/abs/2606.06906) builds on it |
| [aTTT](https://arxiv.org/abs/2607.03441) | ALFWorld, SWE-bench Lite | Qwen3.5 4B/9B/27B | LoRA r8, online in-episode | own tokens, env text or summary | +5.0 / +4.9; summary in weights 54.3 vs in context 49.3 vs none 51.4 (9B) | 1.9× wall-clock | No |
| [GTTA](https://arxiv.org/abs/2511.04847) (ICLR 2026) | WebArena, BFCL | various | per-episode steering vector | format alignment | small; the 2→23% gain is from **in-context** dynamics notes | ~3% latency | No |
| [Early Experience](https://arxiv.org/abs/2510.08558) (train time) | 8 agent envs | ≤ 70B | full / LoRA | **next-state prediction** on own rollouts; self-reflection | +2.3–5.5 ALFWorld/SciWorld; +11–18 WebShop | offline | — |
| [StochasticGoose](https://github.com/DriesSmit/ARC3-solution) | ARC-AGI-3 preview | 4-layer CNN from scratch | all weights, online, reset per level | "did this action change the frame?" | 12.58% (1st); about 350 wasted moves, then exploitation | 1 GPU | Preview only |

**Takeaways:**
- **Strong** (multiple teams, private sets): per-task TTT on ARC-format data with augmentation and voting, on a base that was
  *pre-trained on the task family*.
- **Moderate:** long-context TTT (qTTT, TTT-E2E), where span selection is the crux; next-state-prediction training for
  agents (Early Experience, at training time).
- **Weak** (single paper or tiny n): SEAL on ARC (8 tasks); aTTT (+5 pts); agent TTT in general.
- **The frontier on ARC-AGI-3 is not weight-based yet** [F]:
  - frontier LLMs score under 1% ([paper](https://arxiv.org/abs/2603.24621));
  - the 2026 milestone winner is in-context;
  - the only weight-learning winner is a tiny CNN predicting "did the frame change".

## 3. How it could work for agents: the option space

**(a) Training signal.** Tags: **E** = evidence exists; **A** = evidence in adjacent settings (non-agent, or at train
time); **U** = untested.

| # | Signal | Tag | Evidence | ARC analogue / note |
|---|---|---|---|---|
| a1 | Next-token prediction on raw observations and transcript | E | weak: SEAL raw passage +0.8; aTTT +5 at 9B; S-TTT says random spans hurt | Dynamic evaluation. Cheapest, weakest. |
| a2 | Forward dynamics: "(state summary, call) → observation" | A | Early Experience; StochasticGoose, whose frame-change bit is a 1-bit forward model | Learns the *physics* of the place. |
| a3 | Inverse dynamics: "(before, after) → which call" | U for LLM agents | common in RL pretraining; untested as LLM TTT | Teaches control. |
| a4 | **Hindsight relabeling** of trajectories into (goal → call sequence) demos | A | SOAR (ARC), HER lineage | **The closest analogue to MindsAI's "task permutation":** it converts experience into test-format supervised pairs. |
| a5 | Self-study Q&A, distilled to the ICL teacher (Cartridges, SDFT) | E (documents) | RESEARCH.md §1–2 | A weak teacher at ≤ 3B is the risk. |
| a6 | Self-edits (SEAL): the model writes its own training notes | A | SEAL; the RL outer loop is too costly here | Rule text such as "purge means archive" as training data. |
| a7 | RL from environment reward | A | TTT-Discover, StochasticGoose | Needs reward during exploration. |
| a7b | RL from majority vote (TTRL) | A | TTRL | Needs reward; votes are cheap. |
| a8 | Verifier-filtered successes (STaR / expert iteration at test time) | A | SOAR; NVARC's filtered synthetic data | Needs a verifier; the simulator state is one, but only if the agent can observe it. |
| a9 | **Augmentation consistency** (the AIRV analogue): consistent renaming of object IDs and values, reordering commuting calls, paraphrasing goals; train on variants, vote at inference | U for agents | ARC: AIRV ≈ TTT in size and additive (MindsAI) | Forces the rule, not the instance, into the weights. |

**(b) Where it is stored:**
- **Full weights:** MindsAI; best on ARC, worst for forgetting.
- **LoRA:** ARChitects, NVARC, Akyürek. Hot-swappable in vLLM, and "slightly less performant" than full fine-tuning per
  Cole.
- **Attention-only subsets** (qTTT W_Q): cheap and targeted, but only re-weights *existing* context.
- **Trained KV prefix** (Cartridges): the best retention in RESEARCH.md, and HF-only here.
- **Hypernetwork-generated adapter** ([Doc-to-LoRA](https://arxiv.org/abs/2602.15902), [LatentSkill](https://arxiv.org/abs/2606.06087)
  with +13 on ALFWorld unseen versus textual skills): needs a meta-training corpus.
- **TTT layers / fast weights:** need architecture pretraining. Not feasible here.

**(c) Schedule:**
- **One "sleep" consolidation after exploration.** This is ARC TTT's regime and the cheapest to evaluate.
- **Continual in-episode updates** (aTTT; online video TTT): locality helps, but self-training on its own repeats drifts.
- **Meta-learned initialisation** (TTT-E2E; in effect MindsAI's years of ARC-format pretraining). The largest lever in the
  literature, but it needs a family-level training run.

**(d) Forgetting safeguards** (detail in RESEARCH.md §6):
- one adapter per environment, discarded after use;
- small rank;
- KL to base on generic prompts;
- replay of generic data;
- repeat-n-gram down-weighting (aTTT);
- an explicit capability guard.

**Recombinations:**

| Combination | Status |
|---|---|
| a4 + a9 + (b) LoRA + (c) sleep: the literal ARC recipe transplanted | **untested for agents**, strongest analogue |
| a2 + a4 | untested; Early Experience used a2 alone at train time |
| a5 teacher-KL + a4 targets | untested |
| a7 test-time RL on a hindsight curriculum | untested |
| meta-trained "TTT-able" base on generator-family environments + sleep adapter | untested for LLM agents; MindsAI's regime |

Nothing in the literature does "explore a provably unseen environment, then consolidate, then run held-out tasks with no
transcript" against an ICL arm. RESEARCH.md §3 reaches the same gap finding.

## 4. What is most likely to work

### Ranked (expected value for the goal; [I] throughout)

1. **ARC-style consolidation:** hindsight-relabeled demos (a4) plus forward-dynamics pairs (a2), multiplied by invertible
   augmentation (a9), into a per-environment LoRA after exploration. This reproduces the three ingredients that made ARC TTT
   work: test-format supervision, symmetry augmentation, and per-task adapters.
2. **Add self-distillation from the ICL teacher (a5)** where the teacher is strong enough. It carries what ICL *infers*,
   not just what was seen (RESEARCH.md #1). At 1.7B, use it as a supplementary loss, not the backbone.
3. **Meta-train the base on the environment family** (MAML-lite: many generator seeds × the same TTT procedure) so a few
   steps suffice. This is where MindsAI's and TTT-E2E's gains really came from. It is the most likely *long-run* winner,
   but it is a one-time multi-hour PREPARE job, so it belongs in stage 2.
4. **Test-time RL (a7) from environment reward on a hindsight curriculum.** Powerful when reward exists, but costly and
   unstable at 1.7B.
5. **Continual in-episode updates (aTTT):** mostly an anti-drift fix, worth +5 at 9B.
6. **Hypernetworks** (Doc-to-LoRA style) come after 1–3 exist and can supply meta-training data.
7. **TTT layers** need pretraining, so they are out of scope.

### The pack: `plastic-agent`, in stages

**Stage 0: `ttt-memory` as harness and effect-size check (keep, but as stage 0; not an arm, not dropped).**

*Why stage 0:*
- It is the same machinery: a PyTorch gradient loop on GB10 with Qwen3-1.7B, parameter-subset training, per-item weight
  reset with the "logits equal base after reset" check, and time/memory calibration.
- It has a **published anchor at our exact model size**: qTTT tested Qwen3-1.7B. Failing to reproduce any gain is a cheap,
  early warning that our TTT loop or 1.7B plasticity is broken.
- It needs no simulator.

*Why not an arm of stage 1:*
- qTTT keeps the context in the KV cache and resets weights *per question*.
- The plastic-agent claim is the opposite: consolidate *once per environment*, then answer many held-out tasks with **no
  transcript in context**.
- Merging them would mix two frozen contracts and two metrics, and the eval paths differ (HF with gradients vs vLLM LoRA).

*Why not drop it:*
- It costs well under a day.
- It de-risks the 1.7B TTT loop.
- It has independent value (the memory brief's prior is about 40% that a candidate clears the MES on the synthetic set).

*Design:* keep the memory brief's E1 (W_Q-only qTTT, RULER-style multi-key and state-tracking set, 400 val / 400 holdout),
with these changes:
- **Drop the LongMemEval-16K secondary.** Its expected 0–4 pts is below any MES.
- **Budget:** my estimate is roughly 3–6 s per item (16K prefill plus 32 × 256-token backward passes through a 1.7B model),
  so 400 items may take 20–40 min. Calibrate first; if over budget, use 8K contexts or 16 steps.
- **Span selection is the main surface lever**, per S-TTT and EASE-TTT.

*Gate to stage 1:* the harness checks pass and the timing fits. **The effect size does not gate stage 1**; it only sets
the prior. If stage 0 shows no gain at 1.7B, run stage 1 on Qwen3.5-4B, provided its LoRA path passes a PREPARE smoke test.

**Stage 1: exploration → sleep → held-out tasks.**
- **Reuse RESEARCH.md §7 as-is:**
  - the FauxOS generator with inverted semantics;
  - frozen exploration transcripts;
  - about 40 held-out goals per instance;
  - arms (a) none, (b) ICL, (c) guard, (d) surface adapter;
  - Qwen3-1.7B; MES 0.06.

  This section changes only what TTT evidence argues for:
- **Baseline surface:** naive next-token LoRA on the transcript (a1), expected ≈ (a).
- **Candidates Codex should try first,** in order:
  1. hindsight goal→program demos (a4);
  2. adding dynamics pairs (a2);
  3. adding renaming and reordering augmentation (a9);
  4. adding teacher-KL (a5);
  5. KL-to-base / replay.
- **Frozen rule:** relabeling may only use what the transcript *shows*. The simulator's hidden state is off-limits, or the
  pack leaks the answer key.
- **New control arm (e): a placebo adapter.** Evaluate instance *i* with the adapter trained on instance *i+1*, with no
  extra training.

  Why it matters: TTT gains on ARC and in GTTA partly come from **format** learning. Only d − e measures
  *environment-specific* knowledge. Report d − a, d − e, the ICL gap closure (d−a)/(b−a), and prefill tokens saved.
- **Forgetting guard:** (c) from §7 (a generic battery plus another instance with its transcript in context, within 2 pts
  of base). v2 adds sequential A→B, then a re-test on A.
- **Budget:** §7 estimates about 19 min for 4 instances. Arm (e) adds about 2 min of eval. If calibration overruns, use
  3 instances.

**Stage 2** (only if stage 1 clears the MES): a one-off, frozen PREPARE job that meta-trains a "TTT-able" 1.7B on about
200 training-generator instances with the winning stage-1 procedure (item 3 above). Re-run stage 1 on that base. This is
the MindsAI lesson, and the most likely route to a large effect.

### Honest effect sizes [I]

| Stage | Expected result | My probability |
|---|---|---|
| 0 (synthetic retrieval, 1.7B) | +3 to +10 pts | ~40% that a candidate clears the MES |
| 1, naive next-token | ≈ 0 to +3 over none | — |
| 1, ARC-style surface | +5 to +15 over none; closes 30–60% of the ICL gap; **beats ICL only for transcripts ≥ 20K tokens**, where 1.7B ICL degrades | ~35–45% that it clears MES 0.06 over the naive baseline |
| 1, d − e (environment-specific share) | about half of d − a | — |
| 2 (meta-trained base) | could double stage 1 | ~15% that it runs within budget at all |

The largest single risk is that at 1.7B the relabeled demos teach the *format* but not the inverted semantics. The placebo
arm is there to catch exactly that.
