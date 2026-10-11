# FYP Project Plan: Adversarial Steering of Semantic Tokenizers in Generative Recommender Systems

**Team:** Samin (model/data track) · Ibrahim (attack/audit track)
**Primary idea:** Idea 1, Tokenizer Attack (adversarial steering of RQ-VAE semantic tokenizers)
**Integrated fallback:** Idea 3, Zero-parameter multi-persona detection for shared accounts
**Structure:** Approach A, audit-first and attack-second (with an early feasibility spike)

**Key dates** (project weeks run Tuesday to Monday; Week 1 = 15–21 Sep 2026,
matching [weekly_timeline.md](weekly_timeline.md))

| Milestone | Date |
|---|---|
| Project start (Week 1) | 15 Sep 2026 |
| Mentor checkpoint (Week 2) | 22 Sep 2026 |
| Pivot checkpoint (end of Week 7) | 2 Nov 2026 |
| Experiment scope frozen for the interim | 5 Dec 2026 |
| **Interim report due (end of Week 14)** | **21 Dec 2026** |
| Project resume (post-break) | 5 Jan 2027 |
| **Presentation deck + final report due** | **21 Apr 2027** |

---

## 1. Motivation and Research Gap

Generative retrieval (e.g. TIGER, Rajput et al. 2023) replaces conventional
candidate retrieval with a two-stage pipeline:

1. A **content encoder** (Sentence-T5) embeds each item's metadata.
2. A **residual quantizer** (RQ-VAE) converts the embedding into a discrete
   *semantic ID*, a tuple of codebook indices.
3. A **sequence-to-sequence model** (T5) learns to decode the semantic ID of
   the next item a user will interact with, using constrained beam search
   over valid IDs.

Existing adversarial attacks on recommenders (shilling/profile-injection
attacks, gradient attacks on rankers, poisoning of interaction data) all
target the **downstream ranker or the training data**. Closest to this project,
text-rewriting attacks edit item descriptions to promote items in text-aware
and LLM-based recommenders (ATR, CIKM 2024; ID-free / LLM-agent attacks,
2024; prompt-unknown promotion attacks, SIGIR 2026), but they attack
continuous text representations end to end. On the tokenizer side, recent
work studies semantic-ID collisions for evaluation (Zhang et al., CIKM 2026)
and unstable code boundaries as a robustness issue (DRQ, 2026), not as an
attack surface. We found no published work that targets the **discrete
quantization layer** (semantic IDs) adversarially: its boundary geometry, ID
collisions, or defences at that layer against a strategic seller. This must
be re-checked before the final report.

**Research question:** Can a seller, by rewriting only their product
description, push the item's embedding across an RQ-VAE Voronoi cell
boundary, forcing an ID collision with a bestseller or landing on a
high-traffic code prefix, and thereby gain recommendation share *while the
description remains semantically faithful to the true product*?

*Scope note:* in the classic Beauty data the encoded text is the template
`Title; Brand; Categories; Price`. A seller controls the title (and plausibly
the price); the category list and the template itself are platform-controlled.
"Product description" in this project therefore means the listing title unless
stated otherwise.

**Why this matters:** the attack requires no fake users, no interaction
injection, and no access to the ranker, only the ability to edit one's own
listing text, which every marketplace seller already has.

## 2. Aims and Objectives

**Aim:** Characterise and exploit the adversarial attack surface of the
semantic tokenization layer in generative recommender systems.

**Objectives:**

1. Reproduce a TIGER-style generative retrieval baseline on a standard
   benchmark, as close to the published metrics as the available compute
   allows (any gap documented).
2. Build a *manipulability audit*: a measurement framework quantifying how
   close real catalogue items sit to quantization boundaries and how
   steerable they are. **(Guaranteed deliverable)**
3. Develop white-box and black-box text-steering attacks constrained to
   preserve true product semantics, and measure their success rate.
4. Quantify downstream impact: does a successfully steered item actually
   gain recommendation/exposure share in the full pipeline?
5. Propose and evaluate defences (robust quantization, drift detection),
   including an adaptive re-attack evaluation.

**Non-goals (YAGNI):** no production-scale deployment, no multi-modal (image)
content, no real-marketplace experiments, no user studies.

## 3. Technical Pipeline

Six modules, each independently testable, communicating through files
(embeddings, semantic IDs, audit tables) so the two team members can work in
parallel.

### 3.1 Data
- **Primary**: Classic Amazon Beauty 5-core (the standard TIGER benchmark).
  The RQ-VAE-Recommender repo already supports it.
- **Secondary / stretch**: Amazon Reviews 2023 Video_Games (after 5-core)
  or a down-sampled Beauty_and_Personal_Care if time allows.
- We have the 2023 raw files downloaded; classic Beauty is in use, and a 2023
  category is optional.

### 3.2 Content encoder
- Sentence-T5 (`sentence-transformers`), 768-d item embeddings.
- This is the input layer of the attack surface: the attacker controls the
  text, the encoder maps it to embedding space.

### 3.3 Tokenizer
- RQ-VAE, 3 quantization levels × 256-entry codebooks (TIGER standard
  config) → semantic ID = 3-tuple of codeword indices + deduplication token
  for collisions.
- Training-stability tricks from the LETTER paper (k-means initialisation,
  commitment loss tuning) to avoid codebook collapse.

### 3.4 Generative retriever
- T5 encoder–decoder trained on chronologically-ordered user interaction
  sequences → generates next-item semantic ID.
- TIGER uses constrained beam search over the trie of valid catalogue IDs;
  the upstream implementation we use samples top-k candidates instead
  (documented in the README).
- `t5-small` for iteration (runs on the local 4080 Super); `t5-base` for
  final reported numbers (university cluster).
- Metrics: Recall@{5,10}, NDCG@{5,10}, reported at item level (not only
  semantic-ID level); target = the TIGER paper numbers on the same dataset
  slice.

### 3.5 Audit module (guaranteed deliverable)
Per item, per quantization level:
- **Boundary margin:** distance in embedding space to the nearest Voronoi
  cell boundary (distance to second-nearest codeword minus nearest).
- **Perturbation sensitivity:** smallest input-space / embedding-space
  perturbation that flips the assigned code (finite-difference and
  Jacobian-based estimates).
- **Steerability score:** composite of margin and sensitivity.
Catalogue-wide outputs:
- Distribution of steerability across all items; percentage of items within
  ε of a boundary.
- Regression of steerability against item properties (popularity /
  interaction frequency, description length, brand, category).
- Collision-group analysis (overlaps with Idea 2's audit, shared code).

### 3.6 Attack module
- **White-box:** projected gradient optimisation in embedding space toward a
  target cell (bestseller collision, or a high-traffic level-1 prefix),
  realised in text via:
  - HotFlip-style gradient-guided token substitution against the Sentence-T5
    encoder, and/or
  - LLM rewrite proposals ranked by embedding-space progress.
- **Black-box:** iterative LLM-rewrite search (evolutionary/bandit) querying
  only the tokenizer output, the realistic threat model for a seller.
- **Semantic-preservation constraints (both paths):**
  - Sentence-T5 cosine similarity to the original description above a
    tuned threshold;
  - LLM-as-judge equivalence check (local open model, e.g. Llama/Qwen
    instruct, to avoid API cost and rate limits);
  - Perplexity/naturalness screen.
- **Targets:** (a) force ID collision with a chosen bestseller; (b) steer
  onto a high-traffic code prefix shared by popular items.

## 4. Phases and Timeline

Weeks follow [weekly_timeline.md](weekly_timeline.md). The two tracks run
in parallel (§5), so Phases 1–3 overlap.

### Phase 0: Foundations (Weeks 1–2, 15–28 Sep)
- Literature review, split by topic (see §8 reading list).
- Repository + experiment-tracking setup; classic Beauty 5-core download
  (the upstream repo does this automatically), 5-core filtering,
  Sentence-T5 embedding pipeline.
- **Feasibility spike (Week 2):** can a gradient perturbation in embedding
  space flip *any* item's semantic ID at all? Cheap, deliberately hacky,
  answer-only. Reported at the Week-2 mentor checkpoint (22 Sep).
- **Output:** annotated bibliography draft; reproducible data pipeline;
  embeddings cached for the full Beauty catalogue; spike verdict.

### Phase 1: Baseline reproduction (model track, Weeks 2–7, 22 Sep – 2 Nov)
- Semantic IDs from the published RQ-VAE checkpoint (Weeks 2–3); retriever
  sequence data (Week 4); TIGER-style retriever trained and evaluated at
  semantic-ID and item level (Week 7). Own RQ-VAE training is deferred to
  after the interim.
- **Output:** working local-scale baseline with a documented gap to the
  paper's numbers.

### Phase 2: Manipulability audit (attack track, Weeks 3–5, 29 Sep – 19 Oct)
- Boundary-margin, sensitivity, and steerability metrics; catalogue-wide run
  on Beauty; regression vs. item properties (Week 3); figure pipeline
  (Week 4); self-review, bug fixes, re-run and checkpoint replication
  (Week 5).
- **Output:** audit tables + figures, the core preliminary results of the
  interim report.

### Phase 3: Attack v1 (attack track, Weeks 4–7, 6 Oct – 2 Nov)
- White-box attack v1 with enforced constraints (Weeks 4–5); text-level
  attack v1 with real re-encoded listing text (Week 6); results assembly
  (Week 7). Black-box search is a plumbing stub only.
- Measure attack success rate (ASR@flip, ASR@target-prefix) and semantic
  preservation.
- **Output:** ASR tables for the interim report.

### ⚑ Pivot checkpoint: 2 Nov 2026 (end of Week 7)
- If the spike and attack v1 had shown semantic IDs to be effectively
  unflippable from the text side (quantization too lossy), the primary
  direction would pivot to **Idea 3** (multi-persona shared-account
  detection), which reuses the same decode-time logit-diagnostic
  infrastructure and the same trained retriever.
- The manipulability audit (Phase 2) stands as a deliverable either way.

### Interim report writing (Weeks 8–14, 3 Nov – 21 Dec)
- Report drafting from the frozen results; experiment scope frozen on
  5 Dec; the first two weeks of December reserved for writing only.
- Optional before the freeze: a small downstream pilot on the trained
  retriever (see Phase 4).
- **Output: INTERIM REPORT, due 21 Dec 2026.**

### Phase 4: Downstream impact (5 Jan – 8 Feb 2027)
- Evaluate the trained retriever against the steered catalogue: does the
  attacked item's recommendation/exposure share actually rise? Effect on
  overall Recall@K/NDCG@K?
- Realistic seller-budget constraints (bounded rewrite count/length).
- Attack v2: LLM rewrites and HotFlip-style substitution, informed by
  Phase 4 findings; black-box search.
- **Output:** end-to-end impact study.

### Phase 5: Defences (9 Feb – 8 Mar 2027)
- Boundary-margin regularisation during RQ-VAE training; stochastic /
  randomised quantization (smoothing); embedding-drift anomaly detection for
  listing edits.
- Adaptive evaluation: re-run the strongest attack against each defence.
- **Output:** defence evaluation + attack–defence arms-race analysis.

### Phase 6: Ablations, writing, deck (9 Mar – 21 Apr 2027)
- Ablations: codebook size, quantization depth, dataset category
  (Beauty vs. Sports & Outdoors), t5-small vs. t5-base.
- Complete results tables; final report drafting from mid-March.
- Two weeks of slack built in before the deadline.
- **Output: FINAL REPORT + PRESENTATION DECK, due 21 Apr 2027.**

## 5. Team Split

| Person | Track | Phases owned |
|---|---|---|
| **Samin (A, model track)** | data pipeline, RQ-VAE + TIGER reproduction, downstream evaluation, defences | 0 (data), 1, 4, 5 |
| **Ibrahim (B, attack track)** | audit metrics, white-box/black-box attack, LLM judge + constraints | spike, 2, 3 |

- **Shared:** literature review (split by topic), all report writing, weekly
  integration sync (same repo, module interfaces via files per §3).
- From Phase 4 onward both members converge on the full pipeline.
- The tracks are deliberately decoupled until January.

## 6. Evaluation Metrics

| Layer | Metrics |
|---|---|
| Baseline retriever | Recall@{5,10}, NDCG@{5,10} at item level vs. published TIGER numbers |
| Audit | % items within ε of a boundary; steerability distribution; regression R²/effects |
| Attack | ASR@flip, ASR@target-prefix, ASR@bestseller-collision; semantic similarity preserved; LLM-judge equivalence rate; perplexity of rewrites; query/rewrite budget used |
| Downstream impact | Δ exposure / recommendation share of target item; Δ Recall@K / NDCG@K catalogue-wide |
| Defences | ASR under defence vs. undefended; utility cost (Δ NDCG) of each defence; detection precision/recall |

**Attack baselines:** random paraphrase, TextAttack-style word substitution
with no embedding-space targeting (no target-guided search), and an
"unconstrained" attack (no semantic judge) as an upper bound. The interim's
random-vocabulary control still uses the target-guided search, so the
no-search baseline remains to be run.

## 7. Compute Plan

- **Local (4080 Super, 16 GB):** all development and iteration; t5-small
  retriever; RQ-VAE training; Beauty-scale audit and attack runs.
- **University GPU cluster:** t5-base final runs; full catalogue-wide audit
  sweeps; Phase 4–5 large evaluations; ablation grid.
- **Cloud budget (contingency):** held in reserve for cluster-access delays
  and the final ablation sweep; also covers any burst LLM-judge needs if a
  local judge proves inadequate.
- **Rule of thumb:** nothing should be cluster-only. Every experiment must
  have a reduced-scale variant that runs locally overnight.

## 8. Key Literature (starting reading list)

Full annotated bibliography with verified citations:
[docs/literature_review.md](literature_review.md).

**Generative retrieval / semantic IDs**
- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER), NeurIPS 2023.
- Follow-ups on semantic-ID recommendation: LETTER (learnable item
  tokenization), LC-Rec, EAGER, for tokenizer variants and training tricks.
- RQ-VAE: Lee et al., *Autoregressive Image Generation using Residual
  Quantization* (RQ-VAE origin), plus VQ-VAE (van den Oord et al.).

**Attacks on recommenders**
- Shilling / profile-injection attacks (Lam & Riedl; Burke et al.).
- Data-poisoning and adversarial item-promotion attacks on neural
  recommenders (e.g. AUSH and follow-ups).

**Text-rewriting attacks on recommenders (closest prior work)**
- *Adversarial Text Rewriting for Text-aware Recommender Systems* (ATR),
  CIKM 2024, arXiv 2408.00312.
- *ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free
  Recommender Systems*, arXiv 2409.11690.
- *Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender
  Systems*, SIGIR 2026, arXiv 2604.23640.
- Nazary, Deldjoo & Di Noia, *Poison-RAG*, arXiv 2501.11759; Nazary et al.,
  *Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based
  Retrieval-Augmented Recommender Systems*, arXiv 2505.05196.

**Semantic-ID evaluation and quantization robustness**
- Zhang et al., *Faithful Evaluation of Semantic-ID Tokenizers for Generative
  Recommendation*, CIKM 2026, arXiv 2605.25330: SID-level metrics overstate
  item-level Hit@10 under collisions (affects our baseline numbers).
- Wang et al., *Decoupled Residual Quantization for Robust Semantic IDs in
  Recommendation*, arXiv 2606.01844: boundary instability as a robustness
  issue; Phase-5 defence baseline.
- Dong & Mao, *Adversarial Defenses via Vector Quantization* (arXiv
  2305.13651): VQ as a defence; relevant to Phase 5.

**Adversarial text**
- HotFlip (Ebrahimi et al.), BERT-Attack, TextAttack framework (Morris et al.).
- LLM-as-judge methodology for semantic equivalence.

**Fallback (Idea 3)**
- Multi-persona / shared-account recommendation literature; generative
  retrieval decoding diagnostics.

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TIGER baseline doesn't reproduce paper numbers | Medium (realised: local baseline is below the paper at item level) | High | Fall back to "close enough + documented gap"; the baseline is a means, not the contribution; paper-scale t5-base run on the cluster after the interim |
| RQ-VAE training instability / codebook collapse | Medium | Medium | LETTER tricks (k-means init, EMA, commitment loss tuning); smaller codebook ablation |
| Embedding-space flips not realisable in actual text | Medium (partly retired: real text edits flip IDs) | High | HotFlip token-level path is the realisable backstop; pivot checkpoint 2 Nov; audit remains a deliverable regardless |
| ID flips but no downstream recommendation effect | Medium | Medium | Frame as rigorous negative result + pivot weight to defences; prefix-level steering may still show inference-time effects |
| Cluster access delayed | Medium | Low | 4080 Super covers all Beauty-scale work; cloud budget for final sweeps |
| LLM judge cost / rate limits | Low | Low | Local open-model judge (Llama/Qwen instruct) |
| Team member unavailable (illness etc.) | Low | High | Decoupled tracks until January; weekly sync keeps both members familiar with both halves |
| Interim writing eats experiment time | High | Medium | Weeks 8–14 reserved for writing; experiment scope frozen on 5 Dec |
| Classic Beauty data unavailable (only 2023 dumps) | Low (retired: classic Beauty downloaded in Week 2) | Medium | 2023-dump fallback path implemented in `prepare_data` |

## 10. Interim Report Outline (due 21 Dec 2026)

1. **Introduction & motivation**: generative retrieval pipeline; the
   unexamined quantization attack surface.
2. **Literature review**: generative retrieval + semantic IDs; attacks on
   recommenders; adversarial text; the gap at their intersection.
3. **Methodology**: pipeline (§3), audit metric definitions, threat model.
4. **Preliminary results**: baseline reproduction numbers (Phase 1);
   manipulability audit tables/figures (Phase 2); spike + attack v1 early
   ASR numbers (Phase 3).
5. **Plan & risk management**: Phases 4–6, fallback strategy, risk register.
6. **References.**

## 11. Final Report Outline (due 21 Apr 2027)

Everything in the interim, plus:
- Full white-box/black-box attack evaluation with ablations.
- Downstream impact study (Phase 4).
- Defence proposals and adaptive attack–defence evaluation (Phase 5).
- Discussion: implications for generative-recsys deployment, ethics and
  responsible disclosure, limitations, future work.

## 12. Deliverables Checklist

- [x] Annotated bibliography (Phase 0; drafted Weeks 1–2, verified and
      finalised Week 7): `docs/literature_review.md`
- [x] Feasibility spike verdict (Week 2) and pivot decision (2 Nov):
      feasible, continue with Idea 1
- [x] Manipulability audit: framework + catalogue-wide results (Phase 2)
- [x] TIGER baseline reproduced at local scale (Phase 1); item-level
      Recall@10 0.017–0.021 vs the paper's 0.0648, a documented gap
      (paper-scale run deferred to the cluster)
- [x] White-box attack v1 with ASR numbers (Phase 3)
- [x] Text-level attack v1 with ASR numbers and a random-vocabulary control
      (Phase 3)
- [ ] **Interim report (21 Dec 2026)**
- [ ] Black-box attack + attack v2 (LLM rewrites / HotFlip) (Phase 4)
- [ ] Downstream impact study (Phase 4)
- [ ] Defences + adaptive evaluation (Phase 5)
- [ ] Ablation grid (Phase 6)
- [ ] **Final report + presentation deck (21 Apr 2027)**
