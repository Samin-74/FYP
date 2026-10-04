# FYP Project Plan — Adversarial Steering of Semantic Tokenizers in Generative Recommender Systems

**Team:** 2 people
**Primary idea:** Idea 1 — Tokenizer Attack (adversarial steering of RQ-VAE semantic tokenizers)
**Integrated fallback:** Idea 3 — Zero-parameter multi-persona detection for shared accounts
**Structure:** Approach A — Audit-first, attack-second (with early feasibility spike)

**Key dates**

| Milestone | Date |
|---|---|
| Project start | Mon 22 Sep 2026 |
| Pivot checkpoint (spike result) | 2 Nov 2026 |
| **Interim report due** | **21 Dec 2026** |
| Project resume (post-break) | 5 Jan 2027 |
| **Presentation deck + final report due** | **21 Apr 2027** |

---

## 1. Motivation and Research Gap

Generative retrieval (e.g. TIGER, Rajput et al. 2023) replaces conventional
candidate-retrieval with a two-stage pipeline:

1. A **content encoder** (Sentence-T5) embeds each item's metadata.
2. A **residual quantizer** (RQ-VAE) converts the embedding into a discrete
   *semantic ID* — a tuple of codebook indices.
3. A **sequence-to-sequence model** (T5) learns to decode the semantic ID of
   the next item a user will interact with, using constrained beam search
   over valid IDs.

Existing adversarial attacks on recommenders (shilling/profile-injection
attacks, gradient attacks on rankers, poisoning of interaction data) all
target the **downstream ranker or the training data**. Closest to this project,
text-rewriting attacks edit item descriptions to promote items in text-aware
and LLM-based recommenders (ATR, CIKM 2024; ID-free / LLM-agent attacks,
2024), but they attack continuous text representations end to end. We found
no published work that targets the **discrete quantization layer** (semantic
IDs) itself: its boundary geometry, ID collisions, or defences at that layer.
This must be re-checked before the final report.

**Research question:** Can a seller, by rewriting only their product
description, push the item's embedding across an RQ-VAE Voronoi cell
boundary — forcing an ID collision with a bestseller or landing on a
high-traffic code prefix — and thereby gain recommendation share, *while the
description remains semantically faithful to the true product*?

*Scope note:* in the classic Beauty data the encoded text is the template
`Title; Brand; Categories; Price`. A seller controls the title (and plausibly
the price); the category list and the template itself are platform-controlled.
"Product description" in this project therefore means the listing title unless
stated otherwise.

**Why this matters:** the attack requires no fake users, no interaction
injection, and no access to the ranker — only the ability to edit one's own
listing text, which every marketplace seller already has.

## 2. Aims and Objectives

**Aim:** Characterise and exploit the adversarial attack surface of the
semantic tokenization layer in generative recommender systems.

**Objectives:**

1. Reproduce a TIGER-style generative retrieval baseline on a standard
   benchmark, matching published metrics.
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
- We currently have the 2023 raw files downloaded; we will first get the 
  classic Beauty working, then optionally add a 2023 category.

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
- Constrained beam search over the trie of valid catalogue IDs.
- `t5-small` for iteration (runs on the local 4080 Super); `t5-base` for
  final reported numbers (university cluster).
- Metrics: Recall@{5,10}, NDCG@{5,10}; target = match TIGER paper numbers on
  the same dataset slice.

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
  only the tokenizer output — the realistic threat model for a seller.
- **Semantic-preservation constraints (both paths):**
  - Sentence-T5 cosine similarity to the original description above a
    tuned threshold;
  - LLM-as-judge equivalence check (local open model, e.g. Llama/Qwen
    instruct, to avoid API cost and rate limits);
  - Perplexity/naturalness screen.
- **Targets:** (a) force ID collision with a chosen bestseller; (b) steer
  onto a high-traffic code prefix shared by popular items.

## 4. Phases and Timeline

### Phase 0 — Foundations (22 Sep – 12 Oct, weeks 1–3)
- Literature review, split by topic (see §8 reading list).
- Repository + experiment-tracking setup; dataset download, 5-core
  filtering, Sentence-T5 embedding pipeline.
- Download the classic Beauty 5-core data (the repo can do this automatically).
- **Output:** annotated bibliography; reproducible data pipeline; embeddings
  cached for the full Beauty catalogue.

### Phase 1 — Baseline reproduction (13 Oct – 2 Nov, weeks 4–6)
- Train RQ-VAE; train TIGER-style retriever; validate against published
  Recall@K / NDCG@K.
- **Parallel feasibility spike (weeks 4–5):** can a gradient perturbation in
  embedding space flip *any* item's semantic ID at all? Cheap, deliberately
  hacky, answer-only.
- **Output:** working baseline with paper-matching metrics; spike verdict.

### ⚑ Pivot checkpoint — 2 Nov 2026
- If the spike shows semantic IDs are effectively unflippable from the text
  side (quantization too lossy), pivot primary direction to **Idea 3**
  (multi-persona shared-account detection), which reuses the same decode-time
  logit-diagnostic infrastructure and the same trained retriever.
- The manipulability audit (Phase 2) stands as a deliverable either way.

### Phase 2 — Manipulability audit (3 – 23 Nov, weeks 7–9)
- Implement boundary-margin, sensitivity, and steerability metrics;
  catalogue-wide run on Beauty; regression vs. item properties.
- **Output:** audit tables + figures — the core preliminary results of the
  interim report.

### Phase 3 — Attack v1 + interim report (24 Nov – 21 Dec, weeks 10–13)
- White-box attack v1 on audit-flagged steerable items; black-box v1 if time
  allows.
- Measure attack success rate (ASR@flip, ASR@target-prefix) and semantic
  preservation.
- First two weeks of December reserved primarily for writing.
- **Output: INTERIM REPORT — due 21 Dec 2026.**

### Phase 4 — Downstream impact (5 Jan – 8 Feb 2027)
- Evaluate the trained retriever against the steered catalogue: does the
  attacked item's recommendation/exposure share actually rise? Effect on
  overall Recall@K/NDCG@K?
- Realistic seller-budget constraints (bounded rewrite count/length).
- Attack v2: adaptive refinements informed by Phase 4 findings.
- **Output:** end-to-end impact study.

### Phase 5 — Defences (9 Feb – 8 Mar 2027)
- Boundary-margin regularisation during RQ-VAE training; stochastic /
  randomised quantization (smoothing); embedding-drift anomaly detection for
  listing edits.
- Adaptive evaluation: re-run the strongest attack against each defence.
- **Output:** defence evaluation + attack–defence arms-race analysis.

### Phase 6 — Ablations, writing, deck (9 Mar – 21 Apr 2027)
- Ablations: codebook size, quantization depth, dataset category
  (Beauty vs. Sports & Outdoors), t5-small vs. t5-base.
- Complete results tables; final report drafting from mid-March.
- Two weeks of slack built in before the deadline.
- **Output: FINAL REPORT + PRESENTATION DECK — due 21 Apr 2027.**

## 5. Team Split

| Person | Track | Phases owned |
|---|---|---|
| **A — model track** | RQ-VAE + TIGER reproduction, downstream evaluation, defences | 1, 4, 5 |
| **B — attack track** | audit metrics, white-box/black-box attack, LLM judge + constraints | spike, 2, 3 |

- **Shared:** literature review (split by topic), all report writing, weekly
  integration sync (same repo, module interfaces via files per §3).
- From Phase 4 onward both members converge on the full pipeline.
- Swap A/B roles as preferred — the tracks are deliberately decoupled until
  January.

## 6. Evaluation Metrics

| Layer | Metrics |
|---|---|
| Baseline retriever | Recall@{5,10}, NDCG@{5,10} vs. published TIGER numbers |
| Audit | % items within ε of a boundary; steerability distribution; regression R²/effects |
| Attack | ASR@flip, ASR@target-prefix, ASR@bestseller-collision; semantic similarity preserved; LLM-judge equivalence rate; perplexity of rewrites; query/rewrite budget used |
| Downstream impact | Δ exposure / recommendation share of target item; Δ Recall@K / NDCG@K catalogue-wide |
| Defences | ASR under defence vs. undefended; utility cost (Δ NDCG) of each defence; detection precision/recall |

**Attack baselines:** random paraphrase, TextAttack-style word substitution
with no embedding-space targeting, and an "unconstrained" attack (no semantic
judge) as an upper bound.

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
- Zhou et al. / follow-ups on semantic-ID recommendation: LETTER (learnable
  item tokenization), LC-Rec, EAGER — for tokenizer variants and training tricks.
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
- *Poison-RAG* (ECIR 2025, arXiv 2501.11759); LLM-driven poisoning of
  embedding-based retrieval-augmented recommenders (arXiv 2505.05196).

**Semantic-ID evaluation and quantization robustness**
- *Faithful Evaluation of Semantic-ID Tokenizers for Generative
  Recommendation* (arXiv 2605.25330): SID-level metrics overstate item-level
  Hit@10 under collisions (affects our baseline numbers).
- Dong & Mao, *Adversarial Defenses via Vector Quantization* (arXiv
  2305.13651) — VQ as a defence; relevant to Phase 5.

**Adversarial text**
- HotFlip (Ebrahimi et al.), BERT-Attack, TextAttack framework (Morris et al.).
- LLM-as-judge methodology for semantic equivalence.

**Fallback (Idea 3)**
- Multi-persona / shared-account recommendation literature; generative
  retrieval decoding diagnostics.

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TIGER baseline doesn't reproduce paper numbers | Medium | High | Start in week 4; cross-check public reimplementations; fall back to "close enough + documented gap" — baseline is a means, not the contribution |
| RQ-VAE training instability / codebook collapse | Medium | Medium | LETTER tricks (k-means init, EMA, commitment loss tuning); smaller codebook ablation |
| Embedding-space flips not realisable in actual text | Medium | High | HotFlip token-level path is the realisable backstop; pivot checkpoint 2 Nov; audit remains a deliverable regardless |
| ID flips but no downstream recommendation effect | Medium | Medium | Frame as rigorous negative result + pivot weight to defences; prefix-level steering may still show inference-time effects |
| Cluster access delayed | Medium | Low | 4080 Super covers all Beauty-scale work; cloud budget for final sweeps |
| LLM judge cost / rate limits | Low | Low | Local open-model judge (Llama/Qwen instruct) |
| Team member unavailable (illness etc.) | Low | High | Decoupled tracks until January; weekly sync keeps both members familiar with both halves |
| Interim writing eats experiment time | High | Medium | First two weeks of December reserved for writing; freeze experiment scope after 5 Dec |
| We only have 2023 data, classic Beauty missing | Medium | Medium | Use the repo’s automatic download for classic Beauty first |

## 10. Interim Report Outline (due 21 Dec 2026)

1. **Introduction & motivation** — generative retrieval pipeline; the
   unexamined quantization attack surface.
2. **Literature review** — generative retrieval + semantic IDs; attacks on
   recommenders; adversarial text; the gap at their intersection.
3. **Methodology** — pipeline (§3), audit metric definitions, threat model.
4. **Preliminary results** — baseline reproduction numbers (Phase 1);
   manipulability audit tables/figures (Phase 2); spike + attack v1 early
   ASR numbers (Phase 3).
5. **Plan & risk management** — Phases 4–6, fallback strategy, risk register.
6. **References.**

## 11. Final Report Outline (due 21 Apr 2027)

Everything in the interim, plus:
- Full white-box/black-box attack evaluation with ablations.
- Downstream impact study (Phase 4).
- Defence proposals and adaptive attack–defence evaluation (Phase 5).
- Discussion: implications for generative-recsys deployment, ethics and
  responsible disclosure, limitations, future work.

## 12. Deliverables Checklist

- [x] Annotated bibliography (end of Phase 0) — `docs/literature_review.md`
- [ ] Reproduced TIGER baseline with paper-matching metrics (Phase 1)
- [ ] Feasibility spike verdict + pivot decision (2 Nov)
- [ ] Manipulability audit: framework + catalogue-wide results (Phase 2)
- [ ] **Interim report (21 Dec 2026)**
- [ ] White-box + black-box attack with ASR numbers (Phases 3–4)
- [ ] Downstream impact study (Phase 4)
- [ ] Defences + adaptive evaluation (Phase 5)
- [ ] Ablation grid (Phase 6)
- [ ] **Final report + presentation deck (21 Apr 2027)**
