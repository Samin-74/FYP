# FYP: Adversarial Steering of RQ-VAE Semantic Tokenizers in Generative Recommender Systems

**Team:** Samin · Ibrahim
**Interim checkpoint:** 21 December 2026 (audit + white-box attack results frozen)

We audit the quantization layer of a TIGER-style generative recommender
(Rajput et al., 2023) — built on the unmodified open-source
[RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
(Botta, 2026, MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) — and
ask: *can a seller push their item across an RQ-VAE codebook boundary and
steal recommendation share, without touching the ranker?*

Findings so far: the published RQ-VAE checkpoint on classic Amazon Beauty
5-core (12,101 items) leaves 19.0% of items needing a dedup token (≈27% share
their full ID with another item); every sampled item's level-0 code flips with
a single embedding perturbation of norm ≤ 0.05 (the search's step size, so an
upper bound); **surface-level text perturbations change the full semantic ID in
60% of edits**; and greedy keyword stuffing (mined from popular items,
including other brands' names) steers 30% of 50 near-boundary items onto the
high-traffic level-0 prefix. A local-scale TIGER retriever baseline is trained
(Recall@10 0.073). The earlier
headline figures (92% any-flip at cos ≥ 0.95, etc.) came from buggy code and
are **invalid** — the numbers below are the re-run (25 Sep) on the fixed
pipeline.

## Repository layout

```
FYP/
├── README.md                      ← you are here
├── THIRD_PARTY_NOTICES.md         ← upstream attribution (MIT)
├── docs/
│   ├── FYP_Project_Plan.md        ← full project plan (source of truth)
│   ├── weekly_timeline.md         ← dated work log & team breakdown
│   └── interim_checkpoint.md      ← what is frozen for interim vs. reserved
├── configs/                       ← our gin configs (paths/wandb overridden)
├── fyp/                           ← ALL custom code (see fyp/README.md)
│   ├── common/                    ← data prep, embeddings, semantic IDs, sequences
│   ├── audit/                     ← margin/sensitivity/steerability + figures
│   ├── attack/                    ← white-box / black-box / text-level steering + eval
│   ├── eval/                      ← TIGER retriever baseline launcher
│   └── spike/                     ← end-to-end smoke test (runs, not results)
├── artifacts/                     ← results: report files (figures/numbers/JSONs) committed;
│   │                                data, embeddings, checkpoints regenerable (git-ignored)
│   ├── audit/                     ← audit tables + 5 figures + report_numbers.md
│   ├── audit_high_entropy/        ← same audit, high-entropy checkpoint
│   ├── semantic_ids/              ← per-item IDs, codebooks, retriever sequences
│   ├── embeddings/                ← 768-d Sentence-T5 catalogue embeddings
│   ├── checkpoints/               ← trained retriever (decoder baseline)
│   ├── runs/                      ← white-box / text-level / baseline results
│   └── dataset/amazon/            ← classic Beauty 5-core (auto-downloaded)
├── RQ-VAE-Recommender-main/       ← unmodified upstream (MIT, vendored)
└── data_raw_2023/                 ← 2023 dumps, secondary/fallback (git-ignored)
```

## Quick start

```bash
cd <path-to>/FYP                   # the repository root
PY=.venv/Scripts/python.exe        # see fyp/README.md for env setup (uv)

$PY -m fyp.common.prepare_data                      # one-time data+embeddings (~1 h)
$PY -m fyp.common.generate_semantic_ids             # semantic IDs + margins
$PY -m fyp.audit.run_audit --sensitivity-sample 2048
$PY -m fyp.audit.report_figures                     # 5 figures + report_numbers.md
$PY -m fyp.attack.run_whitebox_eval --n-per-set 200 --cosine 0.95 --level 0
$PY -m fyp.attack.run_text_eval                     # text-level: benign edits + steering (~40 min)
$PY -m fyp.eval.train_decoder_baseline configs/decoder_beauty_fyp.gin   # TIGER baseline (~1.5 h)
$PY -m fyp.spike.test_pipeline --n-items 64 --n-attack 8   # smoke test (checks it runs, not results)
```

Full setup, module contracts, parameters, and the documented upstream
workarounds: **[fyp/README.md](fyp/README.md)**.

## Key interim results (Amazon Beauty, 12,101 items)

**Still valid** (unaffected by the fixes below):

| Finding | Number |
|---|---|
| Items needing a dedup token (1 − unique IDs / items) | **19.0%** (9,805 unique IDs) |
| Items sharing their full ID with ≥ 1 other item | **≈27%** (983 colliding groups; largest 248 items) |
| Items within 0.25× median boundary margin (level 0) | 19.4%, a *relative* measure (≈16% expected for any exponential-shaped distribution), so not evidence of fragility on its own |
| Margin–popularity correlation | R² ≤ 1e-4 (none) |

**Re-run on the fixed code (25 Sep)** — sensitivity (unconstrained, norm-preserving,
level 0, sample of 2,048) and white-box ASR (cos ≥ 0.95 **and** original norm
enforced by projection, n = 200 per set, level 0):

| Finding | Number |
|---|---|
| Sensitivity flip rate (sample of 2,048) | **100%**, all within one gradient step, so ‖δ‖ ≤ 0.05 (the step size; the reported median 0.050 is the step size, not a measured minimum); effective budget 10 |
| Any-flip ASR | **100%** near-boundary, **100%** random (mean cos 0.9987 = one 0.05 step on the unit sphere, far inside the 0.95 budget) |
| High-traffic prefix ASR | **97.0%** near-boundary, **98.5%** random (mean cos ≈ 0.983) |
| Bestseller prefix ASR (level-0 code of a top-1% item) | 69.5% near-boundary, 72.0% random (mean cos ≈ 0.976) |
| Bestseller full-ID collision ASR (all 3 levels) | 13.5% near-boundary, 17.0% random (mean cos ≈ 0.98) |

Read-outs from the re-run:

- The quantizer is **far more fragile than the buggy numbers suggested**: the
  wrong-way search understated flippability. Any item's level-0 code moves with
  ‖δ‖ ≤ 0.05 on the unit sphere (cos ≥ 0.9987). The true minimum is below the
  search's resolution; for scale, the median distance to the nearest *other*
  product is ≈ 0.32 (cos 0.949).
- **Near-boundary ≈ random everywhere** — when every item flips trivially,
  margin-based target selection adds nothing.
- **The cos ≥ 0.95 budget is weak for this catalogue**: 48.5% of items already
  have a *different* product within cos 0.95 (median nearest-other-item cosine
  0.949). Successes satisfy the constraint yet need not preserve the product.
- **Not a quirk of one checkpoint**: the same audit on
  `checkpoint_high_entropy.pt` (shipped with upstream; its training setup is
  undocumented) also gives a 100% one-step flip rate (margins ~4× larger, still
  ≤ one 0.05 step). Its full-ID collision rate is much *worse* — 84.2% (1,908
  unique IDs; 92.1% of items in a colliding group) — so a second tokenizer
  checkpoint does not yield more unique full IDs.
  (`artifacts/audit_high_entropy/`).
- Black-box random search is a plumbing stub and not a reportable result.

**Text-level results (25 Sep, `fyp.attack.run_text_eval`)** — real listing text,
re-encoded through the same sentence-t5-xxl and re-quantized
(`artifacts/runs/text_eval.json`, examples in `text_eval_examples.md`):

| Finding | Number |
|---|---|
| Surface-level perturbations changing the full semantic ID | **60.4%** of 2,500 edits (500 items × 5 operators) |
| Surface-level perturbations flipping the level-0 code | **27.4%** (spelling only 1.0%; word shuffle 39.6%; case/punctuation 34.2%; field reorder 33.8%; word dropout 28.4%) |
| Flip rate at cos ∈ [0.99, 1.0] | **20.5%** — even near-identical text lands across a boundary |
| Text steering ASR (greedy keyword stuffing → high-traffic prefix, 50 smallest-margin items) | **30%** = 15/50 (95% Wilson CI ≈ 19–44%; 24% at cos ≥ 0.95), mean cos of successes 0.962, mean 1.9 keyword edits |
| Encoder floor (re-encoding identical text) | level-0 agreement 99.5%, full-ID 98.5% — benign flip rates are far above this noise floor |

Read-outs: fragility is **not** an embedding-space artefact — small
surface-level text changes routinely cross codebook boundaries, and the
cos ≥ 0.95 attack budget is far looser than real edits need (these edits
average cos 0.992). The operators are surface perturbations, not guaranteed
paraphrases: word shuffle scrambles the title, word dropout can remove key
words, case/punctuation stripping splits numbers ("12.5" → "12 5"), field
reorder changes the platform template rather than seller text, and most
spelling edits leave the text unchanged. Steerability is real but bounded at
text level: 30% by 1–2 keyword edits vs 97% with free embedding movement.
The successful steering edits are **keyword stuffing, not faithful rewrites**:
the vocabulary is mined from popular items on the target prefix and includes
other brands' product names (e.g. "regenerist" added to nail glitter, "cream"
added to a bath sponge; see `text_eval_examples.md`). They show that the
cosine check does not catch misleading edits; they do not show that a
semantically faithful rewrite can steer an item.

**TIGER retriever baseline (25 Sep, `artifacts/runs/decoder_baseline.json`)** —
upstream decoder trained on the published tokenizer (t5-small-scale T5,
10k iterations, batch 640, 1h27m local): **Recall@1 0.023, Recall@5 0.053,
Recall@10 0.073, NDCG 0.045** (upstream's NDCG is computed over the top-10
generated IDs, i.e. effectively NDCG@10). This is at or above the TIGER paper's
Beauty numbers (Recall@5 0.0454, Recall@10 0.0648, NDCG@10 0.0384), not
below them; differences in the evaluation protocol (split, dedup handling,
beam search) may explain the gap, so the paper figures are a reference rather
than a like-for-like comparison. The checkpoint (`artifacts/checkpoints/decoder/amazon/
checkpoint_9999.pt`) is the substrate for the downstream-impact study.

### Known limitations (to improve; numbers above are unaffected)

These are method limits in the current code, not errors in the reported
numbers. They are left as they are for the interim and listed in full in
[docs/interim_checkpoint.md](docs/interim_checkpoint.md#known-limitations-future-improvements).

- Sensitivity search resolution is the 0.05 step size, so every value reads ≈ 0.050.
- The steerability/sensitivity figures (vs popularity, description length,
  category) therefore plot a near-constant; their visible variation is noise.
- Most `spelling` edits are no-ops and still count as edits.
- The steering vocabulary includes other brands' product names.
- Benign edits are compared with stored (not freshly re-encoded) embeddings;
  "high-traffic prefix" is picked from ~121 top items only.

### Bugs fixed (why the old attack numbers are invalid)

1. **Margin search went the wrong way.** `min_flip_perturbation*` *ascended*
   the margin d2 − d1, i.e. moved items away from their nearest boundary; it
   now descends it. All sensitivity / steerability / any-flip numbers came from
   the wrong search.
2. **Cosine projection was broken.** The bisection in the cosine projection
   collapsed its interval, so each step either halved the perturbation or reset
   it to the original. The "cos ≥ 0.95" attacks were really capped at
   ‖δ‖ ≈ one step (0.05), not at cos = 0.95.
3. **Any-flip was not cosine-constrained.** It now enforces the same cos ≥ 0.95
   budget as the targeted goals.
4. **Norm loophole.** Cosine is scale-invariant, so an attack could stretch the
   embedding at cos = 1 into vectors no text can produce. All attacks now
   re-project onto the original embedding norm after every step.
5. **"Bestseller collision" only matched level 0.** Now split into
   `bestseller_prefix` (level 0) and `bestseller_collision` (full 3-level ID).
   Items already on the high-traffic prefix are steered to the second most
   frequent prefix instead of an arbitrary `code+1`.
6. **Flip rate divided by the whole catalogue.** It was averaged over all
   12,101 items although only 2,048 were tested (reported 9.2%, true ≈ 54%
   under the old, wrong search). It is now computed over the sample.
7. **Budget label.** The search can reach at most
   `min(norm_cap, max_steps × step_size)` = 10, not 50; reported as
   `effective_budget`.

The audit also now records embedding-norm percentiles and each item's cosine
to its nearest *other* catalogue item, to check whether cos ≥ 0.95 really
preserves the product's meaning.

All numbers are regenerated by the commands above; the authoritative copy for
the report is `artifacts/audit/report_numbers.md`.
