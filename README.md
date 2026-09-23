# FYP: Adversarial Steering of RQ-VAE Semantic Tokenizers in Generative Recommender Systems

**Team:** Samin · Ibrahim
**Interim checkpoint:** 21 December 2026 (audit + white-box attack results frozen)

We audit the quantization layer of a TIGER-style generative recommender
(Rajput et al., 2023) — built on the unmodified open-source
[RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
(Botta, 2026, MIT; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) — and
ask: *can a seller push their item across an RQ-VAE codebook boundary and
steal recommendation share, without touching the ranker?*

Findings so far are **in embedding space only** (no text-level attack yet). On
classic Amazon Beauty 5-core (12,101 items) the published RQ-VAE checkpoint
leaves 19.0% of items needing a dedup token (≈27% share their full ID with
another item). The attack/sensitivity numbers are being regenerated after the
bug fixes listed below. The earlier headline figures (92% any-flip at
cos ≥ 0.95, etc.) are **invalid** and must not be quoted.

## Repository layout

```
FYP/
├── README.md                      ← you are here
├── THIRD_PARTY_NOTICES.md         ← upstream attribution (MIT)
├── docs/
│   ├── FYP_Project_Plan.md        ← full project plan (source of truth)
│   ├── weekly_timeline.md         ← Week 1-7 work log & team breakdown
│   └── interim_checkpoint.md      ← what is frozen for interim vs. reserved
├── configs/                       ← our gin configs (paths/wandb overridden)
├── fyp/                           ← ALL custom code (see fyp/README.md)
│   ├── common/                    ← data prep, embeddings, semantic IDs, sequences
│   ├── audit/                     ← margin/sensitivity/steerability + figures
│   ├── attack/                    ← white-box / black-box steering + eval
│   └── spike/                     ← end-to-end smoke test (runs, not results)
├── artifacts/                     ← ALL results (git-ignored, regenerable)
│   ├── audit/                     ← audit tables + 7 figures + report_numbers.md
│   ├── semantic_ids/              ← per-item IDs, codebooks, retriever sequences
│   ├── embeddings/                ← 768-d Sentence-T5 catalogue embeddings
│   ├── runs/                      ← white-box evaluation results
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
$PY -m fyp.audit.report_figures                     # 7 figures + report_numbers.md
$PY -m fyp.attack.run_whitebox_eval --n-per-set 200 --cosine 0.95 --level 0
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

**Invalid, pending re-run** of `run_audit`, `report_figures` and
`run_whitebox_eval`: sensitivity flip rate, steerability, and every white-box
ASR (any-flip, prefix, bestseller). Black-box random search is a plumbing stub
and not a reportable result.

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
