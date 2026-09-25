# Weekly Timeline & Team Work Breakdown

**As of 25 Sep 2026 (Week 1 of the plan).** Week numbering follows the project
plan §4 (Week 1 = 22–28 Sep). We started a week early (15 Sep) and pulled the
feasibility spike + a preliminary audit/attack forward ahead of their planned
phases, so the pivot checkpoint on 2 Nov is already informed by real numbers.
Track split follows plan §5: **Samin — model/data track**, **Ibrahim —
attack/audit track**; documentation and mentor checkpoints shared.

> **Mentor anchor (22 Sep, Week 1):**
> *"We finished our initial exploration on the GPU setup using the Amazon Beauty
> dataset (12k+ items). We generated all the Semantic IDs through Sentence-T5 +
> RQ-VAE and ran some quick baseline audits to see if the system is actually
> vulnerable. According to our results it was actually fragile, around 19% of
> items share duplicate codes, 20% are sitting right on the decision boundary,
> and our basic vector attack managed to flip codes 62.5% of the time (5 out of
> 8 test items). Now that the feasibility check is done, we are thinking of
> moving on to actual text-level attacks with LLMs…"*
>
> **Correction (sent after the 23 Sep self-review):** the 62.5% (5/8) spike
> result and the first white-box numbers came from code with a wrong-way margin
> search and a broken cosine projection, and "20% on the decision boundary" was
> a distribution-shape statistic, not an absolute one. The 19% duplicate-code
> figure stands. Superseded numbers are marked in the root README ("Bugs
> fixed"); valid numbers are the post-fix re-run (25 Sep) in
> `artifacts/audit/report_numbers.md` and `artifacts/runs/whitebox_eval.json`.

## Pre-phase (15 – 21 Sep) — Early start: environment & literature

| Person | Work |
|---|---|
| Samin | Local GPU environment (RTX 4080 SUPER): uv-managed Python 3.12 venv, CUDA torch (cu128) install, editable install of upstream repo, upstream pytest sanity (15/15). Literature: TIGER (Rajput et al.), RQ-VAE (Lee et al.). |
| Ibrahim | Literature: LETTER (tokenizer stability), adversarial text (HotFlip, TextAttack), attacks on recommenders survey. Workspace layout + gin config overrides (`configs/`). Dataset feasibility review (classic Beauty vs 2023 dumps). |

## Week 1 (22 – 28 Sep) — Data pipeline, pulled-forward feasibility, bug fix & re-run

**22 Sep — pipeline + spike + mentor checkpoint:**

| Person | Work |
|---|---|
| Samin | Classic Beauty 5-core acquisition via upstream auto-download; fixed the P5_data.zip string-user-id incompatibility and Windows extract-lock retry (`fyp/common/prepare_data.py`). sentence-t5-xxl encoding pipeline (fp16; fp32 does not fit 16 GB), full-catalogue embeddings cached (`artifacts/embeddings/`). |
| Ibrahim | Published RQ-VAE checkpoint loading (`fyp/common/generate_semantic_ids.py`); semantic IDs for all 12,101 items; **feasibility spike: 5/8 (62.5%) code flips** on a hand-picked near-boundary sample (later found to be from buggy code — see 23 Sep); quick duplicate-code count (≈19%, stands). **Mentor update delivered 22 Sep.** |

**22–23 Sep — preliminary audit + attack suite (plan Phase 2/spike, pulled forward; first commit `99b78a8`):**

| Person | Work |
|---|---|
| Samin | Batched GPU ID/margin computation; per-item `semantic_ids.parquet` (codes + d1 + margins, 3 levels); codebook dumps; popularity table from interaction logs; automated figure suite (`fyp/audit/report_figures.py`: 7 figures + `tab_summary_stats.csv` + `report_numbers.md`); retriever data prep (`sequences.parquet`, 22,363 users); workspace cleanup (.gitignore, attribution), root README. |
| Ibrahim | Boundary-margin metric (d2 − d1, residual space); batched sensitivity (`min_flip_perturbation_batch`, 2,048-item sample); steerability composite; collision-group analysis (983 groups, largest 248); popularity/margin regressions (R² ≈ 0); cosine-constrained white-box optimiser + evaluation runner (`run_whitebox_eval.py`); black-box query-only stub (plumbing check only); `fyp/README.md` reproduction guide; end-to-end smoke test (`fyp/spike/test_pipeline.py`). |

**23 Sep — pre-interim self-review & bug fix (`46f3059`, main + interim):** seven
bugs invalidated every attack number from the 22–23 Sep runs (wrong-way margin
search, broken cosine projection, unconstrained any-flip, norm loophole,
prefix-only "bestseller collision", flip rate divided by the whole catalogue,
unreachable budget label). All fixed; audit extended with collision-group share
and embedding-geometry calibration (norms, nearest-other-item cosine). Old
numbers marked invalid in the README. See root README, "Bugs fixed".

**25 Sep — re-run on the fixed code:** full pipeline from semantic IDs onward
(semantic IDs → audit, n=2,048 → figures/`report_numbers.md` → white-box eval,
n=200/set, cos ≥ 0.95 + norm-preserving). Headline: level-0 codes are *far*
more fragile than the buggy numbers suggested — 100% of the audit sample flips
within one 0.05-norm step; constrained white-box ASR in
`artifacts/runs/whitebox_eval.json`. Calibration finding: 48.5% of items
already have a *different* product within cos 0.95, so that budget does not by
itself preserve meaning. Secondary audit on `checkpoint_high_entropy.pt` to
check the results are not a quirk of one checkpoint. `report_figures` UTF-8
write fix.

## Upcoming (per plan §4)

- **Rest of Phase 0 (→ 12 Oct):** annotated bibliography write-up; pipeline is
  already done (ahead of schedule).
- **Phase 1 (13 Oct – 2 Nov):** train our own RQ-VAE + TIGER retriever;
  baseline Recall@K/NDCG@K (interim outline §10 expects these numbers).
  Re-run the audit on our own checkpoint as well as the published ones.
- **Text-level attack path (pulled forward from Phase 3):** benign-paraphrase
  ID-stability test (~500 LLM-paraphrased descriptions) + LLM-rewrite search
  on near-boundary items — the first real-text results, currently missing.
- **2 Nov:** pivot checkpoint (already informed: embedding-space feasibility
  confirmed; text-side realisation is the open question).
- **Phases 2–3 (3 Nov – 21 Dec):** full audit write-up with the fixed numbers,
  attack v1 text results, interim report (due **21 Dec 2026**).

## Contribution summary

| Area | Samin | Ibrahim |
|---|---|---|
| Environment / GPU / tooling | Primary | Support |
| Data pipeline & embeddings | Primary | Review |
| RQ-VAE / semantic IDs | Primary | Review |
| Audit metrics & analysis | Support | Primary |
| Attack module (white-box, black-box) | Review | Primary |
| Figures & report tables | Primary | Support |
| Docs & reproducibility | Shared | Shared |
