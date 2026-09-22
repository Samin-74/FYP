# Weekly Timeline & Team Work Breakdown (Weeks 1–7: Sept 15 – Nov 2)

Anchored to the actual Week-2 mentor update (Sept 22). Track split follows the
project plan §5: **Samin — model/data track**, **Ibrahim — attack/audit track**;
documentation and mentor checkpoints shared. Hours are kept roughly equal.

> **Mentor anchor (Sept 22, Week 2):**
> *"We finished our initial exploration on the GPU setup using the Amazon Beauty
> dataset (12k+ items). We generated all the Semantic IDs through Sentence-T5 +
> RQ-VAE and ran some quick baseline audits to see if the system is actually
> vulnerable. According to our results it was actually fragile, around 19% of
> items share duplicate codes, 20% are sitting right on the decision boundary,
> and our basic vector attack managed to flip codes 62.5% of the time (5 out of
> 8 test items). Now that the feasibility check is done, we are thinking of
> moving on to actual text-level attacks with LLMs…"*

## Week 1 (Sept 15 – 21) — Environment & literature

| Person | Work |
|---|---|
| Samin | Local GPU environment (RTX 4080 SUPER): uv-managed Python 3.12 venv, CUDA torch (cu128) install, editable install of upstream repo, upstream pytest sanity (15/15). Literature: TIGER (Rajput et al.), RQ-VAE (Lee et al.). |
| Ibrahim | Literature: LETTER (tokenizer stability), adversarial text (HotFlip, TextAttack), attacks on recommenders survey. Workspace layout + gin config overrides (`configs/`). Dataset feasibility review (classic Beauty vs 2023 dumps). |

## Week 2 (Sept 22 – 28) — Data pipeline & feasibility spike — *mentor checkpoint Sept 22*

| Person | Work |
|---|---|
| Samin | Classic Beauty 5-core acquisition: upstream auto-download; fixed the P5_data.zip string-user-id incompatibility and Windows extract-lock retry (`fyp/common/prepare_data.py`). sentence-t5-xxl encoding pipeline: fp16 loading (fp32 does not fit 16 GB), full-catalogue embeddings cached (`artifacts/embeddings/`). |
| Ibrahim | Published RQ-VAE checkpoint loading (`fyp/common/generate_semantic_ids.py`); semantic IDs for all 12,101 items; **feasibility spike: 5/8 (62.5%) code flips** on a hand-picked near-boundary sample; quick duplicate-code count (≈19%). **Mentor update delivered Sept 22.** |

## Week 3 (Sept 29 – Oct 5) — Catalogue-wide margins

| Person | Work |
|---|---|
| Samin | Refactored ID/margin computation to batched GPU paths; exported per-item `semantic_ids.parquet` (codes + d1 + margins, 3 levels); codebook dumps. |
| Ibrahim | Boundary-margin metric formalized (d2 − d1 in residual space, upstream assignment rule); first margin distribution analysis per level. |

## Week 4 (Oct 6 – 12) — Sensitivity at scale

| Person | Work |
|---|---|
| Samin | Batched gradient-ascent sensitivity (`min_flip_perturbation_batch`, ~40× speedup), 2,048-item sample wired into `run_audit`. |
| Ibrahim | Steerability composite (exp(−sens/τ)); collision-group analysis (983 colliding groups, largest 248 items); per-item audit table. |

## Week 5 (Oct 13 – 19) — Item-property regressions

| Person | Work |
|---|---|
| Samin | Popularity table from interaction logs; description-length and category extraction from item text; joins into audit table. |
| Ibrahim | Popularity/margin OLS regressions (R² ≤ 1e-4 — manipulability is popularity-independent); category and description-length stratification of sensitivity. |

## Week 6 (Oct 20 – 26) — Constrained attacks & figure suite

| Person | Work |
|---|---|
| Samin | Automated report figure generation (`fyp/audit/report_figures.py`: 7 figures + `tab_summary_stats.csv` + `report_numbers.md`); downstream retriever data prep (`sequences.parquet`, 22,363 users). |
| Ibrahim | Constraint enforcement inside the white-box optimiser (cosine-projection, ≥ 0.95); evaluation runner (`run_whitebox_eval.py`): any-flip 92.0% (near-boundary) vs 50.0% (random), prefix steering 47.5%, bestseller collision 16.5%; zero-order black-box benchmark (0/8) documented as the motivation for LLM-guided search. |

## Week 7 (Oct 27 – Nov 2) — Reproducibility & interim freeze

| Person | Work |
|---|---|
| Samin | Workspace cleanup (runtime artifacts out of upstream tree, .gitignore, licensing/attribution), root README + THIRD_PARTY_NOTICES. |
| Ibrahim | `fyp/README.md` reproduction guide (commands, parameters, upstream workarounds), end-to-end regression verification (8/8 PASS), interim-results table assembly. |
| Both | Interim report draft: motivation, methodology, preliminary results (audit tables + white-box evaluation), plan & risk management. **Freeze.** |

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
