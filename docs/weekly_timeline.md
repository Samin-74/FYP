# Weekly Timeline & Work Breakdown (Weeks 1–7: Sept 15 – Nov 2)

Anchored to the Week-2 mentor update (Sept 22). Track split follows the
project plan §5: **Samin — model/data track**, **Ibrahim — attack/audit
track**; documentation and mentor checkpoints shared. Hours kept roughly
equal; each week lists the concrete artefacts produced.

> **Mentor anchor (Sept 22, Week 2):**
> *"We finished our initial exploration on the GPU setup using the Amazon Beauty
> dataset (12k+ items). We generated all the Semantic IDs through Sentence-T5 +
> RQ-VAE and ran some quick baseline audits to see if the system is actually
> vulnerable. According to our results it was actually fragile, around 19% of
> items share duplicate codes, 20% are sitting right on the decision boundary,
> and our basic vector attack managed to flip codes 62.5% of the time (5 out of
> 8 test items). Now that the feasibility check is done, we are thinking of
> moving on to actual text-level attacks with LLMs…"*
>
> *Note: the 62.5% spike figure and the "20% on the boundary" phrasing were
> superseded by the Week-5 bug fix and re-run (see below); the 19% duplicate
> figure stands.*

## Week 1 (Sept 15 – 21) — Environment & literature

| Person | Work |
|---|---|
| Samin | Local GPU environment (RTX 4080 SUPER): uv-managed Python 3.12 venv, CUDA torch (cu128) install, editable install of upstream RQ-VAE repo, upstream pytest sanity (15/15). Literature: TIGER (Rajput et al.), RQ-VAE (Lee et al.). |
| Ibrahim | Literature: LETTER (tokenizer stability), adversarial text attacks (HotFlip, TextAttack), attacks-on-recommenders survey. Workspace layout + gin config overrides (`configs/`). Dataset feasibility review (classic Beauty 5-core vs 2023 dumps). |

## Week 2 (Sept 22 – 28) — Data pipeline & feasibility spike — *mentor checkpoint Sept 22*

| Person | Work |
|---|---|
| Samin | Classic Beauty 5-core acquisition via upstream auto-download; fixed the P5_data.zip string-user-id incompatibility and Windows extract-lock retry (`fyp/common/prepare_data.py`). sentence-t5-xxl encoding pipeline (fp16; fp32 does not fit 16 GB), full-catalogue embeddings cached (`artifacts/embeddings/`). |
| Ibrahim | Published RQ-VAE checkpoint loading (`fyp/common/generate_semantic_ids.py`); semantic IDs for all 12,101 items; **feasibility spike: 5/8 (62.5%) code flips** on a hand-picked near-boundary sample; quick duplicate-code count (≈19%). **Mentor update delivered Sept 22.** |

## Week 3 (Sept 29 – Oct 5) — Catalogue-wide audit

| Person | Work |
|---|---|
| Samin | Batched GPU ID/margin computation; per-item `semantic_ids.parquet` (codes + d1 + margins, 3 levels); codebook dumps; popularity table from interaction logs (`popularity.parquet`). |
| Ibrahim | Boundary-margin metric formalized (d2 − d1, residual space, upstream assignment rule); batched gradient sensitivity on a 2,048-item sample wired into `run_audit`; steerability composite (exp(−sens/τ)); collision-group analysis (983 colliding groups, largest 248); popularity/margin regressions (R² ≈ 0 — boundary margin is unrelated to popularity). |

## Week 4 (Oct 6 – 12) — White-box attack suite & figure pipeline

| Person | Work |
|---|---|
| Samin | Automated report figure generation (`fyp/audit/report_figures.py`: figures + `tab_summary_stats.csv` + `report_numbers.md`); downstream retriever data prep (`sequences.parquet`, 22,363 users); workspace cleanup (artifacts out of the upstream tree, .gitignore, licensing/attribution), root README + THIRD_PARTY_NOTICES. |
| Ibrahim | Constraint enforcement inside the white-box optimiser (cosine projection ≥ 0.95, norm preservation); evaluation runner (`run_whitebox_eval.py`) with four goals (any-flip, bestseller prefix, full-ID bestseller collision, high-traffic prefix) and a random-item comparison set; zero-order black-box stub (plumbing check, not a result); `fyp/README.md` reproduction guide; end-to-end smoke test (`fyp/spike/test_pipeline.py`). |

## Week 5 (Oct 13 – 19) — Pre-interim self-review, bug fix & full re-run

| Person | Work |
|---|---|
| Ibrahim | Self-review found seven bugs invalidating the first attack numbers (wrong-way margin search, broken cosine-projection bisection, unconstrained any-flip, norm loophole, prefix-only "bestseller collision", flip-rate denominator, unreachable budget label); fixes on main + interim; full pipeline re-run from semantic IDs onward. Re-run headline: 100% of the 2,048-item sample flips within the budget, ≥ 95% within one 0.05-norm step (p99 0.092); constrained white-box ASR regenerated (traffic prefix 97–98.5%, bestseller prefix ~70%, full-ID collision 13.5–17%). |
| Samin | Checkpoint-robustness replication: audit on `checkpoint_high_entropy.pt` (also 100% one-step flips; 84.2% collision rate — a second checkpoint does not yield more unique IDs). Embedding-geometry calibration added to the audit (norm percentiles; nearest-other-item cosine — 48.5% of items have another product within cos 0.95). Category-extraction fix (second-level category) and figure regeneration. |

## Week 6 (Oct 20 – 26) — Text-level attack v1

| Person | Work |
|---|---|
| Ibrahim | Rule-based text operators (`fyp/attack/text_edits.py`): benign edits (spelling variants, word shuffle/dropout, case/punctuation, field reorder) and seller steering edits (keyword append/prepend, brand swap). Benign-edit stability run (500 items × 5 ops): **60.4% of surface-level edits (not guaranteed paraphrases) change the full semantic ID, 27.4% flip level 0** (mean cos 0.992). Greedy keyword steering onto the high-traffic prefix: **30% ASR** (24% at cos ≥ 0.95), mean 1.9 edits. |
| Samin | Encoder-consistency harness (fresh vs cached encodings: median cos 0.9999999, full-ID agreement 98.5% — the noise floor for all flip rates); target-prefix vocabulary mining (top-1% popular items on code 208); attack-examples write-up (`text_eval_examples.md`); Windows torch.compile/triton workaround for upstream training entry points. |

## Week 7 (Oct 27 – Nov 2) — TIGER baseline & interim assembly — *pivot checkpoint Nov 2*

| Person | Work |
|---|---|
| Samin | TIGER decoder baseline on the published tokenizer (`configs/decoder_beauty_fyp.gin`, 10k iterations, t5-small-scale T5): **Recall@1 0.023 / Recall@5 0.053 / Recall@10 0.073 / NDCG 0.045** (semantic-ID-level, i.e. prefix hits over collision groups — not comparable with the paper's item-level numbers); checkpoint archived (`artifacts/checkpoints/decoder/amazon/`); baseline numbers file (`artifacts/runs/decoder_baseline.json`). |
| Ibrahim | Interim results assembly: audit tables + figures, white-box and text-level attack tables, calibration arguments (cosine budget, margin-threshold caveats), threats-to-validity section. |
| Both | Interim report draft: motivation, methodology, preliminary results, revised plan & risk management. **Nov 2 pivot checkpoint:** embedding-space *and* text-level feasibility confirmed; downstream impact on the trained retriever is the open question for Phase 4. |

## Contribution summary

| Area | Samin | Ibrahim |
|---|---|---|
| Environment / GPU / tooling | Primary | Support |
| Data pipeline & embeddings | Primary | Review |
| RQ-VAE / semantic IDs | Primary | Review |
| Audit metrics & analysis | Support | Primary |
| Attack module (white-box, text-level, black-box) | Review | Primary |
| Retriever baseline | Primary | Review |
| Figures & report tables | Primary | Support |
| Docs & reproducibility | Shared | Shared |

## After Week 7 (per plan §4)

- Own RQ-VAE training + audit replication on it; paper-scale retriever
  (t5-base, full iterations) on the cluster.
- Attack v2: LLM paraphrases / rewrite search on the `text -> text` operator
  interface; HotFlip-style token substitution.
- Phase 4: downstream impact study on the trained retriever (Δ exposure of
  steered items, Δ Recall@K/NDCG@K).
- Interim report due **Dec 21, 2026**.
