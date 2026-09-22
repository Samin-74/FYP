# FYP — Manipulability Audit Pipeline

Adversarial steering of RQ-VAE semantic tokenizers in generative recommender
systems. This package sits **on top of** `RQ-VAE-Recommender-main/` (installed
editable, never modified) and implements the preliminary pipeline:

```
data -> Sentence-T5 embeddings -> RQ-VAE semantic IDs -> manipulability audit
                                                          -> attack stubs
```

All modules communicate through files under `artifacts/` (project plan §3), so
the model track and attack track can run in parallel.

## Layout

| Path | Contents |
|---|---|
| `fyp/common/` | data prep, embeddings/semantic IDs, sequences, shared paths |
| `fyp/audit/` | margin/sensitivity/steerability metrics, catalogue runner, report figures |
| `fyp/attack/` | white-box targeted flip, black-box stub, constraints, eval runner |
| `fyp/spike/` | end-to-end verification + throwaway experiments |
| `artifacts/` | every pipeline output (embeddings, IDs, audit tables/figures, runs) |
| `configs/` | our gin configs (upstream defaults overridden: paths, wandb off) |
| `data_raw_2023/` | 2023 Amazon dumps — secondary/fallback only |

## Environment

```bash
# from the FYP root (once)
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e "RQ-VAE-Recommender-main[test]"
# CUDA-enabled torch (PyPI Windows wheels are CPU-only):
uv pip install --python .venv/Scripts/python.exe --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cu128 torch
```

## Run everything, in order

```bash
cd c:/Users/chowd/Downloads/FYP
PY=.venv/Scripts/python.exe

# 0. Sanity: upstream tests still pass (CPU is fine)
$PY -m pytest RQ-VAE-Recommender-main/tests -q

# 1. Data: classic Amazon Beauty 5-core auto-download (fallback: 2023 dump),
#    then Sentence-T5-xxl encoding -> artifacts/embeddings/   [one-time, ~1 h]
$PY -m fyp.common.prepare_data

# 2. Semantic IDs + boundary margins -> artifacts/semantic_ids/   [~1 min]
$PY -m fyp.common.generate_semantic_ids

# 3. Catalogue-wide audit: margins for all items + batched gradient sensitivity
#    on a 2048-item sample + popularity/category/description-length joins
#    -> artifacts/audit/audit_items.csv, audit_summary.json   [~2 min]
$PY -m fyp.audit.run_audit --sensitivity-sample 2048

# 4. Publication figures + headline numbers
#    -> artifacts/audit/fig_*.png, tab_summary_stats.csv, report_numbers.md
$PY -m fyp.audit.report_figures

# 5. White-box attack evaluation on audit-flagged items (any-flip, bestseller
#    collision, high-traffic prefix; cosine >= 0.95 enforced by projection)
#    -> artifacts/runs/whitebox_eval.{json,csv}   [~10-30 min]
$PY -m fyp.attack.run_whitebox_eval --n-per-set 200 --cosine 0.95 --level 0

# 6. (Optional, Phase 1 model track) semantic-ID sequences for the retriever
$PY -m fyp.common.prepare_sequences

# 7. Quick regression check: end-to-end mini-batch, prints PASS/FAIL
$PY -m fyp.spike.test_pipeline --n-items 64 --n-attack 8
```

Steps 2-5 are the interim-report reproduction path; step 1 is the only
expensive one and its outputs are cached under `artifacts/`.

## Module interactions (file contracts)

```
prepare_data ──► artifacts/embeddings/{item_embeddings.npy, items.parquet,
                                        popularity.parquet}
     │
generate_semantic_ids ──► artifacts/semantic_ids/{semantic_ids.parquet,
                                                   codebooks.npy, meta.json}
     │
run_audit ──► artifacts/audit/{audit_items.csv, audit_summary.json}
     │
report_figures ──► artifacts/audit/{fig_*.png, tab_summary_stats.csv,
                                    report_numbers.md}
     │
run_whitebox_eval ──► artifacts/runs/whitebox_eval.{json,csv}
     ▲ (reads audit_items.csv to select near-boundary items)
prepare_sequences ──► artifacts/semantic_ids/sequences.parquet (retriever input)
test_pipeline ── end-to-end regression check (writes nothing)
```

## Key parameters

| Where | Parameter | Default | Notes |
|---|---|---|---|
| `audit.metrics.min_flip_perturbation_batch` | `step_size`, `max_steps`, `norm_cap`, `batch` | 0.05 / 200 / 50 / 512 | batched gradient-ascent flip distance |
| `audit.metrics.steerability` | `tau` | median sensitivity of flippable sample | composite score = `exp(-sens/tau)` |
| `attack.white_box.targeted_flip` | `cosine_threshold`, `max_steps` | 0.95 / 300 | enforced by projection after every step |
| `attack.run_whitebox_eval` | `--n-per-set` | 200 | near-boundary + random comparison sets |
| `attack.constraints` | cosine threshold | 0.95 | semantic preservation, embedding-space proxy |
| `audit.run_audit` | `--eps` | 0.25 0.5 1.0 2.0 5.0 | multiples of the median margin for "% within ε" |
| `audit.run_audit` | `--level`, `--sensitivity-sample` | 0 / 2048 | which RQ level sensitivity is measured on |

## Metric definitions (plan §3.5)

- **Boundary margin** — for item embedding `x`, level-k residual `r_k`, and
  squared-L2 distances to the level-k codebook: `d2 - d1` where `d1 < d2` are
  the two nearest codeword distances (same assignment rule as the upstream
  quantizer). Small margin = near a Voronoi boundary.
- **Sensitivity** — smallest input-space L2 perturbation that flips the level-k
  code, estimated by following the margin gradient w.r.t. `x` until the argmin
  changes. `inf` if the norm cap is exceeded.
- **Steerability** — `exp(-sensitivity / median_sensitivity)` on items that
  flip within budget, 0 otherwise.

## Dataset switching

- **Classic Beauty (primary):** nothing to do — `prepare_data` auto-downloads.
- **2023 Beauty (fallback):** if the Google Drive download fails, the same
  command rebuilds the raw layout from `data_raw_2023/` (5-core filter,
  chronological ordering) and reuses the upstream processor.
- **Force reprocessing:** `python -m fyp.common.prepare_data --force-process`.
- **Other Amazon split (sports/toys):** regenerate via the upstream repo's own
  gin configs; the audit reads whatever is in `artifacts/embeddings/`.

## Reproducing the TIGER baseline (Phase 1, model track)

```bash
# tokenizer (or reuse the published checkpoint: RQ-VAE-Recommender-main/
# trained_models/rqvae_amazon_beauty/checkpoint_399999.pt — already used above)
$PY RQ-VAE-Recommender-main/train_rqvae.py configs/rqvae_beauty_fyp.gin
# retriever (t5-small locally; t5-base on the cluster)
$PY RQ-VAE-Recommender-main/train_decoder.py configs/decoder_beauty_fyp.gin
```

Our gin configs redirect all dataset caches and checkpoints into
`artifacts/` and disable wandb; the upstream tree is never written to.

## Known upstream issues & our workarounds (upstream files untouched)

1. **sentence-transformers 5.x incompatibility** — upstream `process()` passes a
   raw pandas Series to `SentenceTransformer.encode(sentences=...)`; ST 5.7
   renamed the argument to `inputs` and rejects Series. Fixed at runtime by
   monkeypatching `PreprocessingMixin._encode_text_feature` in
   `fyp/common/prepare_data.py` (list conversion + positional call).
2. **sentence-t5-xxl OOM on 16 GB** — xxl is ~19 GB in fp32. We load it in
   fp16 (`model_kwargs={"dtype": "float16"}`), batch size 32. Embedding
   differences vs fp32 are ~1e-3 cosine; the audit always uses the same
   encoder, so comparisons are internally consistent.
3. **P5_data.zip ships string user ids** — upstream `train_test_split` does
   `map(int, ...)` over every token and crashes. `prepare_data` downloads and
   extracts the zip itself (same torch_geometric helpers), then rewrites
   `sequential_data.txt` with integer user ids (stable mapping) before the
   upstream processor runs.
4. **Google Drive extract file-lock (Windows)** — the post-extract rename can
   hit a transient `PermissionError`; the download retries up to 3 times.

First-run cost: dataset zip download (seconds) + one-time sentence-t5-xxl
encoding of the ~12k-item catalogue (~30-60 min on the 4080 Super, cached in
`artifacts/dataset/amazon/processed/` afterwards).

## Hard rules

- Never edit anything inside `RQ-VAE-Recommender-main/` (runtime import and
  monkeypatching only).
- Audit numbers must always come from the same encoder the baseline uses
  (`sentence-t5-xxl`).
- Every experiment has a reduced-scale local variant (plan §7).
