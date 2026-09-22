"""Shared artifact paths for the FYP pipeline.

All modules communicate exclusively through files under ``artifacts/``
(plan section 3). Nothing is ever written inside ``RQ-VAE-Recommender-main/``.
"""

from pathlib import Path

FYP_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = FYP_ROOT / "artifacts"
DATA_2023 = FYP_ROOT / "data_raw_2023"

DATASET_ROOT = ARTIFACTS / "dataset" / "amazon"  # repo-style root (raw/ + processed/)
CHECKPOINT = (
    FYP_ROOT
    / "RQ-VAE-Recommender-main"
    / "trained_models"
    / "rqvae_amazon_beauty"
    / "checkpoint_399999.pt"
)

EMBEDDINGS_DIR = ARTIFACTS / "embeddings"
ITEMS_PARQUET = EMBEDDINGS_DIR / "items.parquet"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "item_embeddings.npy"
POPULARITY_PARQUET = EMBEDDINGS_DIR / "popularity.parquet"

SEMANTIC_IDS_DIR = ARTIFACTS / "semantic_ids"
SEMANTIC_IDS_PARQUET = SEMANTIC_IDS_DIR / "semantic_ids.parquet"
CODEBOOKS_NPY = SEMANTIC_IDS_DIR / "codebooks.npy"

AUDIT_DIR = ARTIFACTS / "audit"
AUDIT_ITEMS_CSV = AUDIT_DIR / "audit_items.csv"
AUDIT_SUMMARY_JSON = AUDIT_DIR / "audit_summary.json"

RUNS_DIR = ARTIFACTS / "runs"
