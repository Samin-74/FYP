"""Data preparation for the manipulability audit pipeline.

Primary path: classic Amazon Beauty 5-core, auto-downloaded by the upstream
repo (``AmazonReviews`` fetches ``P5_data.zip`` from Google Drive on first
use) into ``artifacts/dataset/amazon/raw/beauty/``.

Fallback path: if the Google Drive download fails (quota / network), build the
same raw layout (``sequential_data.txt``, ``datamaps.json``, ``meta.json.gz``)
from the 2023 dumps in ``data_raw_2023/`` with a standard 5-core filter and
chronological per-user ordering. The upstream ``AmazonReviews.process()`` is
then reused unchanged for embeddings + splits.

Exports (the audit's input contract):
  - artifacts/embeddings/item_embeddings.npy   (n_items, 768) float32
  - artifacts/embeddings/items.parquet         item_id, text
  - artifacts/embeddings/popularity.parquet    item_id, n_interactions

Usage:
  python -m fyp.common.prepare_data [--split beauty] [--force-process]
"""

import argparse
import gzip
import json
import os
import shutil
from collections import Counter

import numpy as np
import polars as pl
import torch

from data.preprocessing import PreprocessingMixin
from data.processed import ItemData
from data.processed import RecDataset

from fyp.common.paths import (
    DATA_2023,
    DATASET_ROOT,
    EMBEDDINGS_DIR,
    EMBEDDINGS_NPY,
    ITEMS_PARQUET,
    POPULARITY_PARQUET,
)

SPLIT = "beauty"

RAW_FILES = ["sequential_data.txt", "datamaps.json", "meta.json.gz"]


def _patch_encoder_batch_size() -> None:
    """Speed up the upstream text encoder at runtime (batch_size=2 -> 128).

    Runtime monkeypatch only; upstream files are not modified. Encoding
    numerics are identical (same model, same sentences).
    """

    def _fast_encode(self, text_feat, model=None):
        if model is None:
            model = self._encoder_model()
        # sentence-transformers 5.x renamed `sentences`->`inputs` and rejects
        # pandas Series; pass a plain list positionally.
        return model.encode(
            [str(s) for s in text_feat],
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()

    def _encoder_model(self):
        if not hasattr(self, "_cached_st_model"):
            from sentence_transformers import SentenceTransformer

            # xxl is ~19 GB in fp32 — does not fit a 16 GB card; fp16 required.
            self._cached_st_model = SentenceTransformer(
                "sentence-transformers/sentence-t5-xxl",
                model_kwargs={"dtype": "float16"},
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        return self._cached_st_model

    # PreprocessingMixin._encode_text_feature signature is (text_feat, model=None)
    PreprocessingMixin._encode_text_feature = _fast_encode
    # Attach lazy model cache; AmazonReviews instances delegate to the class
    # method, so the cache lives on the class.
    PreprocessingMixin._encoder_model = _encoder_model


def raw_split_dir(root, split: str) -> str:
    return os.path.join(root, "raw", split)


def raw_files_present(root, split: str) -> bool:
    d = raw_split_dir(root, split)
    return all(os.path.exists(os.path.join(d, f)) for f in RAW_FILES)


def _sanitize_user_ids(seq_path: str) -> None:
    """Rewrite sequential_data.txt with integer user ids (stable order-of-appearance).

    The current P5_data.zip ships alphanumeric user ids that upstream's
    ``train_test_split`` (``map(int, ...)``) cannot parse; we normalize the
    raw file in our own artifacts dir. No-op when already integer.
    """
    with open(seq_path) as f:
        lines = [ln.strip().split() for ln in f if ln.strip()]
    try:
        for toks in lines[:100]:
            int(toks[0])
        return  # already int-like
    except ValueError:
        pass
    uid_map = {}
    with open(seq_path, "w") as f:
        for toks in lines:
            uid = uid_map.setdefault(toks[0], len(uid_map) + 1)
            f.write(f"{uid} {' '.join(toks[1:])}\n")
    print(f"[prepare_data] sanitized {len(uid_map)} string user ids -> ints")


def try_repo_download(root, split: str) -> None:
    """Download + extract the upstream P5 data ourselves, then sanitize user ids.

    Replicates the upstream ``AmazonReviews.download`` steps (same torch_geometric
    helpers) because instantiating AmazonReviews directly would crash inside
    ``process()`` on the zip's string user ids before we get a chance to fix them.
    """
    import time

    from torch_geometric.data import download_google_url, extract_zip

    _patch_encoder_batch_size()  # upstream encode() breaks on ST 5.x otherwise

    gd_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    last_err = None
    for attempt in range(3):
        try:
            path = download_google_url(gd_id, root, "P5_data.zip")
            extract_zip(path, root)
            os.remove(path)
            folder = os.path.join(root, "data")
            raw_dir = os.path.join(root, "raw")
            if os.path.exists(folder):
                if os.path.exists(raw_dir):
                    shutil.rmtree(raw_dir)
                os.replace(folder, raw_dir)
            break
        except PermissionError as e:  # transient Windows file lock
            last_err = e
            print(f"[prepare_data] extract locked (attempt {attempt + 1}/3), retrying...")
            time.sleep(5)
    else:
        raise last_err

    _sanitize_user_ids(os.path.join(raw_split_dir(root, split), "sequential_data.txt"))


def build_raw_from_2023(root, split: str = SPLIT) -> None:
    """Construct classic-Beauty raw files from the 2023 dump (5-core, chronological)."""
    reviews_path = DATA_2023 / f"Beauty_and_Personal_Care.jsonl"
    meta_path = DATA_2023 / f"meta_Beauty_and_Personal_Care.jsonl"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Fallback reviews file missing: {reviews_path}")

    print("[prepare_data] Building 5-core sequences from 2023 dump (streaming)...")
    lf = (
        pl.scan_ndjson(str(reviews_path))
        .select(["user_id", "asin", "timestamp"])
        .filter(pl.col("user_id").is_not_null() & pl.col("asin").is_not_null())
    )
    df = lf.collect(streaming=True)

    # 5-core filter: users and items each with >= 5 interactions (iterate to fixpoint)
    for _ in range(5):
        n = df.height
        user_counts = df.group_by("user_id").len().rename({"len": "u_len"})
        item_counts = df.group_by("asin").len().rename({"len": "i_len"})
        df = (
            df.join(user_counts, on="user_id")
            .join(item_counts, on="asin")
            .filter((pl.col("u_len") >= 5) & (pl.col("i_len") >= 5))
            .drop("u_len", "i_len")
        )
        if df.height == n:
            break

    df = df.sort(["user_id", "timestamp"])

    items = sorted(df["asin"].unique().to_list())
    item2id = {asin: i + 1 for i, asin in enumerate(items)}  # 1-based; repo remaps -1
    id2item = {v: k for k, v in item2id.items()}

    out_dir = raw_split_dir(root, split)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "sequential_data.txt"), "w") as f:
        for uid, (key, grp) in enumerate(
            df.group_by("user_id", maintain_order=True), start=1
        ):
            seq = [str(item2id[a]) for a in grp["asin"].to_list()]
            f.write(f"{uid} {' '.join(seq)}\n")

    with open(os.path.join(out_dir, "datamaps.json"), "w") as f:
        json.dump({"item2id": item2id}, f)

    print("[prepare_data] Writing meta.json.gz ...")
    wanted = set(items)
    with gzip.open(os.path.join(out_dir, "meta.json.gz"), "wt", encoding="utf-8") as out:
        with open(meta_path, "r", encoding="utf-8") as src:
            for line in src:
                rec = json.loads(line)
                asin = rec.get("parent_asin") or rec.get("asin")
                if asin not in wanted:
                    continue
                cats = rec.get("categories") or []
                flat = [c for sub in cats for c in (sub if isinstance(sub, list) else [sub])]
                meta = {
                    "asin": asin,
                    "title": rec.get("title") or "",
                    "brand": rec.get("store") or rec.get("brand") or "Unknown",
                    "categories": [flat if flat else ["Beauty"]],
                    "price": rec.get("price") or "unknown",
                }
                out.write(json.dumps(meta) + "\n")
                wanted.discard(asin)
        # Any items without meta still need a record (repo does fillna, but the
        # column must exist on every row).
        for asin in wanted:
            out.write(
                json.dumps(
                    {
                        "asin": asin,
                        "title": "",
                        "brand": "Unknown",
                        "categories": [["Beauty"]],
                        "price": "unknown",
                    }
                )
                + "\n"
            )
    print(f"[prepare_data] Fallback raw data written to {out_dir}")


def ensure_raw(root=DATASET_ROOT, split: str = SPLIT) -> str:
    if raw_files_present(root, split):
        print(f"[prepare_data] Raw files already present under {raw_split_dir(root, split)}")
        return raw_split_dir(root, split)

    try:
        print("[prepare_data] Attempting upstream auto-download of classic Beauty 5-core ...")
        try_repo_download(root, split)
    except Exception as e:  # gdrive quota, network, etc.
        print(f"[prepare_data] Upstream download failed ({type(e).__name__}: {e})")
        print("[prepare_data] Falling back to 2023 dump conversion.")
        build_raw_from_2023(root, split)

    assert raw_files_present(root, split), "raw data unavailable from both paths"
    return raw_split_dir(root, split)


def compute_popularity(root=DATASET_ROOT, split: str = SPLIT) -> pl.DataFrame:
    counts = Counter()
    with open(os.path.join(raw_split_dir(root, split), "sequential_data.txt")) as f:
        for line in f:
            toks = line.split()
            counts.update(int(t) - 1 for t in toks[1:])  # repo stores ids 1-based
    df = pl.DataFrame(
        {
            "item_id": sorted(counts.keys()),
            "n_interactions": [counts[i] for i in sorted(counts.keys())],
        }
    )
    return df


def export_item_corpus(root=DATASET_ROOT, split: str = SPLIT, force: bool = False):
    """Load (or build) the processed dataset and export embeddings + text + popularity."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and EMBEDDINGS_NPY.exists()
        and ITEMS_PARQUET.exists()
        and POPULARITY_PARQUET.exists()
    ):
        print("[prepare_data] Item corpus already exported; skipping.")
        return

    _patch_encoder_batch_size()
    ds = ItemData(
        root=str(root),
        dataset=RecDataset.AMAZON,
        split=split,
        train_test_split="all",
        force_process=force,
    )
    x = ds.item_data[:, :768].float().numpy()
    text = np.array([str(t) for t in ds.item_text])

    np.save(EMBEDDINGS_NPY, x)
    pl.DataFrame({"item_id": np.arange(x.shape[0]), "text": text}).write_parquet(
        ITEMS_PARQUET
    )
    compute_popularity(root, split).write_parquet(POPULARITY_PARQUET)
    print(
        f"[prepare_data] Exported {x.shape[0]} items, dim {x.shape[1]} -> {EMBEDDINGS_DIR}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--force-process", action="store_true")
    args = ap.parse_args()

    ensure_raw(args.root, args.split)
    export_item_corpus(args.root, args.split, force=args.force_process)


if __name__ == "__main__":
    main()
