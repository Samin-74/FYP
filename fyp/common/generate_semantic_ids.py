"""Load the trained RQ-VAE and compute semantic IDs + boundary margins.

Outputs (consumed by fyp.audit and fyp.attack):
  - artifacts/semantic_ids/semantic_ids.parquet  item_id, code_l0..l{L-1}, margin_l0..l{L-1}
  - artifacts/semantic_ids/codebooks.npy         list of (codebook_size, embed_dim) arrays

Margin definition (plan section 3.5): for item x and quantization level k with
quantizer input residual r_k, let d1 < d2 be the two smallest squared-L2
distances from r_k to the level-k codebook. Boundary margin = d2 - d1. Small
margins mean the item sits close to a Voronoi cell boundary.

Usage:
  python -m fyp.common.generate_semantic_ids [--checkpoint PATH] [--limit N]
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from modules.rqvae import RqVae

from fyp.common.paths import (
    CHECKPOINT,
    CODEBOOKS_NPY,
    EMBEDDINGS_NPY,
    SEMANTIC_IDS_DIR,
    SEMANTIC_IDS_PARQUET,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 2048


def load_rqvae(checkpoint: str = str(CHECKPOINT)) -> RqVae:
    """Rebuild RqVae from an upstream checkpoint (state dict + constructor config)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = {
        k: v
        for k, v in state["model_config"].items()
        if k != "self" and k in inspect.signature(RqVae.__init__).parameters
    }
    model = RqVae(**cfg)
    model.load_state_dict(state["model"])
    model.to(DEVICE).eval()
    return model


def codebook(level_module) -> torch.Tensor:
    """Effective level codebook (matches modules.quantize.Quantize.forward)."""
    return level_module.out_proj(level_module.embedding.weight)


def codebook_dist(layer, r: torch.Tensor) -> torch.Tensor:
    """Squared-L2 distances from residual r to the level codebook (B, n_embed)."""
    cb = codebook(layer)
    return (r**2).sum(axis=1, keepdim=True) + (cb.T**2).sum(
        axis=0, keepdim=True
    ) - 2 * r @ cb.T


@torch.no_grad()
def semantic_ids_and_margins(model: RqVae, x: torch.Tensor, batch: int = BATCH):
    """Compute semantic IDs and per-level boundary margins for a batch of embeddings.

    Returns (sem_ids, margins, d1s): (L, B) tensors — assigned code, boundary
    margin (d2 - d1), and nearest-codeword distance d1, all in the upstream
    squared-L2 residual space.
    """
    sem_ids, margins, d1s = [], [], []
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch].to(DEVICE)
        r = model.encode(xb)
        ids_b, marg_b, d1_b = [], [], []
        for layer in model.layers:
            d = codebook_dist(layer, r)
            top2 = torch.topk(d, k=2, dim=1, largest=False)
            ids_b.append(top2.indices[:, 0])
            marg_b.append(top2.values[:, 1] - top2.values[:, 0])
            d1_b.append(top2.values[:, 0])
            r = r - layer.get_item_embeddings(top2.indices[:, 0])
        sem_ids.append(torch.stack(ids_b))
        margins.append(torch.stack(marg_b))
        d1s.append(torch.stack(d1_b))
    return (
        torch.cat(sem_ids, dim=1),
        torch.cat(margins, dim=1),
        torch.cat(d1s, dim=1),
    )


def run(checkpoint: str = str(CHECKPOINT), limit: int | None = None) -> pl.DataFrame:
    x = np.load(EMBEDDINGS_NPY)
    if limit:
        x = x[:limit]
    model = load_rqvae(checkpoint)

    sem_ids, margins, d1s = semantic_ids_and_margins(model, torch.from_numpy(x))
    sem_ids, margins, d1s = (
        sem_ids.cpu().numpy(),
        margins.cpu().numpy(),
        d1s.cpu().numpy(),
    )

    L = sem_ids.shape[0]
    df = pl.DataFrame({"item_id": np.arange(x.shape[0])})
    for k in range(L):
        df = df.with_columns(
            pl.Series(f"code_l{k}", sem_ids[k]),
            pl.Series(f"margin_l{k}", margins[k]),
            pl.Series(f"d1_l{k}", d1s[k]),
        )

    SEMANTIC_IDS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(SEMANTIC_IDS_PARQUET)
    np.save(
        CODEBOOKS_NPY,
        np.stack(
            [codebook(l).detach().cpu().numpy() for l in model.layers]
        ),
    )
    meta = {
        "checkpoint": checkpoint,
        "n_items": int(x.shape[0]),
        "n_layers": L,
        "codebook_size": int(model.codebook_size),
        "margin": "d2 - d1, squared L2 in residual space (upstream assignment rule)",
    }
    (SEMANTIC_IDS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[semantic_ids] wrote {SEMANTIC_IDS_PARQUET} ({x.shape[0]} items, {L} levels)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.checkpoint, args.limit)


if __name__ == "__main__":
    main()
