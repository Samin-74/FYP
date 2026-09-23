"""End-to-end pipeline verification on a mini-batch (spike-quality).

Flow: data -> embeddings -> RQ-VAE quantization -> audit metrics -> attack
perturbation tests. Prints a verification summary; exits non-zero on failure.

Run from the FYP root:
  python -m fyp.spike.test_pipeline [--n-items 64] [--n-attack 8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fyp.attack.black_box import query_only_flip
from fyp.attack.constraints import EmbeddingCosineConstraint
from fyp.attack.white_box import targeted_flip
from fyp.audit.metrics import min_flip_perturbation
from fyp.common import prepare_data
from fyp.common.generate_semantic_ids import (
    DEVICE,
    load_rqvae,
    semantic_ids_and_margins,
)
from fyp.common.paths import CHECKPOINT, EMBEDDINGS_NPY, POPULARITY_PARQUET

CHECKS = []


def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--n-attack", type=int, default=8)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()

    print(f"== device: {DEVICE} ==")

    print("\n[1/5] Data + embeddings")
    prepare_data.ensure_raw()
    prepare_data.export_item_corpus()
    x = np.load(EMBEDDINGS_NPY)[: args.n_items]
    check("embeddings loaded", x.ndim == 2 and x.shape[1] == 768, f"shape={x.shape}")
    check("popularity table exists", POPULARITY_PARQUET.exists())

    print("\n[2/5] Quantization (trained RQ-VAE)")
    model = load_rqvae(args.checkpoint)
    n_layers = len(model.layers)
    sem_ids, margins, _d1s = semantic_ids_and_margins(
        model, torch.from_numpy(x)
    )
    sem_ids, margins = sem_ids.cpu(), margins.cpu()
    check("semantic IDs shape", sem_ids.shape == (n_layers, x.shape[0]),
          f"{tuple(sem_ids.shape)}")
    check("ID range", bool((sem_ids >= 0).all() and (sem_ids < model.codebook_size).all()))
    check("margins non-negative", bool((margins >= 0).all()),
          f"median={float(margins.median()):.3f}")

    print("\n[3/5] Audit: sensitivity (gradient flip)")
    idx = np.random.default_rng(0).choice(x.shape[0], size=min(4, x.shape[0]), replace=False)
    flip_results = []
    for i in idx:
        r = min_flip_perturbation(model, torch.from_numpy(x[i:i+1]).to(DEVICE), level=0)
        flip_results.append(r)
    n_flipped = sum(r["flipped"] for r in flip_results)
    check("sensitivity ran on sample", len(flip_results) == len(idx))
    print(f"       flip rate on {len(idx)} items: {n_flipped}/{len(idx)} "
          f"(norms: {[f'{r['perturbation_norm']:.3f}' for r in flip_results]})")

    print("\n[4/5] White-box targeted flip (embedding space)")
    constraint = EmbeddingCosineConstraint(threshold=0.95)
    wb = []
    for i in range(min(args.n_attack, x.shape[0])):
        x0 = torch.from_numpy(x[i:i+1]).to(DEVICE)
        with torch.no_grad():
            cur = int(sem_ids[0, i])
        target = (cur + 1) % model.codebook_size
        r = targeted_flip(model, x0, target_code=target, level=0, constraint=constraint)
        wb.append(r)
    wb_ok = sum(r["success"] for r in wb)
    check("white-box attack ran", len(wb) == min(args.n_attack, x.shape[0]))
    print(f"       targeted flips: {wb_ok}/{len(wb)}, "
          f"median cosine={np.median([r['cosine_to_original'] for r in wb]):.4f}, "
          f"median ||delta||={np.median([r['perturbation_norm'] for r in wb]):.3f}")

    print("\n[5/5] Black-box query-only flip (stub)")
    bb = []
    for i in range(min(args.n_attack, x.shape[0])):
        x0 = torch.from_numpy(x[i:i+1]).to(DEVICE)
        with torch.no_grad():
            cur = int(sem_ids[0, i])
        target = (cur + 1) % model.codebook_size
        r = query_only_flip(
            model, x0, target_code=target, level=0, constraint=constraint,
            budget=100, sigma=float(margins[:, i].median().sqrt().clamp(min=0.05)),
        )
        bb.append(r)
    bb_ok = sum(r["success"] for r in bb)
    check("black-box attack ran", len(bb) == min(args.n_attack, x.shape[0]))
    print(f"       query-only flips: {bb_ok}/{len(bb)} "
          f"(queries used: {[r['queries'] for r in bb]})")

    print("\n== Verification summary ==")
    failed = [n for n, ok in CHECKS if not ok]
    for name, ok in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
