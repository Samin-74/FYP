"""Catalogue-wide manipulability audit runner (plan section 3.5).

Reads cached embeddings + the trained RQ-VAE, computes per-item boundary
margins (all items) and flip sensitivity (batched gradient descent on the margin on a
sampled subset), joins popularity / description length / category, and exports:

  - artifacts/audit/audit_items.csv       per-item margins, sensitivity, steerability
  - artifacts/audit/audit_summary.json    distributions, eps table, regression, collisions

Usage:
  python -m fyp.audit.run_audit [--limit N] [--sensitivity-sample M]
                                [--eps 0.25 0.5 ...] [--level 0]
"""

import argparse
import json
import math
import re

import numpy as np
import polars as pl
import torch

from fyp.audit.metrics import (
    bisect_flip_distance,
    effective_budget,
    min_flip_perturbation_batch,
    steerability,
)
from fyp.common.generate_semantic_ids import DEVICE, load_rqvae, run as run_semantic_ids
from fyp.common.paths import (
    AUDIT_DIR,
    AUDIT_ITEMS_CSV,
    AUDIT_SUMMARY_JSON,
    CHECKPOINT,
    EMBEDDINGS_NPY,
    ITEMS_PARQUET,
    POPULARITY_PARQUET,
    SEMANTIC_IDS_PARQUET,
)

# Sensitivity search parameters; reachable ||delta|| is min(NORM_CAP, MAX_STEPS * STEP_SIZE).
MAX_STEPS = 200
STEP_SIZE = 0.05
NORM_CAP = 50.0

_CATEGORY_RE = re.compile(r"Categories: \[(.*?)\]")


def parse_category(text: str) -> str:
    """Second-level category (element 0 is always the top-level 'Beauty')."""
    m = _CATEGORY_RE.search(text)
    if not m:
        return "Unknown"
    cats = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
    return cats[1] if len(cats) > 1 else (cats[0] if cats else "Unknown")


def percentile(name: str, values: np.ndarray) -> dict:
    qs = [1, 5, 25, 50, 75, 95, 99]
    ps = np.percentile(values, qs)
    return {f"{name}_p{q}": float(v) for q, v in zip(qs, ps)}


def popularity_regression(df: pl.DataFrame, level: int) -> dict:
    """OLS of margin against log-popularity; returns slope/intercept/R^2."""
    d = df.drop_nulls(["n_interactions", f"margin_l{level}"])
    if d.height < 10:
        return {"note": "insufficient data"}
    x = np.log1p(d["n_interactions"].to_numpy())
    y = d[f"margin_l{level}"].to_numpy()
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r_squared": r2,
        "n": int(d.height),
    }


def collision_stats(df: pl.DataFrame, n_layers: int) -> dict:
    """Two collision measures — report both, they answer different questions.

    - ``collision_rate`` = 1 - unique/items: share of items that need a
      dedup token > 0 (the "excess" items beyond one per ID).
    - ``frac_items_in_colliding_groups``: share of items whose 3-level ID is
      shared with at least one other item.
    """
    cols = [f"code_l{k}" for k in range(n_layers)]
    ids = df.select(cols)
    n_items = ids.height
    n_unique = ids.unique().height
    counts = ids.group_by(cols).len().sort("len", descending=True)
    sizes = counts["len"].to_numpy()
    colliding = sizes[sizes > 1]
    return {
        "n_items": n_items,
        "n_unique_ids": n_unique,
        "collision_rate": 1 - n_unique / n_items,
        "n_colliding_groups": int(len(colliding)),
        "frac_items_in_colliding_groups": float(colliding.sum() / n_items),
        "largest_collision_group": int(sizes[0]) if len(sizes) else 0,
    }


def embedding_geometry(x: np.ndarray, chunk: int = 2048) -> dict:
    """Calibration for the embedding-space constraints.

    - Norm percentiles: Sentence-T5 should emit unit-norm vectors; if so, any
      attack must stay on that sphere (see fyp.attack.constraints).
    - Nearest-*other*-item cosine: if distinct catalogue products already sit
      above the attack's cosine threshold, that threshold does not guarantee
      semantic preservation.
    """
    xt = torch.from_numpy(x).to(DEVICE).float()
    norms = xt.norm(dim=1)
    xn = xt / torch.clamp(norms[:, None], min=1e-12)
    nn_cos = torch.empty(xt.shape[0], device=DEVICE)
    for i in range(0, xt.shape[0], chunk):
        sims = xn[i : i + chunk] @ xn.T
        rows = torch.arange(sims.shape[0], device=DEVICE)
        sims[rows, rows + i] = -2.0  # exclude self
        nn_cos[i : i + chunk] = sims.max(dim=1).values
    nn_cos = nn_cos.cpu().numpy()
    return {
        **percentile("embedding_norm", norms.cpu().numpy()),
        **percentile("nn_cosine", nn_cos),
        "frac_nn_cosine_ge": {
            str(t): float((nn_cos >= t).mean()) for t in (0.8, 0.9, 0.95, 0.99)
        },
    }


def run(
    checkpoint: str = str(CHECKPOINT),
    limit: int | None = None,
    sensitivity_sample: int = 2048,
    eps_multipliers=(0.25, 0.5, 1.0, 2.0, 5.0),
    sensitivity_level: int = 0,
    bisection: bool = True,
    bisect_iters: int = 25,
):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # --- semantic IDs + margins (cached if already generated for this corpus)
    # (regenerated if the cached IDs came from a different checkpoint)
    meta_path = SEMANTIC_IDS_PARQUET.parent / "meta.json"
    cached_ckpt = (
        json.loads(meta_path.read_text()).get("checkpoint")
        if meta_path.exists() else None
    )
    if SEMANTIC_IDS_PARQUET.exists() and cached_ckpt == str(checkpoint):
        sem = pl.read_parquet(SEMANTIC_IDS_PARQUET)
    else:
        if SEMANTIC_IDS_PARQUET.exists():
            print(
                f"[audit] cached semantic IDs are from {cached_ckpt}, not"
                f" {checkpoint}; regenerating"
            )
        sem = run_semantic_ids(checkpoint)
    if limit:
        sem = sem.head(limit)

    x = np.load(EMBEDDINGS_NPY)[: sem.height]
    model = load_rqvae(checkpoint)

    # --- sensitivity on a sample (batched gradient descent on the margin)
    n_layers = len([c for c in sem.columns if c.startswith("code_l")])
    idx = np.random.default_rng(42).choice(
        sem.height, size=min(sensitivity_sample, sem.height), replace=False
    )
    X_sample = torch.from_numpy(x[idx])
    print(f"[audit] computing sensitivity on {X_sample.shape[0]} items (batched)...")
    res = min_flip_perturbation_batch(
        model, X_sample, level=sensitivity_level,
        max_steps=MAX_STEPS, step_size=STEP_SIZE, norm_cap=NORM_CAP,
        record_pre=bisection,
    )
    budget = effective_budget(MAX_STEPS, STEP_SIZE, NORM_CAP)

    bdist = None
    if bisection:
        flipped_t = res["flipped"]
        print(f"[audit] bisecting flip segments on {int(flipped_t.sum())} flipped items...")
        bdist = bisect_flip_distance(
            model,
            X_sample[flipped_t.cpu()],
            res["x_pre"][flipped_t],
            res["x"][flipped_t],
            res["original_id"][flipped_t],
            level=sensitivity_level,
            n_bisect=bisect_iters,
        ).numpy()

    sens = np.full(sem.height, np.nan, dtype=np.float64)
    flipped = np.zeros(sem.height, dtype=bool)
    flipped[idx] = res["flipped"].cpu().numpy()
    sens[idx] = np.where(
        flipped[idx], res["perturbation_norm"].cpu().numpy(), math.inf
    )

    flipped_sens = sens[flipped]
    tau = float(np.median(flipped_sens)) if flipped.any() else 1.0
    steer = steerability(
        torch.from_numpy(np.nan_to_num(sens, nan=math.inf, posinf=math.inf)), tau
    )
    steer_np = steer.numpy()
    steer_np[~flipped] = 0.0

    items = sem.with_columns(
        pl.Series(f"sensitivity_l{sensitivity_level}", sens),
        pl.Series(f"flipped_l{sensitivity_level}", flipped),
        pl.Series(f"steerability_l{sensitivity_level}", steer_np),
    )
    if bdist is not None:
        sens_bis = np.full(sem.height, np.nan, dtype=np.float64)
        sens_bis[idx[res["flipped"].cpu().numpy()]] = bdist
        items = items.with_columns(
            pl.Series(f"sensitivity_bisect_l{sensitivity_level}", sens_bis)
        )

    # --- item properties: description length + category
    texts = pl.read_parquet(ITEMS_PARQUET)
    items = items.join(
        texts.with_columns(
            pl.col("text").str.len_chars().alias("desc_len"),
            pl.col("text").map_elements(parse_category, return_dtype=str).alias("category"),
        ).select(["item_id", "desc_len", "category"]),
        on="item_id",
        how="left",
    )

    if POPULARITY_PARQUET.exists():
        pop = pl.read_parquet(POPULARITY_PARQUET)
        items = items.join(pop, on="item_id", how="left")
    else:
        items = items.with_columns(pl.lit(None).cast(pl.Int64).alias("n_interactions"))

    items.write_csv(AUDIT_ITEMS_CSV)

    # --- summary (rates are over the sampled items only, not the catalogue)
    sample_flipped = flipped[idx]
    summary = {
        "n_items": items.height,
        "sensitivity": {
            "level": sensitivity_level,
            "sample_size": int(len(idx)),
            "flip_rate": float(sample_flipped.mean()),
            "budget_exceeded_rate": float((~sample_flipped).mean()),
            "tau_median": tau,
            "norm_cap": NORM_CAP,
            "max_steps": MAX_STEPS,
            "step_size": STEP_SIZE,
            "effective_budget": budget,
            "sensitivity_percentiles": percentile(
                "sensitivity", flipped_sens[np.isfinite(flipped_sens)]
            ) if flipped.any() else {},
        },
        "collisions": collision_stats(items, n_layers),
        "embedding_geometry": embedding_geometry(x),
    }
    for k in range(n_layers):
        m = items[f"margin_l{k}"].to_numpy()
        d1 = items[f"d1_l{k}"].to_numpy()
        med = float(np.median(m))
        summary[f"level_{k}"] = {
            **percentile("margin", m),
            "margin_median": med,
            # "within eps of a boundary", eps in units of the catalogue median margin
            "frac_within_eps_x_median": {
                str(e): float((m < e * med).mean()) for e in eps_multipliers
            },
            **percentile("margin_over_d1", m / np.maximum(d1, 1e-12)),
            "popularity_regression": popularity_regression(items, k),
        }

    if bdist is not None:
        summary["sensitivity"]["bisection"] = {
            "n_bisect_iters": bisect_iters,
            "n_flipped": int(bdist.size),
            "mean": float(bdist.mean()),
            "frac_le_0.005": float((bdist <= 0.005).mean()),
            "frac_le_0.01": float((bdist <= 0.01).mean()),
            **percentile("flip_distance", bdist),
            "note": (
                f"measured flip distance: {bisect_iters}-iteration bisection on the"
                " final search segment; still an upper bound on the true minimum"
                " (heuristic descent direction, distance along the search path)"
            ),
        }

    AUDIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[audit] wrote {AUDIT_ITEMS_CSV}")
    print(f"[audit] wrote {AUDIT_SUMMARY_JSON}")
    print(
        f"[audit] level-{sensitivity_level} flip rate on sample: "
        f"{summary['sensitivity']['flip_rate']:.1%} (tau={tau:.3f})"
    )
    return items, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sensitivity-sample", type=int, default=2048)
    ap.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 5.0],
        help="report %% of items with margin < eps * median_margin",
    )
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument(
        "--no-bisection",
        dest="bisection",
        action="store_false",
        help="skip the bisection refinement of flip distances",
    )
    ap.add_argument("--bisect-iters", type=int, default=25)
    args = ap.parse_args()
    run(
        args.checkpoint,
        args.limit,
        args.sensitivity_sample,
        tuple(args.eps),
        args.level,
        args.bisection,
        args.bisect_iters,
    )


if __name__ == "__main__":
    main()
