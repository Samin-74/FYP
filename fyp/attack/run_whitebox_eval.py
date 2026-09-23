"""White-box attack evaluation on audit-flagged items (interim report, plan section 3.6).

Evaluates four attack goals on two item sets:

  Sets
    - near-boundary: items the audit flags (smallest level-0 margins)
    - random:        same-size random comparison set
  Goals
    - (a) any-flip:             batched gradient flip (level-`level` code changes)
    - (b) bestseller prefix:    targeted_flip onto the level-`level` code of a
                                randomly chosen top-1%% popular item
    - (c) bestseller collision: targeted_collision onto the *full* 3-level
                                semantic ID of that same bestseller
    - (d) high-traffic prefix:  targeted_flip onto the most frequent level-0
                                code among top-1%% popular items (items already
                                on it are steered to the second most frequent)

For every goal, cos(original, perturbed) >= threshold and ||perturbed|| =
||original|| are enforced by projection inside the optimiser, so reported
successes satisfy both.

Outputs:
  artifacts/runs/whitebox_eval.json     headline numbers
  artifacts/runs/whitebox_eval.csv      per-item results

Usage:
  python -m fyp.attack.run_whitebox_eval [--n-per-set 200] [--cosine 0.95]
                                         [--level 0] [--seed 0]
"""

import argparse
import json

import numpy as np
import polars as pl
import torch

from fyp.attack.white_box import targeted_collision, targeted_flip
from fyp.audit.metrics import min_flip_perturbation_batch
from fyp.common.generate_semantic_ids import DEVICE, load_rqvae
from fyp.common.paths import (
    AUDIT_ITEMS_CSV,
    CHECKPOINT,
    EMBEDDINGS_NPY,
    RUNS_DIR,
    SEMANTIC_IDS_PARQUET,
)

EVAL_JSON = RUNS_DIR / "whitebox_eval.json"
EVAL_CSV = RUNS_DIR / "whitebox_eval.csv"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _any_flip_report(model, X, level, cosine_threshold):
    res = min_flip_perturbation_batch(
        model, X, level=level, cosine_threshold=cosine_threshold
    )
    flipped = res["flipped"]
    x0 = X.to(DEVICE).float()
    cos = _cosine(x0, res["x"])
    return {
        "n": int(X.shape[0]),
        "asr": float(flipped.float().mean()),
        "mean_norm_success": float(res["perturbation_norm"][flipped].mean()) if flipped.any() else None,
        "median_norm_success": float(res["perturbation_norm"][flipped].median()) if flipped.any() else None,
        "mean_cosine_success": float(cos[flipped].mean()) if flipped.any() else None,
        "norms": res["perturbation_norm"].cpu().numpy(),
        "cosines": cos.cpu().numpy(),
        "flipped": flipped.cpu().numpy(),
    }


def _row(r):
    return {
        "success": r["success"],
        "norm": r["perturbation_norm"],
        "cosine": r["cosine_to_original"],
        "steps": r["steps"],
    }


def _targeted_report(model, X, codes, level, cosine_threshold, max_steps=300):
    return [
        _row(
            targeted_flip(
                model,
                X[i : i + 1].to(DEVICE).float(),
                target_code=int(codes[i]),
                level=level,
                cosine_threshold=cosine_threshold,
                max_steps=max_steps,
            )
        )
        for i in range(X.shape[0])
    ]


def _collision_report(model, X, target_ids, cosine_threshold, max_steps=300):
    return [
        _row(
            targeted_collision(
                model,
                X[i : i + 1].to(DEVICE).float(),
                target_ids=target_ids[i],
                cosine_threshold=cosine_threshold,
                max_steps=max_steps,
            )
        )
        for i in range(X.shape[0])
    ]


def run(n_per_set=200, cosine_threshold=0.95, level=0, seed=0, checkpoint=str(CHECKPOINT)):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    audit = pl.read_csv(AUDIT_ITEMS_CSV)
    sem = pl.read_parquet(SEMANTIC_IDS_PARQUET)
    x_all = np.load(EMBEDDINGS_NPY)
    model = load_rqvae(checkpoint)

    med_margin = float(np.median(audit[f"margin_l{level}"].to_numpy()))
    near = audit.sort(f"margin_l{level}").head(n_per_set)["item_id"].to_numpy()
    pool = np.setdiff1d(audit["item_id"].to_numpy(), near)
    rand = rng.choice(pool, size=min(n_per_set, len(pool)), replace=False)
    sets = {"near_boundary": near, "random": rand}

    # --- targets for goals (b)-(d): from top-1% popular items
    pop = audit.drop_nulls(["n_interactions"]).sort("n_interactions", descending=True)
    top = pop.head(max(10, pop.height // 100))
    code_cols = sorted(c for c in audit.columns if c.startswith("code_l"))
    bestseller_ids = top.select(code_cols).to_numpy()  # (n_top, L) full semantic IDs
    traffic_rank = top.group_by(f"code_l{level}").len().sort(
        ["len", f"code_l{level}"], descending=[True, False]
    )
    traffic_code, traffic_code_2nd = (int(c) for c in traffic_rank[f"code_l{level}"][:2])

    eval_out = {
        "cosine_threshold": cosine_threshold,
        "norm_preserving": True,
        "level": level,
        "traffic_code": traffic_code,
        "sets": {},
    }
    rows_all = []
    for name, ids in sets.items():
        X = torch.from_numpy(x_all[ids])
        any_flip = _any_flip_report(model, X, level, cosine_threshold)

        # (b)/(c): one randomly drawn bestseller per item. Items that already
        # share the target's code/ID are redrawn so every attack needs a change.
        own_ids = audit.select(code_cols).to_numpy()[ids]
        picks = []
        for i in range(len(ids)):
            cand = bestseller_ids[bestseller_ids[:, level] != own_ids[i, level]]
            picks.append(cand[rng.integers(len(cand))])
        picks = np.stack(picks)
        bs_prefix = _targeted_report(model, X, picks[:, level], level, cosine_threshold)
        coll = _collision_report(model, X, picks, cosine_threshold)

        prefix_codes = np.where(
            own_ids[:, level] == traffic_code, traffic_code_2nd, traffic_code
        )
        pref = _targeted_report(model, X, prefix_codes, level, cosine_threshold)

        def agg(rows):
            ok = [r for r in rows if r["success"]]
            return {
                "asr": len(ok) / len(rows),
                "mean_cosine_success": float(np.mean([r["cosine"] for r in ok])) if ok else None,
                "mean_norm_success": float(np.mean([r["norm"] for r in ok])) if ok else None,
                "mean_steps_success": float(np.mean([r["steps"] for r in ok])) if ok else None,
            }

        eval_out["sets"][name] = {
            "n": len(ids),
            "median_margin_x_level_median": float(
                np.median(audit[f"margin_l{level}"].to_numpy()[ids]) / med_margin
            ),
            "any_flip": {k: v for k, v in any_flip.items() if k not in ("norms", "cosines", "flipped")},
            "bestseller_prefix": agg(bs_prefix),
            "bestseller_collision": agg(coll),
            "traffic_prefix": agg(pref),
        }
        goals = [
            ("any_flip", None),
            ("bestseller_prefix", bs_prefix),
            ("bestseller_collision", coll),
            ("traffic_prefix", pref),
        ]
        for goal, rows in goals:
            if rows is None:
                for i in range(len(ids)):
                    rows_all.append(
                        {
                            "set": name, "goal": "any_flip", "item_id": int(ids[i]),
                            "success": bool(any_flip["flipped"][i]),
                            "norm": float(any_flip["norms"][i]),
                            "cosine": float(any_flip["cosines"][i]),
                        }
                    )
            else:
                for i, r in enumerate(rows):
                    rows_all.append(
                        {
                            "set": name, "goal": goal, "item_id": int(ids[i]),
                            "success": r["success"], "norm": r["norm"],
                            "cosine": r["cosine"], "steps": r["steps"],
                        }
                    )
        print(f"[whitebox] set={name}: " + json.dumps(eval_out["sets"][name], default=str))

    pl.DataFrame(rows_all).write_csv(EVAL_CSV)
    EVAL_JSON.write_text(json.dumps(eval_out, indent=2))
    print(f"[whitebox] wrote {EVAL_JSON}")
    print(f"[whitebox] wrote {EVAL_CSV}")
    return eval_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-set", type=int, default=200)
    ap.add_argument("--cosine", type=float, default=0.95)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()
    run(args.n_per_set, args.cosine, args.level, args.seed, args.checkpoint)


if __name__ == "__main__":
    main()
