"""White-box attack evaluation on audit-flagged items (interim report, plan section 3.6).

Evaluates two attack goals on two item sets:

  Sets
    - near-boundary: items the audit flags (smallest level-0 margins)
    - random:        same-size random comparison set
  Goals
    - (a) any-flip:         batched gradient flip (min ||delta|| s.t. code changes)
    - (b) bestseller collision: targeted_flip onto the level-0 code of a
           randomly chosen top-1%% popular item
    - (c) high-traffic prefix: targeted_flip onto the most frequent level-0
           code among top-1%% popular items

The semantic-preservation constraint cos(original, perturbed) >= 0.95 is
enforced by projection inside the optimiser.

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

from fyp.attack.white_box import targeted_flip
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


def _any_flip_report(model, X, level):
    res = min_flip_perturbation_batch(model, X, level=level)
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


def _targeted_report(model, X, codes, level, cosine_threshold, max_steps=300):
    rows = []
    for i in range(X.shape[0]):
        r = targeted_flip(
            model,
            X[i : i + 1].to(DEVICE).float(),
            target_code=int(codes[i]),
            level=level,
            cosine_threshold=cosine_threshold,
            max_steps=max_steps,
        )
        rows.append(
            {
                "success": r["success"],
                "norm": r["perturbation_norm"],
                "cosine": r["cosine_to_original"],
                "steps": r["steps"],
                "target_code": int(codes[i]),
            }
        )
    return rows


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

    # --- target codes for goals (b) and (c): from top-1% popular items
    pop = audit.drop_nulls(["n_interactions"]).sort("n_interactions", descending=True)
    top = pop.head(max(10, pop.height // 100))
    bestsellers = top.sort("n_interactions", descending=True)
    traffic_code = (
        top.group_by(f"code_l{level}").len().sort("len", descending=True)[f"code_l{level}"][0]
    )

    eval_out = {"cosine_threshold": cosine_threshold, "level": level, "sets": {}}
    rows_all = []
    for name, ids in sets.items():
        X = torch.from_numpy(x_all[ids])
        any_flip = _any_flip_report(model, X, level)

        bs_codes = rng.choice(
            bestsellers[f"code_l{level}"].to_numpy(), size=len(ids), replace=True
        )
        coll = _targeted_report(model, X, bs_codes, level, cosine_threshold)
        prefix_codes = np.where(
            audit[f"code_l{level}"].to_numpy()[ids] == int(traffic_code),
            (int(traffic_code) + 1) % 256,  # already on prefix -> steer to next code
            int(traffic_code),
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
            "bestseller_collision": agg(coll),
            "traffic_prefix": agg(pref),
        }
        for goal, rows in [("any_flip", None), ("bestseller_collision", coll), ("traffic_prefix", pref)]:
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
