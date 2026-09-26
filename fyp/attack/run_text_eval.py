"""Text-level evaluation — the first real-text results (interim report).

All prior numbers are embedding-space: the attacker moves the Sentence-T5
vector directly. This runner connects the audit to actual listing text.

Experiment A (benign-edit stability / paraphrase calibration):
  N random catalogue items x 5 meaning-preserving edit operators
  (fyp.attack.text_edits.BENIGN_OPS). Each edited text is re-encoded with the
  same sentence-t5-xxl encoder and re-quantized with the same RQ-VAE. We report
  per-operator cosine to the original embedding and semantic-ID flip rates,
  plus flip rate binned by cosine — this calibrates the attack's cos >= 0.95
  budget against real text changes (the audit shows 48.5% of items already
  have a *different* product within cos 0.95, so the budget is weak).

Experiment B (greedy text steering):
  For M near-boundary items, greedily append/prepend seller-vocabulary
  keywords (mined from popular items already on the high-traffic level-0
  prefix, plus PROMO_PHRASES) to steer the item onto that prefix — the text
  side of the white-box `traffic_prefix` goal. Reports text-level ASR, the
  cosine cost of successful rewrites, and example rewrites for the report.

Outputs:
  artifacts/runs/text_eval.json          headline numbers
  artifacts/runs/text_eval_examples.md   sample rewrites (for the report)

Usage:
  python -m fyp.attack.run_text_eval [--n-benign 500] [--n-attack 50]
                                     [--max-rounds 5] [--seed 0]
"""

import argparse
import json

import numpy as np
import polars as pl
import torch

from fyp.attack.text_edits import (
    BENIGN_OPS,
    PROMO_PHRASES,
    append_keywords,
    parse,
    prepend_keywords,
    title_words,
)
from fyp.common.generate_semantic_ids import (
    DEVICE,
    codebook_dist,
    load_rqvae,
    semantic_ids_and_margins,
)
from fyp.common.paths import (
    AUDIT_ITEMS_CSV,
    CHECKPOINT,
    EMBEDDINGS_NPY,
    ITEMS_PARQUET,
    RUNS_DIR,
)

EVAL_JSON = RUNS_DIR / "text_eval.json"
EXAMPLES_MD = RUNS_DIR / "text_eval_examples.md"

_COS_BINS = [0.0, 0.9, 0.95, 0.98, 0.99, 1.001]


def load_encoder():
    """The same encoder as prepare_data (sentence-t5-xxl, fp16 — see fyp/README)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/sentence-t5-xxl",
        model_kwargs={"dtype": "float16"},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def encode_texts(enc, texts: list[str], batch: int = 64) -> np.ndarray:
    return (
        enc.encode(texts, batch_size=batch, show_progress_bar=False,
                   convert_to_tensor=True)
        .float()
        .cpu()
        .numpy()
    )


def sem_ids(model, x: np.ndarray) -> np.ndarray:
    ids, _margins, _d1 = semantic_ids_and_margins(model, torch.from_numpy(x))
    return ids.cpu().numpy().T  # (B, L)


def level0_dist_to(model, x: torch.Tensor, code: int) -> torch.Tensor:
    with torch.no_grad():
        r = model.encode(x.to(DEVICE))
        return codebook_dist(model.layers[0], r)[:, code].cpu()


def run(n_benign=500, n_attack=50, max_rounds=5, seed=0, checkpoint=str(CHECKPOINT)):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    texts = pl.read_parquet(ITEMS_PARQUET)["text"].to_list()
    x_stored = np.load(EMBEDDINGS_NPY)
    audit = pl.read_csv(AUDIT_ITEMS_CSV)
    model = load_rqvae(checkpoint)
    enc = load_encoder()

    out = {"n_items_catalogue": len(texts), "seed": seed}

    # --- 0. Encoder consistency: fresh encodings vs the cached corpus -------
    idx0 = rng.choice(len(texts), size=min(200, len(texts)), replace=False)
    x_fresh = encode_texts(enc, [texts[i] for i in idx0])
    cos_check = np.sum(x_fresh * x_stored[idx0], axis=1) / (
        np.linalg.norm(x_fresh, axis=1) * np.linalg.norm(x_stored[idx0], axis=1)
    )
    ids_fresh = sem_ids(model, x_fresh)
    ids_stored = sem_ids(model, x_stored[idx0])
    out["encoder_check"] = {
        "n": int(len(idx0)),
        "median_cos_fresh_vs_stored": float(np.median(cos_check)),
        "full_id_agreement_fresh_vs_stored": float(
            (ids_fresh == ids_stored).all(axis=1).mean()
        ),
        "level0_agreement_fresh_vs_stored": float(
            (ids_fresh[:, 0] == ids_stored[:, 0]).mean()
        ),
    }
    print(f"[text_eval] encoder check: {out['encoder_check']}")

    # --- A. Benign-edit stability -------------------------------------------
    idxA = rng.choice(len(texts), size=min(n_benign, len(texts)), replace=False)
    op_names = list(BENIGN_OPS)
    edit_texts, edit_meta = [], []
    for i in idxA:
        p0 = parse(texts[i])
        if p0 is None:
            continue
        for name in op_names:
            p = parse(texts[i])
            edit_texts.append(BENIGN_OPS[name](p, rng))
            edit_meta.append((int(i), name))
    x_edit = encode_texts(enc, edit_texts)
    ids_edit = sem_ids(model, x_edit)
    ids_orig = sem_ids(model, x_stored[[m[0] for m in edit_meta]])
    x_orig = x_stored[[m[0] for m in edit_meta]]
    cos_edit = np.sum(x_edit * x_orig, axis=1) / (
        np.linalg.norm(x_edit, axis=1) * np.linalg.norm(x_orig, axis=1)
    )
    flip0 = ids_edit[:, 0] != ids_orig[:, 0]
    flip_full = (ids_edit != ids_orig).any(axis=1)

    per_op = {}
    for name in op_names:
        mask = np.array([m[1] == name for m in edit_meta])
        per_op[name] = {
            "n": int(mask.sum()),
            "mean_cos": float(cos_edit[mask].mean()),
            "p5_cos": float(np.percentile(cos_edit[mask], 5)),
            "level0_flip_rate": float(flip0[mask].mean()),
            "full_id_change_rate": float(flip_full[mask].mean()),
        }
    by_bin = {}
    for lo, hi in zip(_COS_BINS[:-1], _COS_BINS[1:]):
        m = (cos_edit >= lo) & (cos_edit < hi)
        if m.sum() >= 20:
            by_bin[f"[{lo},{hi})"] = {
                "n": int(m.sum()),
                "level0_flip_rate": float(flip0[m].mean()),
                "full_id_change_rate": float(flip_full[m].mean()),
            }
    out["benign_edit_stability"] = {
        "n_items": int(len(set(m[0] for m in edit_meta))),
        "n_edits": int(len(edit_meta)),
        "per_operator": per_op,
        "flip_rate_by_cosine_bin": by_bin,
        "overall": {
            "mean_cos": float(cos_edit.mean()),
            "level0_flip_rate": float(flip0.mean()),
            "full_id_change_rate": float(flip_full.mean()),
        },
    }
    print(f"[text_eval] benign edits: {out['benign_edit_stability']['overall']}")

    # --- B. Greedy text steering onto the high-traffic prefix ---------------
    code_cols = sorted(c for c in audit.columns if c.startswith("code_l"))
    pop = audit.drop_nulls(["n_interactions"]).sort("n_interactions", descending=True)
    top = pop.head(max(10, pop.height // 100))
    traffic_code = int(
        top.group_by("code_l0").len().sort(["len", "code_l0"], descending=[True, False])
        .row(0)[0]
    )
    target_texts = [texts[i] for i in top.filter(pl.col("code_l0") == traffic_code)["item_id"].to_list()]
    stop = set("the a an of for and with set kit pack pcs oz fl ml".split())
    vocab = title_words(target_texts, stop)[:12] + PROMO_PHRASES

    margins = audit.sort("margin_l0")
    cand_ids = [
        i for i in margins["item_id"].to_list()
        if int(margins.filter(pl.col("item_id") == i)["code_l0"][0]) != traffic_code
    ][:n_attack]

    results = []
    for i in cand_ids:
        p0 = parse(texts[i])
        if p0 is None:
            continue
        cur_text, cur_x = texts[i], x_stored[i]
        cur_d = float(level0_dist_to(model, torch.from_numpy(cur_x[None]), traffic_code)[0])
        applied = []
        success = False
        for _round in range(max_rounds):
            cands = []
            for kw in vocab:
                cands.append(append_keywords(parse(cur_text), [kw]))
                cands.append(prepend_keywords(parse(cur_text), [kw]))
            xc = torch.from_numpy(encode_texts(enc, cands))
            dc = level0_dist_to(model, xc, traffic_code)
            best = int(dc.argmin())
            if float(dc[best]) >= cur_d - 1e-9:
                break  # greedy stall
            cur_text, cur_x, cur_d = cands[best], xc[best].numpy(), float(dc[best])
            applied.append(("+" if best % 2 == 0 else "^") + vocab[best // 2])
            r0 = sem_ids(model, cur_x[None])[0, 0]
            if int(r0) == traffic_code:
                success = True
                break
        cos_final = float(
            np.dot(cur_x, x_stored[i])
            / (np.linalg.norm(cur_x) * np.linalg.norm(x_stored[i]))
        )
        results.append(
            {
                "item_id": int(i),
                "success": success,
                "rounds": len(applied),
                "keywords": applied,
                "cos_to_original": cos_final,
                "original": texts[i],
                "final": cur_text,
            }
        )

    ok = [r for r in results if r["success"]]
    ok95 = [r for r in ok if r["cos_to_original"] >= 0.95]
    out["text_steering"] = {
        "target_traffic_code": traffic_code,
        "vocab": vocab,
        "n_items": len(results),
        "max_rounds": max_rounds,
        "asr": len(ok) / max(1, len(results)),
        "asr_cos_ge_0.95": len(ok95) / max(1, len(results)),
        "mean_cos_success": float(np.mean([r["cos_to_original"] for r in ok])) if ok else None,
        "min_cos_success": float(np.min([r["cos_to_original"] for r in ok])) if ok else None,
        "mean_rounds_success": float(np.mean([r["rounds"] for r in ok])) if ok else None,
    }
    print(f"[text_eval] steering: {out['text_steering']}")

    EVAL_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    ex_lines = ["# Text steering examples (greedy keyword edits)", ""]
    for r in sorted(ok, key=lambda r: -r["cos_to_original"])[:10]:
        ex_lines += [
            f"## item {r['item_id']} — cos {r['cos_to_original']:.3f}, {r['rounds']} rounds",
            f"- original: {parse(r['original']).title if parse(r['original']) else r['original']}",
            f"- rewrite:  {parse(r['final']).title if parse(r['final']) else r['final']}",
            "",
        ]
    EXAMPLES_MD.write_text("\n".join(ex_lines), encoding="utf-8")
    print(f"[text_eval] wrote {EVAL_JSON} and {EXAMPLES_MD}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-benign", type=int, default=500)
    ap.add_argument("--n-attack", type=int, default=50)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()
    run(args.n_benign, args.n_attack, args.max_rounds, args.seed, args.checkpoint)


if __name__ == "__main__":
    main()
