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

Experiment B-control (random-keyword steering):
  The SAME items, rounds, candidate budget and success criterion, but the 12
  mined keywords are replaced by 12 words drawn uniformly at random from
  catalogue-wide title vocabulary (targeted words and promo-phrase words
  excluded; the PROMO_PHRASES tail is kept identical so the only difference
  between the arms is targeted-vs-random keyword choice). Without this arm,
  the share of the targeted ASR due to *targeting* rather than general ID
  churn is unknown (docs/interim_checkpoint.md, limitation 10).

Outputs:
  artifacts/runs/text_eval.json          headline numbers
  artifacts/runs/text_eval_examples.md   sample rewrites (for the report)

Usage:
  python -m fyp.attack.run_text_eval [--n-benign 500] [--n-attack 50]
                                     [--max-rounds 5] [--seed 0]
                                     [--vocab-mode targeted|both]
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


def _greedy_steering(model, enc, vocab, cand_ids, traffic_code, texts, x_stored,
                     max_rounds):
    """Greedy append/prepend keyword steering; one implementation for all arms.

    Round-synchronized over items: each round, every still-active item's
    append+prepend candidates for all keywords are encoded in ONE batched call
    (34-text per-item batches underfill the encoder batch and are ~3x slower).
    Items are independent, so per-item choices are identical to a sequential
    per-item loop; an item deactivates on success or greedy stall. Returns the
    per-item records and an aggregate summary.
    """
    n_cand = 2 * len(vocab)
    states = {}
    for i in cand_ids:
        if parse(texts[i]) is None:
            continue
        states[i] = {
            "cur_text": texts[i],
            "cur_x": x_stored[i],
            "cur_d": float(
                level0_dist_to(model, torch.from_numpy(x_stored[i][None]), traffic_code)[0]
            ),
            "applied": [],
            "success": False,
            "active": True,
        }

    for _round in range(max_rounds):
        active = [i for i in states if states[i]["active"]]
        if not active:
            break
        flat_cands, owner = [], []
        for i in active:
            for kw in vocab:
                flat_cands.append(append_keywords(parse(states[i]["cur_text"]), [kw]))
                owner.append("+")
                flat_cands.append(prepend_keywords(parse(states[i]["cur_text"]), [kw]))
                owner.append("^")
        xc = encode_texts(enc, flat_cands)
        dc = level0_dist_to(model, torch.from_numpy(xc), traffic_code).numpy()
        for pos, i in enumerate(active):
            sl = dc[pos * n_cand : (pos + 1) * n_cand]
            best = int(sl.argmin())
            st = states[i]
            if float(sl[best]) >= st["cur_d"] - 1e-9:
                st["active"] = False  # greedy stall
                continue
            gi = pos * n_cand + best
            st["cur_text"] = flat_cands[gi]
            st["cur_x"] = xc[gi]
            st["cur_d"] = float(sl[best])
            st["applied"].append(owner[gi] + vocab[best // 2])
            r0 = sem_ids(model, st["cur_x"][None])[0, 0]
            if int(r0) == traffic_code:
                st["success"] = True
                st["active"] = False

    results = []
    for i in states:
        st = states[i]
        cos_final = float(
            np.dot(st["cur_x"], x_stored[i])
            / (np.linalg.norm(st["cur_x"]) * np.linalg.norm(x_stored[i]))
        )
        results.append(
            {
                "item_id": int(i),
                "success": st["success"],
                "rounds": len(st["applied"]),
                "keywords": st["applied"],
                "cos_to_original": cos_final,
                "original": texts[i],
                "final": st["cur_text"],
            }
        )

    ok = [r for r in results if r["success"]]
    ok95 = [r for r in ok if r["cos_to_original"] >= 0.95]
    summary = {
        "n_items": len(results),
        "n_success": len(ok),
        "n_success_cos_ge_0.95": len(ok95),
        "max_rounds": max_rounds,
        "asr": len(ok) / max(1, len(results)),
        "asr_cos_ge_0.95": len(ok95) / max(1, len(results)),
        "mean_cos_success": float(np.mean([r["cos_to_original"] for r in ok])) if ok else None,
        "min_cos_success": float(np.min([r["cos_to_original"] for r in ok])) if ok else None,
        "mean_rounds_success": float(np.mean([r["rounds"] for r in ok])) if ok else None,
    }
    return results, summary


def run(n_benign=500, n_attack=50, max_rounds=5, seed=0, checkpoint=str(CHECKPOINT),
        vocab_mode="both"):
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
    mined = title_words(target_texts, stop)[:12]
    vocab = mined + PROMO_PHRASES

    margins = audit.sort("margin_l0")
    cand_ids = [
        i for i in margins["item_id"].to_list()
        if int(margins.filter(pl.col("item_id") == i)["code_l0"][0]) != traffic_code
    ][:n_attack]

    results, summ = _greedy_steering(
        model, enc, vocab, cand_ids, traffic_code, texts, x_stored, max_rounds
    )
    out["text_steering"] = {
        "target_traffic_code": traffic_code,
        "vocab": vocab,
        "vocab_mode": "targeted: top-12 title words of popular items on the target"
        " prefix + PROMO_PHRASES",
        **summ,
    }
    print(f"[text_eval] steering (targeted): {out['text_steering']}")

    if vocab_mode == "both":
        # Control arm: 12 catalogue-wide title words drawn uniformly at random
        # (targeted words and promo-phrase words excluded), the same
        # PROMO_PHRASES tail, and the same items/rounds/candidate budget, so
        # the only difference between the arms is targeted-vs-random keyword
        # choice. A separate rng stream keeps the targeted arm unaffected.
        rng_ctrl = np.random.default_rng(seed + 7919)
        excluded = set(mined) | set(" ".join(PROMO_PHRASES).split())
        pool = [w for w in title_words(texts, stop)[:2000] if w not in excluded]
        ctrl_words = sorted(rng_ctrl.choice(pool, size=12, replace=False).tolist())
        ctrl_vocab = ctrl_words + PROMO_PHRASES
        ctrl_results, ctrl_summ = _greedy_steering(
            model, enc, ctrl_vocab, cand_ids, traffic_code, texts, x_stored, max_rounds
        )
        out["text_steering_control"] = {
            "target_traffic_code": traffic_code,
            "vocab": ctrl_vocab,
            "vocab_mode": "control: 12 uniformly random catalogue-wide title words"
            " (targeted/promo words excluded) + same PROMO_PHRASES; identical items,"
            " rounds and candidate budget as the targeted arm",
            **ctrl_summ,
        }
        print(f"[text_eval] steering (control): {out['text_steering_control']}")

    ok = [r for r in results if r["success"]]

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
    ap.add_argument(
        "--vocab-mode",
        choices=["targeted", "both"],
        default="both",
        help="'both' also runs the random-keyword control arm (default)",
    )
    args = ap.parse_args()
    run(
        args.n_benign,
        args.n_attack,
        args.max_rounds,
        args.seed,
        args.checkpoint,
        args.vocab_mode,
    )


if __name__ == "__main__":
    main()
