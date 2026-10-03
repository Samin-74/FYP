"""Item-level decoder evaluation on the trained TIGER retriever baseline.

Fixes the semantic-ID-level metric limitation (docs/interim_checkpoint.md,
limitation 9): upstream's ``TopKAccumulator`` counts a hit when the generated
3-code prefix equals the target's prefix, which identifies a *collision
group*, not an item (27% of items share their prefix; the largest group has
248 items). The decoder was trained to emit only the 3-code prefix (the dedup
token is stripped upstream), so true item-level generation is impossible
without retraining. This script instead evaluates the SAME trained decoder
three ways on the SAME test split:

- ``prefix``: the upstream criterion, reproduced as a regression gate (must
  match artifacts/runs/decoder_baseline.json within sampling noise).
- ``item_strict``: prefix hit AND the target's prefix is unique in the
  catalogue (group size 1) — the generated ID then identifies the item.
- ``item_fractional``: a prefix hit is credited 1/group_size — the expected
  item-level recall if the platform breaks ties uniformly at random inside a
  collision group. ``item_strict`` <= true item-level <= ``prefix``; the
  fractional variant is the point estimate under a stated tie-breaking rule.

Generation samples candidates (torch.multinomial), so every metric is
averaged over several seeds and reported as mean +/- s.d.

Hyperparameters mirror configs/decoder_beauty_fyp.gin (kept in sync by hand;
the baseline checkpoint was trained with them).

Usage:
  python -m fyp.eval.eval_decoder_itemlevel [--seeds 0 1 2 3 4]
      [--checkpoint artifacts/checkpoints/decoder/amazon/checkpoint_9999.pt]
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer

from fyp.common.paths import CHECKPOINT, DATASET_ROOT, RUNS_DIR

# --- configs/decoder_beauty_fyp.gin (baseline run) ---
T5 = {"d_model": 384, "num_heads": 6, "d_ff": 1024, "num_layers": 4}
VAE = {"input_dim": 768, "hidden_dims": [512, 256, 128], "embed_dim": 32,
       "codebook_size": 256, "n_layers": 3}
TOP_K = 10
DEFAULT_DECODER_CKPT = "artifacts/checkpoints/decoder/amazon/checkpoint_9999.pt"
OUT_JSON = RUNS_DIR / "decoder_itemlevel.json"
KS = (1, 5, 10)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cached_ids: torch.Tensor) -> EncoderDecoderRetrievalModel:
    return EncoderDecoderRetrievalModel(
        codebooks=cached_ids[:, : VAE["n_layers"]].cpu(),
        num_hierarchies=VAE["n_layers"],
        num_embeddings_per_hierarchy=VAE["codebook_size"],
        t5_d_model=T5["d_model"],
        t5_num_heads=T5["num_heads"],
        t5_d_ff=T5["d_ff"],
        t5_num_layers=T5["num_layers"],
        top_k_for_generation=TOP_K,
        should_add_sep_token=True,
    )


def evaluate_seed(model, eval_dataloader, tokenizer, cached_ids, group_size):
    """One full pass over the test split; returns the three metric variants."""
    topk = TopKAccumulator(ks=list(KS))
    hits = {m: {k: 0.0 for k in KS} for m in ("item_strict", "item_fractional")}
    ndcg = {"prefix": 0.0, "item_strict": 0.0, "item_fractional": 0.0}
    total = 0
    for batch in eval_dataloader:
        data = batch_to(batch, DEVICE)
        tokenized = tokenizer(data)
        generated = model.generate_next_sem_id(tokenized, top_k=True, temperature=1)

        # upstream regression gate (collision-group hit criterion)
        topk.accumulate(
            actual=tokenized.sem_ids_fut[:, : VAE["n_layers"]],
            top_k=generated.sem_ids,
        )

        targets = data.ids_fut.squeeze(-1)
        assert (targets >= 0).all(), "eval targets must be catalogue items"
        t_prefix = cached_ids[targets]  # (B, n_layers)
        match = (generated.sem_ids == t_prefix[:, None, :]).all(dim=-1)  # (B, k)
        has = match.any(dim=-1)
        rank = torch.where(
            has, match.float().argmax(dim=-1), torch.full_like(targets, 10**9)
        )
        gsize = group_size[targets].float()
        unique = gsize == 1
        frac = 1.0 / gsize
        w = 1.0 / torch.log2(rank.float() + 2.0)  # 0 when rank = 1e9? no: tiny, not 0
        w = torch.where(has, w, torch.zeros_like(w))
        for k in KS:
            hit_k = has & (rank < k)
            hits["item_strict"][k] += (hit_k & unique).sum().item()
            hits["item_fractional"][k] += (hit_k.float() * frac).sum().item()
        ndcg["prefix"] += w.sum().item()
        ndcg["item_strict"] += (w * unique.float()).sum().item()
        ndcg["item_fractional"] += (w * frac).sum().item()
        total += targets.shape[0]

    prefix = topk.reduce()
    out = {
        "prefix": {
            "recall@1": prefix["h@1"],
            "recall@5": prefix["h@5"],
            "recall@10": prefix["h@10"],
            "ndcg@10": prefix["ndcg"],
        },
        "n_eval": total,
    }
    for m in ("item_strict", "item_fractional"):
        out[m] = {f"recall@{k}": hits[m][k] / total for k in KS}
        out[m]["ndcg@10"] = ndcg[m] / total
    return out


def run(seeds=(0, 1, 2, 3, 4), checkpoint=DEFAULT_DECODER_CKPT, batch_size=640):
    item_dataset = ItemData(
        root=str(DATASET_ROOT), dataset=RecDataset.AMAZON, split="beauty"
    )
    eval_dataset = SeqData(
        root=str(DATASET_ROOT), dataset=RecDataset.AMAZON,
        is_train=False, subsample=False, split="beauty",
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    tokenizer = SemanticIdTokenizer(
        input_dim=VAE["input_dim"],
        hidden_dims=VAE["hidden_dims"],
        output_dim=VAE["embed_dim"],
        codebook_size=VAE["codebook_size"],
        n_layers=VAE["n_layers"],
        n_cat_feats=0,
        rqvae_weights_path=str(CHECKPOINT),
    ).to(DEVICE)
    tokenizer.precompute_corpus_ids(item_dataset)
    cached_ids_full = tokenizer.cached_ids  # (n_items, n_layers + 1) with dedup

    # collision-group structure of the 3-code prefix space
    ids3 = cached_ids_full[:, : VAE["n_layers"]].cpu().numpy()
    keys, counts = np.unique(ids3, axis=0, return_counts=True)
    size_of = {tuple(k): int(c) for k, c in zip(keys.tolist(), counts.tolist())}
    group_size = torch.tensor(
        [size_of[tuple(row)] for row in ids3.tolist()], dtype=torch.long
    ).to(DEVICE)
    cached_ids = cached_ids_full[:, : VAE["n_layers"]].to(DEVICE)

    model = build_model(cached_ids_full)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(DEVICE).eval()
    print(f"[itemlevel] loaded decoder checkpoint {checkpoint} (iter {state['iter']})")

    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        metrics = evaluate_seed(model, eval_dataloader, tokenizer, cached_ids, group_size)
        metrics["seed"] = seed
        per_seed.append(metrics)
        p, fs = metrics["prefix"], metrics["item_fractional"]
        print(
            f"[itemlevel] seed {seed}: prefix R@10 {p['recall@10']:.4f} |"
            f" item-strict R@10 {metrics['item_strict']['recall@10']:.4f} |"
            f" item-frac R@10 {fs['recall@10']:.4f}"
        )

    variants = ("prefix", "item_strict", "item_fractional")
    names = ("recall@1", "recall@5", "recall@10", "ndcg@10")
    mean, std = {}, {}
    for v in variants:
        mean[v] = {
            m: float(np.mean([s[v][m] for s in per_seed])) for m in names
        }
        std[v] = {
            m: float(np.std([s[v][m] for s in per_seed], ddof=1)) for m in names
        }

    gsize_cpu = group_size.cpu()
    n_eval = per_seed[0]["n_eval"]
    out = {
        "model": "EncoderDecoderRetrievalModel (TIGER-style T5 decoder)",
        "t5": T5,
        "tokenizer": "RQ-VAE-Recommender-main/trained_models/rqvae_amazon_beauty/"
        "checkpoint_399999.pt (published, 3x256)",
        "dataset": "classic Amazon Beauty 5-core (12,101 items, 22,363 users)",
        "config": "configs/decoder_beauty_fyp.gin",
        "checkpoint": checkpoint,
        "eval_split": "test (SeqData is_train=False; same split as decoder_baseline.json)",
        "n_eval": n_eval,
        "seeds": list(seeds),
        "metric_definitions": {
            "prefix": "upstream TopKAccumulator: hit iff a generated 3-code prefix"
            " equals the target's prefix (collision-group hit)",
            "item_strict": "prefix hit AND the target's prefix is unique in the"
            " catalogue (group size 1), so the generated ID identifies the item",
            "item_fractional": "prefix hit credited 1/group_size (expected item-level"
            " recall under uniform tie-breaking inside a collision group)",
        },
        "collision_structure": {
            "frac_items_unique_prefix": float((gsize_cpu == 1).float().mean()),
            "mean_inv_group_size_items": float((1.0 / gsize_cpu.float()).mean()),
            "note": "item-level ceiling: strict recall can never exceed"
            " frac_items_unique_prefix for prefix-only generation",
        },
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "reference_prefix_level_baseline": {
            "recall@1": 0.02343156106068059,
            "recall@5": 0.05330232974109019,
            "recall@10": 0.0729329696373474,
            "ndcg": 0.04527881333862408,
            "source": "artifacts/runs/decoder_baseline.json (single upstream full-eval)",
        },
        "tiger_paper_item_level_beauty": {
            "recall@5": 0.0454,
            "recall@10": 0.0648,
            "ndcg@10": 0.0384,
            "source": "Rajput et al. 2023 (TIGER), Table; paper-scale model + data",
        },
        "note": "Item-level metrics for the SAME checkpoint without retraining."
        " The decoder only emits 3-code prefixes, so true item-level generation"
        " (with the dedup token) is impossible here; item_strict and"
        " item_fractional bound/estimate the item-level value. Comparison with"
        " the TIGER paper is indicative only: their model is paper-scale and"
        " generates full 4-token IDs.",
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[itemlevel] wrote {OUT_JSON}")

    r10 = mean["prefix"]["recall@10"]
    ref = out["reference_prefix_level_baseline"]["recall@10"]
    print(
        f"[itemlevel] GATE prefix recall@10 mean {r10:.4f} vs baseline {ref:.4f}"
        f" (|diff| {abs(r10 - ref):.4f}; expect sampling-noise agreement)"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--checkpoint", default=DEFAULT_DECODER_CKPT)
    ap.add_argument("--batch-size", type=int, default=640)
    args = ap.parse_args()
    run(tuple(args.seeds), args.checkpoint, args.batch_size)


if __name__ == "__main__":
    main()
