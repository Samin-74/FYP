"""Prepare semantic-ID sequences for the generative retriever (Phase 1, model track).

Maps raw user interaction sequences (chronological) to semantic-ID sequences
using the audit's item -> code mapping, following the upstream split protocol
(history capped at 20; last item = test target, second-last = eval target).

Output: artifacts/semantic_ids/sequences.parquet with columns
  user_id, item_ids (list[i64]), sem_ids (list[i64], flattened L*len codes),
  sem_id_rows (i64 = len(item_ids)), target_eval, target_test

This is a *data-format* deliverable only — no T5 training here.

Usage:
  python -m fyp.common.prepare_sequences [--max-seq-len 20]
"""

import argparse

import numpy as np
import polars as pl

from fyp.common.paths import DATASET_ROOT, SEMANTIC_IDS_DIR, SEMANTIC_IDS_PARQUET

MAX_SEQ_LEN = 20


def build_sequences(max_seq_len: int = MAX_SEQ_LEN) -> pl.DataFrame:
    sem = pl.read_parquet(SEMANTIC_IDS_PARQUET).sort("item_id")
    codes = np.stack(
        [sem[f"code_l{k}"].to_numpy() for k in range(3)], axis=1
    )  # (n_items, 3)

    user_ids, item_seqs, sem_seqs, targets_eval, targets_test = [], [], [], [], []
    with open(DATASET_ROOT / "raw" / "beauty" / "sequential_data.txt") as f:
        for line in f:
            toks = line.split()
            if len(toks) < 3:
                continue
            uid = int(toks[0])
            items = [int(t) - 1 for t in toks[1:]]  # repo stores ids 1-based
            if max(items) >= codes.shape[0]:
                continue  # safety: item missing from processed catalogue
            hist = items[:-2][-max_seq_len:]
            item_seqs.append(hist)
            sem_seqs.append(codes[hist].flatten().tolist())
            targets_eval.append(items[-2])
            targets_test.append(items[-1])
            user_ids.append(uid)

    df = pl.DataFrame(
        {
            "user_id": user_ids,
            "item_ids": item_seqs,
            "sem_ids": sem_seqs,
            "sem_id_rows": [len(h) for h in item_seqs],
            "target_eval": targets_eval,
            "target_test": targets_test,
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = ap.parse_args()

    df = build_sequences(args.max_seq_len)
    out = SEMANTIC_IDS_DIR / "sequences.parquet"
    df.write_parquet(out)
    print(f"[sequences] wrote {out}: {df.height} users, "
          f"mean history len {np.mean([len(x) for x in df['item_ids'].to_list()]):.1f}")


if __name__ == "__main__":
    main()
