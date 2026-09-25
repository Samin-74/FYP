"""Publication-ready audit figures + report numbers (interim report material).

Reads artifacts/audit/audit_items.csv + audit_summary.json (produced by
fyp.audit.run_audit) and writes, into artifacts/audit/:

  fig_margin_hist_by_level.png      boundary-margin distributions (log-log)
  fig_margin_cdf_by_level.png       CDF of margin relative to level median
  fig_eps_threshold_curve.png       %% within eps x median vs eps
  fig_collision_group_sizes.png     collision group size distribution (log-log)
  fig_steerability_vs_popularity.png
  fig_steerability_vs_desclen.png
  fig_sensitivity_by_category.png
  tab_summary_stats.csv             one flat table of headline numbers
  report_numbers.md                 every key number, labelled, for the report

Usage:
  python -m fyp.audit.report_figures
"""

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fyp.common.paths import AUDIT_DIR, AUDIT_ITEMS_CSV, AUDIT_SUMMARY_JSON

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})

LEVELS = [0, 1, 2]
LEVEL_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def fig_margin_hist(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in LEVELS:
        m = df[f"margin_l{k}"].to_numpy()
        m = m[m > 0]
        bins = np.logspace(np.log10(m.min()), np.log10(m.max()), 60)
        ax.hist(m, bins=bins, histtype="step", lw=1.5,
                color=LEVEL_COLORS[k], label=f"level {k}", density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("boundary margin $d_2 - d_1$ (squared L2, residual space)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_margin_hist_by_level.png")
    plt.close(fig)


def fig_margin_cdf(df, summary):
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.logspace(-2, 2, 200)
    for k in LEVELS:
        m = df[f"margin_l{k}"].to_numpy()
        med = summary[f"level_{k}"]["margin_median"]
        r = np.sort(m / med)
        cdf = np.searchsorted(r, grid, side="right") / len(r)
        ax.plot(grid, cdf * 100, color=LEVEL_COLORS[k], label=f"level {k}")
    for e in (0.25, 0.5, 1.0):
        ax.axvline(e, color="gray", ls=":", lw=0.8)
        ax.text(e, 5, f"{e}x", rotation=90, color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("margin / level median")
    ax.set_ylabel("% of items within factor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_margin_cdf_by_level.png")
    plt.close(fig)


def fig_eps_curve(df, summary):
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.logspace(np.log10(0.05), np.log10(20), 60)
    for k in LEVELS:
        m = df[f"margin_l{k}"].to_numpy()
        med = summary[f"level_{k}"]["margin_median"]
        pct = [(m < e * med).mean() * 100 for e in grid]
        ax.plot(grid, pct, color=LEVEL_COLORS[k], label=f"level {k}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$ (units of level median margin)")
    ax.set_ylabel(r"% of items with margin $< \varepsilon$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_eps_threshold_curve.png")
    plt.close(fig)


def fig_collision_sizes(df):
    cols = [f"code_l{k}" for k in LEVELS]
    counts = df.group_by(cols).len()["len"].to_numpy()
    sizes, freq = np.unique(counts, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sizes, freq, s=18, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("collision group size (items sharing one semantic ID)")
    ax.set_ylabel("number of groups")
    n_colliding = int((counts > 1).sum())
    ax.set_title(f"{n_colliding} colliding groups / {len(counts)} unique IDs")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_collision_group_sizes.png")
    plt.close(fig)


def _binned_median(x, y, n_bins=12):
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = x[0], x[-1] + 1e-9
    cx, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 5:
            cx.append(x[mask].mean())
            med.append(np.median(y[mask]))
    return np.array(cx), np.array(med)


def fig_steerability_vs_popularity(df):
    d = df.filter(pl.col("flipped_l0").is_not_null() & (pl.col("n_interactions").is_not_null()))
    x = np.log1p(d["n_interactions"].to_numpy())
    y = d["sensitivity_l0"].to_numpy()
    fin = np.isfinite(y)
    cx, med = _binned_median(x[fin], y[fin])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[fin], y[fin], s=4, alpha=0.15, label="items (flippable)")
    ax.plot(cx, med, color="red", lw=2, label="binned median")
    ax.set_xlabel("log(1 + interactions)")
    ax.set_ylabel("sensitivity ($\\|\\delta\\|_2$ to flip level-0 code)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_steerability_vs_popularity.png")
    plt.close(fig)


def fig_steerability_vs_desclen(df):
    d = df.filter(pl.col("desc_len").is_not_null())
    x = d["desc_len"].to_numpy().astype(float)
    y = d["margin_l0"].to_numpy() / np.maximum(d["d1_l0"].to_numpy(), 1e-12)
    cx, med = _binned_median(x, y)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=4, alpha=0.1)
    ax.plot(cx, med, color="red", lw=2, label="binned median")
    ax.set_xlabel("description length (characters)")
    ax.set_ylabel("relative boundary margin $(d_2-d_1)/d_1$, level 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_steerability_vs_desclen.png")
    plt.close(fig)


def fig_sensitivity_by_category(df, top_n=8):
    cats = (
        df.group_by("category").len().sort("len", descending=True)["category"][:top_n].to_list()
    )
    d = df.filter(pl.col("category").is_in(cats) & pl.col("flipped_l0"))
    data = [d.filter(pl.col("category") == c)["sensitivity_l0"].to_numpy() for c in cats]
    counts = [len(a) for a in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, tick_labels=[f"{c}\n(n={n})" for c, n in zip(cats, counts)],
                    showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("sensitivity ($\\|\\delta\\|_2$ to flip level-0 code)")
    ax.set_xlabel("category (top %d by item count)" % top_n)
    plt.setp(bp["medians"], color="red")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fig_sensitivity_by_category.png")
    plt.close(fig)


def summary_table(df, summary):
    rows = {}
    s = summary["sensitivity"]
    rows["n_items"] = summary["n_items"]
    rows["collision_rate"] = summary["collisions"]["collision_rate"]
    rows["n_unique_ids"] = summary["collisions"]["n_unique_ids"]
    rows["largest_collision_group"] = summary["collisions"]["largest_collision_group"]
    rows["frac_items_in_colliding_groups"] = summary["collisions"]["frac_items_in_colliding_groups"]
    rows["n_colliding_groups"] = summary["collisions"]["n_colliding_groups"]
    rows["sensitivity_flip_rate_sample"] = s["flip_rate"]
    rows["sensitivity_effective_budget"] = s["effective_budget"]
    rows["embedding_norm_p50"] = summary["embedding_geometry"]["embedding_norm_p50"]
    rows["nn_cosine_p50"] = summary["embedding_geometry"]["nn_cosine_p50"]
    rows["frac_nn_cosine_ge_0.95"] = summary["embedding_geometry"]["frac_nn_cosine_ge"]["0.95"]
    rows["sensitivity_tau_median"] = s["tau_median"]
    for k in LEVELS:
        L = summary[f"level_{k}"]
        rows[f"l{k}_margin_median"] = L["margin_median"]
        rows[f"l{k}_margin_over_d1_p50"] = L["margin_over_d1_p50"]
        rows[f"l{k}_within_0.5x_median"] = L["frac_within_eps_x_median"]["0.5"]
        rows[f"l{k}_within_0.25x_median"] = L["frac_within_eps_x_median"]["0.25"]
        rows[f"l{k}_popularity_r2"] = L["popularity_regression"].get("r_squared")
    return pl.DataFrame(
        {"metric": list(rows), "value": [float(v) for v in rows.values()]}
    )


def report_numbers_md(df, summary):
    s = summary["sensitivity"]
    c = summary["collisions"]
    g = summary["embedding_geometry"]
    L = {k: summary[f"level_{k}"] for k in LEVELS}
    lines = [
        "# Audit numbers (auto-generated by fyp.audit.report_figures)",
        "",
        f"- Catalogue: **{summary['n_items']:,} items** (classic Amazon Beauty 5-core),"
        f" 3-level RQ-VAE, 256-entry codebooks per level.",
        f"- Semantic ID collisions: {c['n_unique_ids']:,} unique IDs for"
        f" {summary['n_items']:,} items, so **{c['collision_rate']:.1%}** of items need a"
        f" dedup token > 0 (1 - unique/items); **{c['frac_items_in_colliding_groups']:.1%}**"
        f" of items share their full ID with at least one other item"
        f" ({c['n_colliding_groups']:,} colliding groups; largest"
        f" **{c['largest_collision_group']} items**).",
        "",
        "## Embedding geometry (constraint calibration)",
        "",
        f"- Sentence-T5 embedding norm: p1 {g['embedding_norm_p1']:.4f},"
        f" p50 {g['embedding_norm_p50']:.4f}, p99 {g['embedding_norm_p99']:.4f}"
        " (attacks are projected back onto each item's original norm).",
        f"- Cosine to nearest *other* catalogue item: p5 {g['nn_cosine_p5']:.3f},"
        f" p50 {g['nn_cosine_p50']:.3f}, p95 {g['nn_cosine_p95']:.3f}.",
        f"- Items whose nearest distinct product already has cos >= 0.95:"
        f" **{g['frac_nn_cosine_ge']['0.95']:.1%}** — if large, cos >= 0.95 alone does not"
        " guarantee the listing still describes the same product.",
        "",
        "## Boundary margins (squared L2 in residual space; margin = d2 - d1)",
        "",
        "| Level | Median margin | Median (d2-d1)/d1 | p1 | p99 |"
        " within 0.25x median | within 0.5x median |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in LEVELS:
        l = L[k]
        lines.append(
            f"| {k} | {l['margin_median']:.3e} | {l['margin_over_d1_p50']:.3f} "
            f"| {l['margin_p1']:.3e} | {l['margin_p99']:.3e} "
            f"| {l['frac_within_eps_x_median']['0.25']:.1%} "
            f"| {l['frac_within_eps_x_median']['0.5']:.1%} |"
        )
    lines += [
        "",
        "Note: \"within eps x median\" is relative to the level's own median margin,"
        " so it describes the distribution's shape (an exponential distribution"
        " gives ~16% below 0.25x median). It is not by itself evidence that items"
        " are close to a boundary in absolute terms.",
        "",
        "## Flip sensitivity (margin gradient descent, level 0, random sample of "
        f"{s['sample_size']:,} items, reachable ||delta|| <= {s['effective_budget']:g},"
        " norm-preserving, no cosine constraint)",
        "",
        f"- Flip rate on the sample: **{s['flip_rate']:.1%}**.",
        f"- Median perturbation of flippable items (tau): **{s['tau_median']:.3f}** (L2 in"
        " 768-d Sentence-T5 embedding space).",
        "",
        "## Popularity regression (margin vs log-interactions)",
        "",
        "| Level | slope | R^2 |",
        "|---|---|---|",
    ]
    for k in LEVELS:
        r = L[k]["popularity_regression"]
        lines.append(f"| {k} | {r.get('slope', float('nan')):.3e} | {r.get('r_squared', float('nan')):.2e} |")
    lines += [
        "",
        "## Figures",
        "",
        "- `fig_margin_hist_by_level.png` — margin distributions.",
        "- `fig_margin_cdf_by_level.png` — within-factor CDF.",
        "- `fig_eps_threshold_curve.png` — epsilon-threshold curve.",
        "- `fig_collision_group_sizes.png` — collision group sizes.",
        "- `fig_steerability_vs_popularity.png` / `fig_steerability_vs_desclen.png` /",
        "  `fig_sensitivity_by_category.png` — steerability vs item properties.",
    ]
    (AUDIT_DIR / "report_numbers.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = pl.read_csv(AUDIT_ITEMS_CSV)
    summary = json.loads(AUDIT_SUMMARY_JSON.read_text())

    fig_margin_hist(df)
    fig_margin_cdf(df, summary)
    fig_eps_curve(df, summary)
    fig_collision_sizes(df)
    fig_steerability_vs_popularity(df)
    fig_steerability_vs_desclen(df)
    fig_sensitivity_by_category(df)
    summary_table(df, summary).write_csv(AUDIT_DIR / "tab_summary_stats.csv")
    report_numbers_md(df, summary)
    print("[report] wrote figures + tab_summary_stats.csv + report_numbers.md to artifacts/audit/")


if __name__ == "__main__":
    main()
