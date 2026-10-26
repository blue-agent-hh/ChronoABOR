#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl

from nm_exp4_full_universe import RANDOM_SEED, make_xgb, to_jsonable
from nm_exp_p6_secreted_ablation import (
    FEATURE_CUTOFF_YEAR,
    METHOD_SPECS,
    build_fold_c_frame,
    compute_rank_metrics,
    rank_frame,
    summarize_distribution,
)

_mpl.rcParams["svg.fonttype"] = "none"
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p8_rank_stability.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p8_stability.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_p8_stability.pdf"

BOOTSTRAP_ITERATIONS = 2000
TOP_K_VALUES = [5, 10]


def summarize_metric(values: np.ndarray) -> dict:
    summary = summarize_distribution(values)
    summary["ci95"] = {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }
    return summary


def fit_and_score(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    kind: str,
    feature_columns: list[str],
) -> np.ndarray:
    if kind == "constant":
        return np.ones(len(eval_df), dtype=float)
    if kind == "single":
        return eval_df[feature_columns[0]].to_numpy(dtype=float)

    model = make_xgb()
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_eval = eval_df[feature_columns].to_numpy(dtype=float)
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def compute_metric_row(eval_df: pd.DataFrame, scores: np.ndarray, kind: str) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    rank_metrics = compute_rank_metrics(eval_df, scores, kind)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "p_at_5": float(rank_metrics["p_at_5"]),
        "p_at_10": float(rank_metrics["p_at_10"]),
        "top_10_genes": rank_metrics["top_genes"],
    }


def run_jackknife(eval_df: pd.DataFrame) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    positive_genes = sorted(eval_df.loc[eval_df["label"] == 1, "gene"].tolist())
    full_model_spec = next(spec for spec in METHOD_SPECS if spec["name"] == "full_model")

    rows: list[dict] = []
    held_out_ranks: list[int] = []
    aurocs: list[float] = []

    for held_out_gene in positive_genes:
        train_df = eval_df.loc[eval_df["gene"] != held_out_gene].copy().reset_index(drop=True)
        scores = fit_and_score(train_df, eval_df, full_model_spec["kind"], full_model_spec["features"])
        ranked_df = rank_frame(eval_df, scores)
        held_out_rank = int(ranked_df.index[ranked_df["gene"] == held_out_gene][0]) + 1
        held_out_score = float(ranked_df.loc[ranked_df["gene"] == held_out_gene, "score"].iloc[0])
        jackknife_auc = float(roc_auc_score(labels, scores))

        rows.append(
            {
                "held_out_gene": held_out_gene,
                "held_out_rank": held_out_rank,
                "held_out_score": held_out_score,
                "jackknife_auroc": jackknife_auc,
                "top_10_genes": ranked_df.head(10)["gene"].tolist(),
            }
        )
        held_out_ranks.append(held_out_rank)
        aurocs.append(jackknife_auc)

    rank_array = np.asarray(held_out_ranks, dtype=float)
    auc_array = np.asarray(aurocs, dtype=float)
    summary = {
        "n_iterations": int(len(rows)),
        "mean_rank": float(np.mean(rank_array)),
        "median_rank": float(np.median(rank_array)),
        "std_rank": float(np.std(rank_array, ddof=1)) if len(rank_array) > 1 else 0.0,
        "fraction_top_5": float(np.mean(rank_array <= 5)),
        "fraction_top_10": float(np.mean(rank_array <= 10)),
        "mean_auroc": float(np.mean(auc_array)),
        "std_auroc": float(np.std(auc_array, ddof=1)) if len(auc_array) > 1 else 0.0,
    }
    return {
        "summary": summary,
        "iterations": rows,
    }


def sample_stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    sampled_pos = rng.choice(positive_idx, size=len(positive_idx), replace=True)
    sampled_neg = rng.choice(negative_idx, size=len(negative_idx), replace=True)
    return np.concatenate([sampled_pos, sampled_neg])


def run_bootstrap(eval_df: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    labels = eval_df["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)

    metric_arrays = {
        spec["name"]: {
            "auroc": np.empty(BOOTSTRAP_ITERATIONS, dtype=float),
            "p_at_5": np.empty(BOOTSTRAP_ITERATIONS, dtype=float),
            "p_at_10": np.empty(BOOTSTRAP_ITERATIONS, dtype=float),
        }
        for spec in METHOD_SPECS
    }

    static_scores: dict[str, np.ndarray] = {}
    for spec in METHOD_SPECS:
        if spec["kind"] in {"constant", "single"}:
            static_scores[spec["name"]] = fit_and_score(eval_df, eval_df, spec["kind"], spec["features"])

    for iteration in range(BOOTSTRAP_ITERATIONS):
        train_idx = sample_stratified_indices(labels, rng)
        train_df = eval_df.iloc[train_idx].copy().reset_index(drop=True)

        for spec in METHOD_SPECS:
            if spec["name"] in static_scores:
                scores = static_scores[spec["name"]]
            else:
                scores = fit_and_score(train_df, eval_df, spec["kind"], spec["features"])
            metric_row = compute_metric_row(eval_df, scores, spec["kind"])
            metric_arrays[spec["name"]]["auroc"][iteration] = metric_row["auroc"]
            metric_arrays[spec["name"]]["p_at_5"][iteration] = metric_row["p_at_5"]
            metric_arrays[spec["name"]]["p_at_10"][iteration] = metric_row["p_at_10"]

    bootstrap_summary = {
        method_name: {
            metric_name: summarize_metric(metric_values)
            for metric_name, metric_values in metrics.items()
        }
        for method_name, metrics in metric_arrays.items()
    }
    return metric_arrays, bootstrap_summary


def compute_paired_p_values(metric_arrays: dict[str, dict[str, np.ndarray]]) -> dict:
    full_model = metric_arrays["full_model"]
    paired = {
        "definition": "Fraction of bootstrap iterations where full_model metric <= comparison method metric.",
        "auroc": {},
        "p_at_10": {},
    }
    for spec in METHOD_SPECS:
        name = spec["name"]
        paired["auroc"][name] = float(np.mean(full_model["auroc"] <= metric_arrays[name]["auroc"]))
        paired["p_at_10"][name] = float(np.mean(full_model["p_at_10"] <= metric_arrays[name]["p_at_10"]))
    return paired


def create_figure(
    jackknife_results: dict,
    bootstrap_summary: dict,
    paired_p_values: dict,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.5),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.25]},
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    jackknife_df = pd.DataFrame(jackknife_results["iterations"]).sort_values(
        by=["held_out_rank", "held_out_gene"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ax = axes[0]
    x = np.arange(len(jackknife_df))
    ax.vlines(x, 35, jackknife_df["held_out_rank"], color="#b8c0cc", linewidth=1.2, zorder=1)
    ax.scatter(x, jackknife_df["held_out_rank"], color="#bc4749", s=52, zorder=2)
    ax.axhline(5, color="#2a9d8f", linestyle="--", linewidth=1.2)
    ax.axhline(10, color="#577590", linestyle="--", linewidth=1.2)
    ax.axhline(jackknife_results["summary"]["mean_rank"], color="#1d3557", linestyle=":", linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(jackknife_df["held_out_gene"], rotation=45, ha="right")
    ax.set_ylim(35.5, 0.5)
    ax.set_ylabel("Held-out positive rank (1 = best)")
    ax.set_title(
        "Jackknife Leave-One-Positive-Out\n"
        f"mean rank={jackknife_results['summary']['mean_rank']:.2f}, "
        f"median={jackknife_results['summary']['median_rank']:.1f}"
    )
    ax.text(
        0.02,
        0.04,
        (
            f"Top-5: {jackknife_results['summary']['fraction_top_5']:.1%}\n"
            f"Top-10: {jackknife_results['summary']['fraction_top_10']:.1%}\n"
            f"AUROC SD: {jackknife_results['summary']['std_auroc']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    ax = axes[1]
    metric_labels = ["P@5", "P@10"]
    x = np.arange(len(metric_labels))
    width = 0.34
    series = [
        ("full_model", "Full model", "#D55E00", -width / 2),
        ("sig_max_auc", "sig_max_auc", "#56B4E9", width / 2),
    ]
    for method_name, label, color, offset in series:
        means = [
            bootstrap_summary[method_name]["p_at_5"]["mean"],
            bootstrap_summary[method_name]["p_at_10"]["mean"],
        ]
        lower = [
            means[0] - bootstrap_summary[method_name]["p_at_5"]["ci95"]["lower"],
            means[1] - bootstrap_summary[method_name]["p_at_10"]["ci95"]["lower"],
        ]
        upper = [
            bootstrap_summary[method_name]["p_at_5"]["ci95"]["upper"] - means[0],
            bootstrap_summary[method_name]["p_at_10"]["ci95"]["upper"] - means[1],
        ]
        lower = [max(0.0, value) for value in lower]
        upper = [max(0.0, value) for value in upper]
        ax.bar(
            x + offset,
            means,
            width=width,
            label=label,
            color=color,
            edgecolor="#1f2933",
            linewidth=1.0,
            yerr=np.vstack([lower, upper]),
            capsize=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Bootstrap precision")
    ax.set_title("Repeated Bootstrap P@K (2,000 iterations)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.02,
        0.04,
        (
            "One-sided p for P@10:\n"
            f"Pr(full_model <= sig_max_auc) = {paired_p_values['p_at_10']['sig_max_auc']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    ax = axes[2]
    method_names = [spec["name"] for spec in METHOD_SPECS]
    heatmap = np.array(
        [
            [paired_p_values["auroc"][name] for name in method_names],
            [paired_p_values["p_at_10"][name] for name in method_names],
        ],
        dtype=float,
    )
    image = ax.imshow(heatmap, cmap="viridis_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["AUROC", "P@10"])
    ax.set_title("Paired Bootstrap p-values\nPr(full_model <= comparator)")
    for row_idx in range(heatmap.shape[0]):
        for col_idx in range(heatmap.shape[1]):
            value = heatmap[row_idx, col_idx]
            text_color = "white" if value < 0.45 else "#111827"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("NM Experiment P8: Within-secreted rank stability", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    secreted_df, dataset_summary = build_fold_c_frame()
    if len(secreted_df) != 35 or int(secreted_df["label"].sum()) != 13:
        raise RuntimeError(
            f"Unexpected secreted subset composition: {len(secreted_df)} genes, "
            f"{int(secreted_df['label'].sum())} positives."
        )

    jackknife_results = run_jackknife(secreted_df)
    metric_arrays, bootstrap_summary = run_bootstrap(secreted_df)
    paired_p_values = compute_paired_p_values(metric_arrays)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "method_order": [spec["name"] for spec in METHOD_SPECS],
        "dataset_summary": dataset_summary,
        "analysis_1_jackknife_leave_one_positive_out": jackknife_results,
        "analysis_2_repeated_bootstrap_p_at_k": {
            "full_model": {
                "p_at_5": bootstrap_summary["full_model"]["p_at_5"],
                "p_at_10": bootstrap_summary["full_model"]["p_at_10"],
            },
            "sig_max_auc": {
                "p_at_5": bootstrap_summary["sig_max_auc"]["p_at_5"],
                "p_at_10": bootstrap_summary["sig_max_auc"]["p_at_10"],
            },
            "paired_one_sided_p_value_full_model_gt_sig_max_auc_p_at_10": paired_p_values["p_at_10"][
                "sig_max_auc"
            ],
        },
        "analysis_3_paired_bootstrap_significance_vs_full_model": paired_p_values,
        "bootstrap_metric_summaries": bootstrap_summary,
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))
    create_figure(jackknife_results, bootstrap_summary, paired_p_values)

    print("NM Experiment P8: Rank stability within the secreted subset")
    print(
        f"Secreted subset: {dataset_summary['n_secreted_subset']} genes "
        f"({dataset_summary['n_secreted_positives']} positives, {dataset_summary['n_secreted_negatives']} negatives)"
    )
    print(
        "Jackknife held-out positive rank: "
        f"mean={jackknife_results['summary']['mean_rank']:.3f}, "
        f"median={jackknife_results['summary']['median_rank']:.3f}, "
        f"top5={jackknife_results['summary']['fraction_top_5']:.3f}, "
        f"top10={jackknife_results['summary']['fraction_top_10']:.3f}"
    )
    print(
        "Jackknife AUROC SD: "
        f"{jackknife_results['summary']['std_auroc']:.4f}"
    )
    print(
        "Bootstrap P@5/P@10 (full_model): "
        f"{bootstrap_summary['full_model']['p_at_5']['mean']:.3f} "
        f"[{bootstrap_summary['full_model']['p_at_5']['ci95']['lower']:.3f}, "
        f"{bootstrap_summary['full_model']['p_at_5']['ci95']['upper']:.3f}] / "
        f"{bootstrap_summary['full_model']['p_at_10']['mean']:.3f} "
        f"[{bootstrap_summary['full_model']['p_at_10']['ci95']['lower']:.3f}, "
        f"{bootstrap_summary['full_model']['p_at_10']['ci95']['upper']:.3f}]"
    )
    print(
        "Bootstrap P@5/P@10 (sig_max_auc): "
        f"{bootstrap_summary['sig_max_auc']['p_at_5']['mean']:.3f} "
        f"[{bootstrap_summary['sig_max_auc']['p_at_5']['ci95']['lower']:.3f}, "
        f"{bootstrap_summary['sig_max_auc']['p_at_5']['ci95']['upper']:.3f}] / "
        f"{bootstrap_summary['sig_max_auc']['p_at_10']['mean']:.3f} "
        f"[{bootstrap_summary['sig_max_auc']['p_at_10']['ci95']['lower']:.3f}, "
        f"{bootstrap_summary['sig_max_auc']['p_at_10']['ci95']['upper']:.3f}]"
    )
    print(
        "Paired bootstrap p-value Pr(full_model <= sig_max_auc) at P@10: "
        f"{paired_p_values['p_at_10']['sig_max_auc']:.4f}"
    )
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
