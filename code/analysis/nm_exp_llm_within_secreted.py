#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl

from nm_exp_p1_prior_control import RANDOM_SEED, SECRETED_GENES, to_jsonable
from nm_exp_p6_secreted_ablation import (
    FULL_FEATURES,
    build_fold_c_frame,
    compute_cv_scores,
    compute_full_fit_scores,
    compute_rank_metrics,
    stratified_bootstrap_auc,
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
PRIMARY_LLM_CACHE = RESULTS_DIR / "nm_exp5_llm_cache.json"
GEMINI_CACHE = RESULTS_DIR / "nm_exp_gemini_cache.json"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_llm_within_secreted.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_llm_within_secreted.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_llm_within_secreted.pdf"
CHRONOABOR_P6_METHOD_INDEX = 6

NM_CORPUS = [
    "NEFL",
    "GFAP",
    "MAPT",
    "APOE",
    "APP",
    "CLU",
    "VGF",
    "NRGN",
    "SNAP25",
    "YWHAZ",
    "SYT1",
    "NEFH",
    "NEFM",
    "GAP43",
    "ACHE",
    "VILIP1",
    "HSPB1",
    "PKM",
    "ALDOA",
    "CFL1",
    "TMOD2",
    "BASP1",
    "STMN2",
    "STMN1",
    "MAP2",
    "TUBB3",
    "ENO2",
    "NSE",
    "CAMK2A",
    "CAMK2B",
    "CAMK4",
    "BDNF",
]

MODEL_SPECS = [
    {
        "model_name": "gpt-5.2",
        "display_name": "GPT-5.2",
        "cache_file": PRIMARY_LLM_CACHE,
        "cache_section": "models",
    },
    {
        "model_name": "gpt-5.4",
        "display_name": "GPT-5.4",
        "cache_file": PRIMARY_LLM_CACHE,
        "cache_section": "models",
    },
    {
        "model_name": "gemini-2.5-flash",
        "display_name": "Gemini-2.5-Flash",
        "cache_file": GEMINI_CACHE,
        "cache_section": None,
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_scores_for_model(cache_file: Path, cache_section: str | None, model_name: str) -> dict[str, float]:
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing score cache: {cache_file}")
    payload = load_json(cache_file)
    container = payload[cache_section] if cache_section else payload
    if model_name not in container:
        raise KeyError(f"Model {model_name!r} not found in {cache_file}")
    return {gene: float(score) for gene, score in container[model_name].items()}


def evaluate_fixed_scores(
    eval_df: pd.DataFrame,
    score_map: dict[str, float],
    model_name: str,
    display_name: str,
    source_file: Path,
    bootstrap_seed: int,
) -> dict:
    missing_genes = [gene for gene in eval_df["gene"] if gene not in score_map]
    if missing_genes:
        raise RuntimeError(f"{model_name} is missing scores for secreted genes: {missing_genes}")

    scored_df = eval_df[["gene", "label"]].copy()
    scored_df["score"] = scored_df["gene"].map(score_map).astype(float)

    labels = scored_df["label"].to_numpy(dtype=int)
    scores = scored_df["score"].to_numpy(dtype=float)
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    auc_ci, bootstrap_values = stratified_bootstrap_auc(labels=labels, scores=scores, seed=bootstrap_seed)
    rank_metrics = compute_rank_metrics(eval_df, scores, kind="fixed_score")

    return {
        "model_name": model_name,
        "display_name": display_name,
        "model_family": "llm",
        "source_file": str(source_file.relative_to(BASE_DIR)),
        "n_secreted_genes": int(len(scored_df)),
        "n_secreted_positives": int(scored_df["label"].sum()),
        "n_secreted_negatives": int((scored_df["label"] == 0).sum()),
        "n_missing_scores": 0,
        "score_scale_note": "Raw cached scores used as-is; AUROC/AUPRC/ranking metrics depend on ordering, not absolute scale.",
        "auroc": auroc,
        "auroc_bootstrap_ci": {
            "lower": float(auc_ci[0]),
            "upper": float(auc_ci[1]),
        },
        "auprc": auprc,
        "bootstrap_summary": {
            "n": int(len(bootstrap_values)),
            "mean": float(np.mean(bootstrap_values)),
            "std": float(np.std(bootstrap_values, ddof=1)),
        },
        "p_at_5": float(rank_metrics["p_at_5"]),
        "p_at_10": float(rank_metrics["p_at_10"]),
        "p_at_20": float(rank_metrics["p_at_20"]),
        "ef_at_10": float(rank_metrics["ef_at_10"]),
        "base_rate_within_secreted_subset": float(rank_metrics["base_rate_within_secreted_subset"]),
        "top_20_genes": rank_metrics["top_genes"],
    }


def evaluate_chronoabor(eval_df: pd.DataFrame) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    cv_scores, _ = compute_cv_scores(eval_df, kind="model", feature_columns=FULL_FEATURES)
    fit_scores = compute_full_fit_scores(eval_df, kind="model", feature_columns=FULL_FEATURES)
    auroc = float(roc_auc_score(labels, cv_scores))
    auprc = float(average_precision_score(labels, cv_scores))
    auc_ci, bootstrap_values = stratified_bootstrap_auc(
        labels=labels,
        scores=cv_scores,
        seed=RANDOM_SEED + CHRONOABOR_P6_METHOD_INDEX,
    )
    rank_metrics = compute_rank_metrics(eval_df, fit_scores, kind="model")

    return {
        "model_name": "chronoabor_full_model",
        "display_name": "ChronoABOR",
        "model_family": "chronoabor",
        "source_file": "nm_exp_p6_secreted_ablation.py",
        "n_secreted_genes": int(len(eval_df)),
        "n_secreted_positives": int(eval_df["label"].sum()),
        "n_secreted_negatives": int((eval_df["label"] == 0).sum()),
        "auroc": auroc,
        "auroc_bootstrap_ci": {
            "lower": float(auc_ci[0]),
            "upper": float(auc_ci[1]),
        },
        "auprc": auprc,
        "bootstrap_summary": {
            "n": int(len(bootstrap_values)),
            "mean": float(np.mean(bootstrap_values)),
            "std": float(np.std(bootstrap_values, ddof=1)),
        },
        "p_at_5": float(rank_metrics["p_at_5"]),
        "p_at_10": float(rank_metrics["p_at_10"]),
        "p_at_20": float(rank_metrics["p_at_20"]),
        "ef_at_10": float(rank_metrics["ef_at_10"]),
        "base_rate_within_secreted_subset": float(rank_metrics["base_rate_within_secreted_subset"]),
        "top_20_genes": rank_metrics["top_genes"],
        "benchmark_match_note": (
            "Recomputed from the P6 within-secreted setup. "
            "AUROC and P@10 should match the existing benchmark row."
        ),
    }


def build_plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    labels = [row["display_name"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2"]

    aurocs = [row["auroc"] for row in rows]
    p10s = [row["p_at_10"] for row in rows]
    lower_errors = [row["auroc"] - row["auroc_bootstrap_ci"]["lower"] for row in rows]
    upper_errors = [row["auroc_bootstrap_ci"]["upper"] - row["auroc"] for row in rows]

    auc_bars = axes[0].bar(
        x,
        aurocs,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=4,
        color=colors[: len(rows)],
        edgecolor="#1f2933",
        linewidth=1.0,
    )
    axes[0].axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.0)
    axes[0].set_title("AUROC")
    axes[0].set_ylabel("Within-secreted performance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0.0, 1.05)

    p10_bars = axes[1].bar(
        x,
        p10s,
        color=colors[: len(rows)],
        edgecolor="#1f2933",
        linewidth=1.0,
    )
    axes[1].set_title("P@10")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)

    for ax, bars, values in [(axes[0], auc_bars, aurocs), (axes[1], p10_bars, p10s)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.03, 1.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Within-secreted NM benchmark: ChronoABOR vs LLMs", fontsize=11)
    fig.tight_layout()
    paper_figures_dir = BASE_DIR / "paper" / "figures"
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(paper_figures_dir / "fig_llm_within_secreted.png", dpi=300, bbox_inches="tight")
    fig.savefig(paper_figures_dir / "fig_llm_within_secreted.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Model':<18} {'AUROC':>8} {'95% CI':>19} {'AUPRC':>8} "
        f"{'P@5':>8} {'P@10':>8} {'P@20':>8} {'EF@10':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = f"[{row['auroc_bootstrap_ci']['lower']:.3f}, {row['auroc_bootstrap_ci']['upper']:.3f}]"
        print(
            f"{row['display_name']:<18} "
            f"{format_metric(row['auroc']):>8} "
            f"{ci:>19} "
            f"{format_metric(row['auprc']):>8} "
            f"{format_metric(row['p_at_5']):>8} "
            f"{format_metric(row['p_at_10']):>8} "
            f"{format_metric(row['p_at_20']):>8} "
            f"{format_metric(row['ef_at_10']):>8}"
        )


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    secreted_df, dataset_summary = build_fold_c_frame()
    rows: list[dict] = []

    for idx, spec in enumerate(MODEL_SPECS):
        score_map = load_scores_for_model(
            cache_file=spec["cache_file"],
            cache_section=spec["cache_section"],
            model_name=spec["model_name"],
        )
        rows.append(
            evaluate_fixed_scores(
                eval_df=secreted_df,
                score_map=score_map,
                model_name=spec["model_name"],
                display_name=spec["display_name"],
                source_file=spec["cache_file"],
                bootstrap_seed=RANDOM_SEED + 200 + idx,
            )
        )

    rows.append(evaluate_chronoabor(secreted_df))

    results = {
        "analysis_name": "nm_exp_llm_within_secreted",
        "runtime_seconds": float(time.time() - start_time),
        "nm_corpus_requested": NM_CORPUS,
        "secreted_subset": {
            "n_genes": int(len(secreted_df)),
            "n_positive": int(secreted_df["label"].sum()),
            "n_negative": int((secreted_df["label"] == 0).sum()),
            "genes": secreted_df["gene"].tolist(),
            "positive_genes": secreted_df.loc[secreted_df["label"] == 1, "gene"].tolist(),
            "negative_genes": secreted_df.loc[secreted_df["label"] == 0, "gene"].tolist(),
        },
        "comparison_rows": rows,
        "notes": {
            "secreted_subset_source": (
                "Used the benchmark SECRETED_GENES definition from nm_exp_p1_prior_control.py / "
                "nm_exp_p6_secreted_ablation.py because the current target.csv in this workspace "
                "does not contain the requested Gene_symbol and is_secreted columns."
            ),
            "requested_target_csv": "target.csv",
            "current_target_csv_columns": dataset_summary["target_csv_columns"],
            "gemini_flash_source_note": (
                "The current results/nm_exp5_llm_cache.json does not contain gemini-2.5-flash; "
                "that model was loaded from results/nm_exp_gemini_cache.json."
            ),
            "primary_llm_cache_models": sorted(load_json(PRIMARY_LLM_CACHE).get("models", {}).keys()),
            "secreted_genes_reference_size": int(len(SECRETED_GENES)),
        },
    }

    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))
    build_plot(rows)
    print_table(rows)
    print()
    print(f"Secreted subset: {len(secreted_df)} genes ({int(secreted_df['label'].sum())} positives, {int((secreted_df['label'] == 0).sum())} negatives)")
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
