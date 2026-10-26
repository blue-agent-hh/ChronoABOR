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

from nm_exp4_full_universe import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    load_tables,
    to_jsonable,
)
from nm_exp_p1_prior_control import PRIOR_FEATURES, SECRETED_GENES
from nm_exp_p6_secreted_ablation import (
    compute_cv_scores,
    stratified_bootstrap_auc,
    stratified_bootstrap_auc_delta,
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
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p9_rolling_origin.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p9_rolling_origin.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_p9_rolling_origin.pdf"

N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000
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
ROLLING_FOLDS = [
    {"fold_key": "A", "fold_label": "Fold A", "cutoff_year": 2019},
    {"fold_key": "B", "fold_label": "Fold B", "cutoff_year": 2020},
    {"fold_key": "C", "fold_label": "Fold C", "cutoff_year": 2021},
]
FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]
METHOD_SPECS = [
    {
        "name": "full_model",
        "label": "full_model",
        "kind": "model",
        "features": FULL_FEATURES,
    },
    {
        "name": "is_secreted",
        "label": "is_secreted",
        "kind": "constant",
        "features": ["is_secreted"],
    },
    {
        "name": "num_studies",
        "label": "num_studies",
        "kind": "single",
        "features": ["num_studies"],
    },
    {
        "name": "sig_max_auc",
        "label": "sig_max_auc",
        "kind": "single",
        "features": ["sig_max_auc"],
    },
    {
        "name": "prior_only",
        "label": "prior_only",
        "kind": "model",
        "features": PRIOR_FEATURES,
    },
]
TRIVIAL_BASELINES = [spec["name"] for spec in METHOD_SPECS if spec["name"] != "full_model"]


def load_all_inputs() -> tuple[dict[str, pd.DataFrame], list[str], dict[str, int], dict[str, int]]:
    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")
    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    return tables, gene_universe, pmid_year_map, pmid_conflicts


def build_secreted_frame(
    tables: dict[str, pd.DataFrame],
    gene_universe: list[str],
    pmid_year_map: dict[str, int],
    pmid_conflicts: dict[str, int],
    cutoff_year: int,
) -> tuple[pd.DataFrame, dict]:
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=cutoff_year,
    )

    nm_requested_set = set(NM_CORPUS)
    universe_set = set(gene_universe)
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)
    feature_df["label"] = feature_df["gene"].isin(nm_requested_set).astype(int)

    secreted_df = feature_df.loc[feature_df["is_secreted"] == 1].copy().reset_index(drop=True)
    secreted_with_evidence = secreted_df.loc[secreted_df["has_evidence_by_cutoff"]].copy()
    positives_mask = secreted_df["label"] == 1

    dataset_summary = {
        "cutoff_year": int(cutoff_year),
        "n_genes_universe": int(len(feature_df)),
        "n_genes_with_cutoff_evidence": int(len(evidence_by_cutoff)),
        "n_nm_requested": int(len(NM_CORPUS)),
        "n_nm_in_universe": int(feature_df["label"].sum()),
        "nm_genes_in_universe": sorted(universe_set & nm_requested_set),
        "nm_genes_missing_from_universe": sorted(nm_requested_set - universe_set),
        "n_secreted_subset": int(len(secreted_df)),
        "n_secreted_positives": int(positives_mask.sum()),
        "n_secreted_negatives": int((~positives_mask).sum()),
        "n_secreted_with_any_evidence_by_cutoff": int(secreted_df["has_evidence_by_cutoff"].sum()),
        "n_secreted_positives_with_any_evidence_by_cutoff": int(
            secreted_df.loc[positives_mask, "has_evidence_by_cutoff"].sum()
        ),
        "n_secreted_negatives_with_any_evidence_by_cutoff": int(
            secreted_df.loc[~positives_mask, "has_evidence_by_cutoff"].sum()
        ),
        "secreted_positive_genes": sorted(secreted_df.loc[positives_mask, "gene"].tolist()),
        "secreted_negative_genes": sorted(secreted_df.loc[~positives_mask, "gene"].tolist()),
        "secreted_genes_without_evidence_by_cutoff": sorted(
            secreted_df.loc[~secreted_df["has_evidence_by_cutoff"], "gene"].tolist()
        ),
        "secreted_positive_genes_without_evidence_by_cutoff": sorted(
            secreted_df.loc[positives_mask & ~secreted_df["has_evidence_by_cutoff"], "gene"].tolist()
        ),
        "secreted_top_num_studies_by_cutoff": secreted_with_evidence.sort_values(
            by=["num_studies", "gene"],
            ascending=[False, True],
        ).head(10)[["gene", "num_studies"]].to_dict(orient="records"),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
    }
    return secreted_df, dataset_summary


def evaluate_fold(
    fold_spec: dict,
    tables: dict[str, pd.DataFrame],
    gene_universe: list[str],
    pmid_year_map: dict[str, int],
    pmid_conflicts: dict[str, int],
) -> tuple[dict, dict]:
    secreted_df, dataset_summary = build_secreted_frame(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        pmid_conflicts=pmid_conflicts,
        cutoff_year=fold_spec["cutoff_year"],
    )
    if secreted_df["label"].nunique() < 2:
        raise RuntimeError(f"{fold_spec['fold_label']} does not contain both classes.")

    labels = secreted_df["label"].to_numpy(dtype=int)
    prediction_frame = secreted_df[["gene", "label"]].copy()
    method_results: list[dict] = []
    cv_scores_by_method: dict[str, np.ndarray] = {}

    for method_idx, spec in enumerate(METHOD_SPECS):
        cv_scores, fold_aurocs = compute_cv_scores(secreted_df, spec["kind"], spec["features"])
        bootstrap_ci, bootstrap_values = stratified_bootstrap_auc(
            labels=labels,
            scores=cv_scores,
            n_bootstrap=BOOTSTRAP_ITERATIONS,
            seed=RANDOM_SEED + (fold_spec["cutoff_year"] * 100) + method_idx,
        )
        cv_auc = float(roc_auc_score(labels, cv_scores))
        method_results.append(
            {
                "method": spec["name"],
                "label": spec["label"],
                "kind": spec["kind"],
                "feature_columns": spec["features"],
                "cv_auroc": cv_auc,
                "cv_fold_aurocs": fold_aurocs,
                "cv_fold_mean_auroc": float(np.mean(fold_aurocs)),
                "cv_fold_std_auroc": float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0,
                "bootstrap_ci": {
                    "lower": bootstrap_ci[0],
                    "upper": bootstrap_ci[1],
                },
                "bootstrap_summary": summarize_distribution(bootstrap_values),
            }
        )
        cv_scores_by_method[spec["name"]] = cv_scores
        prediction_frame[f"{spec['name']}_score"] = cv_scores

    return (
        {
            "fold_key": fold_spec["fold_key"],
            "fold_label": fold_spec["fold_label"],
            "cutoff_year": int(fold_spec["cutoff_year"]),
            "dataset_summary": dataset_summary,
            "methods": method_results,
            "predictions": prediction_frame.to_dict(orient="records"),
        },
        {
            "fold_key": fold_spec["fold_key"],
            "fold_label": fold_spec["fold_label"],
            "cutoff_year": int(fold_spec["cutoff_year"]),
            "labels": labels,
            "cv_scores_by_method": cv_scores_by_method,
        },
    )


def build_fold_method_lookup(fold_results: list[dict]) -> dict[str, dict[str, dict]]:
    return {
        fold["fold_key"]: {row["method"]: row for row in fold["methods"]}
        for fold in fold_results
    }


def aggregate_methods(fold_results: list[dict]) -> list[dict]:
    lookup = build_fold_method_lookup(fold_results)
    aggregated_rows: list[dict] = []

    for spec in METHOD_SPECS:
        fold_aurocs = np.asarray(
            [lookup[fold["fold_key"]][spec["name"]]["cv_auroc"] for fold in ROLLING_FOLDS],
            dtype=float,
        )
        aggregated_rows.append(
            {
                "method": spec["name"],
                "label": spec["label"],
                "per_fold_auroc": {
                    fold["fold_key"]: float(lookup[fold["fold_key"]][spec["name"]]["cv_auroc"])
                    for fold in ROLLING_FOLDS
                },
                "mean_auroc": float(np.mean(fold_aurocs)),
                "sd_auroc": float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0,
                "min_auroc": float(np.min(fold_aurocs)),
                "max_auroc": float(np.max(fold_aurocs)),
            }
        )

    return aggregated_rows


def compute_consistency_summary(fold_results: list[dict]) -> dict:
    lookup = build_fold_method_lookup(fold_results)
    comparisons_by_fold = {}
    all_better = True

    for fold in ROLLING_FOLDS:
        fold_key = fold["fold_key"]
        full_auc = float(lookup[fold_key]["full_model"]["cv_auroc"])
        fold_comparisons = {}
        for baseline_name in TRIVIAL_BASELINES:
            baseline_auc = float(lookup[fold_key][baseline_name]["cv_auroc"])
            better = full_auc > baseline_auc
            all_better = all_better and better
            fold_comparisons[baseline_name] = {
                "full_model_auroc": full_auc,
                "baseline_auroc": baseline_auc,
                "delta": float(full_auc - baseline_auc),
                "full_model_better": bool(better),
            }
        comparisons_by_fold[fold_key] = fold_comparisons

    summary_sentence = (
        "full_model consistently outperforms trivial baselines across all time windows"
        if all_better
        else "full_model does not consistently outperform every trivial baseline across all time windows"
    )
    return {
        "comparison_methods": TRIVIAL_BASELINES,
        "statement_supported": bool(all_better),
        "summary_sentence": summary_sentence,
        "comparisons_by_fold": comparisons_by_fold,
    }


def compute_pooled_paired_bootstrap(fold_internals: list[dict]) -> dict:
    labels = np.concatenate([fold["labels"] for fold in fold_internals])
    full_scores = np.concatenate([fold["cv_scores_by_method"]["full_model"] for fold in fold_internals])
    sig_scores = np.concatenate([fold["cv_scores_by_method"]["sig_max_auc"] for fold in fold_internals])
    delta_ci, delta_values = stratified_bootstrap_auc_delta(
        labels=labels,
        score_a=full_scores,
        score_b=sig_scores,
        n_bootstrap=BOOTSTRAP_ITERATIONS,
        seed=RANDOM_SEED + 50000,
    )

    observed_full_auc = float(roc_auc_score(labels, full_scores))
    observed_sig_auc = float(roc_auc_score(labels, sig_scores))
    return {
        "comparison": "full_model > sig_max_auc",
        "pooling_note": "Fold-specific gene instances are concatenated, then paired stratified bootstrap resampling is applied to the pooled labels and scores.",
        "n_folds": int(len(fold_internals)),
        "n_pooled_instances": int(len(labels)),
        "n_positive_instances": int(labels.sum()),
        "n_negative_instances": int(len(labels) - labels.sum()),
        "observed_full_model_auroc": observed_full_auc,
        "observed_sig_max_auc_auroc": observed_sig_auc,
        "observed_delta": float(observed_full_auc - observed_sig_auc),
        "bootstrap_ci": {
            "lower": float(delta_ci[0]),
            "upper": float(delta_ci[1]),
        },
        "bootstrap_summary": summarize_distribution(delta_values),
        "paired_p_value_one_sided": float(np.mean(delta_values <= 0.0)),
        "p_value_definition": "Fraction of pooled paired bootstrap iterations where AUROC(full_model) <= AUROC(sig_max_auc).",
    }


def create_plot(fold_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lookup = build_fold_method_lookup(fold_results)
    x = np.arange(len(METHOD_SPECS))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = ["#56B4E9", "#E69F00", "#009E73"]

    for fold_idx, fold in enumerate(ROLLING_FOLDS):
        fold_key = fold["fold_key"]
        rows = [lookup[fold_key][spec["name"]] for spec in METHOD_SPECS]
        aucs = [row["cv_auroc"] for row in rows]
        lower_errors = [max(row["cv_auroc"] - row["bootstrap_ci"]["lower"], 0.0) for row in rows]
        upper_errors = [max(row["bootstrap_ci"]["upper"] - row["cv_auroc"], 0.0) for row in rows]
        bars = ax.bar(
            x + offsets[fold_idx],
            aucs,
            width=width,
            yerr=np.vstack([lower_errors, upper_errors]),
            capsize=4,
            color=colors[fold_idx],
            edgecolor="#1f2933",
            linewidth=0.9,
            label=f"{fold['fold_label']} (<= {fold['cutoff_year']})",
        )
        for bar, auc in zip(bars, aucs, strict=True):
            ax.text(
                bar.get_x() + (bar.get_width() / 2),
                min(bar.get_height() + 0.025, 1.02),
                f"{auc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.2)
    ax.set_ylabel("AUROC (5-fold Stratified CV)")
    ax.set_title("Rolling-origin within-secreted NM evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels([spec["label"] for spec in METHOD_SPECS], rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables, gene_universe, pmid_year_map, pmid_conflicts = load_all_inputs()
    fold_results: list[dict] = []
    fold_internals: list[dict] = []

    for fold_spec in ROLLING_FOLDS:
        fold_result, fold_internal = evaluate_fold(
            fold_spec=fold_spec,
            tables=tables,
            gene_universe=gene_universe,
            pmid_year_map=pmid_year_map,
            pmid_conflicts=pmid_conflicts,
        )
        fold_results.append(fold_result)
        fold_internals.append(fold_internal)

    aggregated_methods = aggregate_methods(fold_results)
    consistency_summary = compute_consistency_summary(fold_results)
    pooled_paired_bootstrap = compute_pooled_paired_bootstrap(fold_internals)

    create_plot(fold_results)

    results = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "nm_corpus": NM_CORPUS,
        "rolling_folds": ROLLING_FOLDS,
        "method_order": [spec["name"] for spec in METHOD_SPECS],
        "folds": fold_results,
        "aggregated_methods": aggregated_methods,
        "consistency_summary": consistency_summary,
        "pooled_paired_bootstrap_full_model_vs_sig_max_auc": pooled_paired_bootstrap,
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    fold_lookup = build_fold_method_lookup(fold_results)

    print("NM Experiment P9: Rolling-origin within-secreted evaluation")
    print()
    for fold in ROLLING_FOLDS:
        fold_key = fold["fold_key"]
        dataset_summary = next(item["dataset_summary"] for item in fold_results if item["fold_key"] == fold_key)
        print(
            f"{fold['fold_label']} (<= {fold['cutoff_year']}): "
            f"{dataset_summary['n_secreted_subset']} secreted genes, "
            f"{dataset_summary['n_secreted_positives']} positives, "
            f"{dataset_summary['n_secreted_negatives']} negatives"
        )
        for spec in METHOD_SPECS:
            row = fold_lookup[fold_key][spec["name"]]
            print(
                f"  {spec['name']}: AUROC={row['cv_auroc']:.3f} "
                f"[{row['bootstrap_ci']['lower']:.3f}, {row['bootstrap_ci']['upper']:.3f}]"
            )
        print()

    print("Mean AUROC across folds:")
    for row in aggregated_methods:
        print(f"  {row['method']}: {row['mean_auroc']:.3f} +/- {row['sd_auroc']:.3f}")

    print()
    print(consistency_summary["summary_sentence"])
    print(
        "Pooled paired bootstrap p-value, full_model > sig_max_auc: "
        f"{pooled_paired_bootstrap['paired_p_value_one_sided']:.4f}"
    )
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
