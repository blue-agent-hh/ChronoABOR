#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    make_xgb,
    to_jsonable,
)
from nm_exp_p1_prior_control import PRIOR_FEATURES, SECRETED_GENES, TEMPORAL_FEATURES

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
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p6_secreted_ablation.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p6_secreted_ablation.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_p6_secreted_ablation.pdf"
TARGET_CSV = BASE_DIR / "target.csv"

FEATURE_CUTOFF_YEAR = 2021
N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000
TOP_K_VALUES = [5, 10, 20]

NM_CORPUS_REQUESTED = [
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

FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]
METHOD_SPECS = [
    {
        "name": "is_secreted",
        "label": "is_secreted",
        "kind": "constant",
        "features": ["is_secreted"],
        "ranking_note": "Within the secreted subset every score is identical, so P@K is reported as the random-tie expectation.",
    },
    {
        "name": "sig_max_auc",
        "label": "sig_max_auc",
        "kind": "single",
        "features": ["sig_max_auc"],
    },
    {
        "name": "num_studies",
        "label": "num_studies",
        "kind": "single",
        "features": ["num_studies"],
    },
    {
        "name": "layer_count",
        "label": "layer_count",
        "kind": "single",
        "features": ["layer_count"],
    },
    {
        "name": "prior_only",
        "label": "prior_only",
        "kind": "model",
        "features": PRIOR_FEATURES,
    },
    {
        "name": "temporal_only",
        "label": "temporal_only",
        "kind": "model",
        "features": TEMPORAL_FEATURES,
    },
    {
        "name": "full_model",
        "label": "full_model",
        "kind": "model",
        "features": FULL_FEATURES,
    },
]


def load_all_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")
    target_df = pd.read_csv(TARGET_CSV)
    return tables, target_df


def build_fold_c_frame() -> tuple[pd.DataFrame, dict]:
    tables, target_df = load_all_inputs()
    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )

    nm_requested_set = set(NM_CORPUS_REQUESTED)
    universe_set = set(gene_universe)
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)
    feature_df["label"] = feature_df["gene"].isin(nm_requested_set).astype(int)

    secreted_df = feature_df.loc[feature_df["is_secreted"] == 1].copy().reset_index(drop=True)
    full_summary = {
        "n_genes_universe": int(len(feature_df)),
        "n_genes_with_2021_evidence": int(len(evidence_by_cutoff)),
        "n_nm_requested": int(len(NM_CORPUS_REQUESTED)),
        "n_nm_in_universe": int(feature_df["label"].sum()),
        "nm_genes_in_universe": sorted(universe_set & nm_requested_set),
        "nm_genes_missing_from_universe": sorted(nm_requested_set - universe_set),
        "n_secreted_subset": int(len(secreted_df)),
        "n_secreted_positives": int(secreted_df["label"].sum()),
        "n_secreted_negatives": int(len(secreted_df) - secreted_df["label"].sum()),
        "secreted_positive_genes": sorted(secreted_df.loc[secreted_df["label"] == 1, "gene"].tolist()),
        "secreted_negative_genes": sorted(secreted_df.loc[secreted_df["label"] == 0, "gene"].tolist()),
        "nm_positive_non_secreted_genes": sorted(
            feature_df.loc[(feature_df["label"] == 1) & (feature_df["is_secreted"] == 0), "gene"].tolist()
        ),
        "target_csv_rows": int(len(target_df)),
        "target_csv_columns": list(target_df.columns),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
    }
    return secreted_df, full_summary


def compute_cv_scores(
    eval_df: pd.DataFrame,
    kind: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[float]]:
    y = eval_df["label"].to_numpy(dtype=int)
    X = eval_df[feature_columns].to_numpy(dtype=float) if feature_columns else np.empty((len(eval_df), 0), dtype=float)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_scores = np.full(len(eval_df), np.nan, dtype=float)
    fold_aurocs: list[float] = []

    for train_idx, test_idx in splitter.split(np.zeros(len(eval_df)), y):
        if kind == "constant":
            test_scores = np.ones(len(test_idx), dtype=float)
        elif kind == "single":
            test_scores = X[test_idx, 0]
        else:
            model = make_xgb()
            model.fit(X[train_idx], y[train_idx])
            test_scores = model.predict_proba(X[test_idx])[:, 1]
        oof_scores[test_idx] = test_scores
        fold_aurocs.append(float(roc_auc_score(y[test_idx], test_scores)))

    if np.isnan(oof_scores).any():
        raise RuntimeError("Failed to populate every held-out score.")
    return oof_scores, fold_aurocs


def compute_full_fit_scores(
    eval_df: pd.DataFrame,
    kind: str,
    feature_columns: list[str],
) -> np.ndarray:
    if kind == "constant":
        return np.ones(len(eval_df), dtype=float)
    if kind == "single":
        return eval_df[feature_columns[0]].to_numpy(dtype=float)

    model = make_xgb()
    X = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def stratified_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[tuple[float, float], np.ndarray]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        resampled_pos = rng.choice(positives, size=len(positives), replace=True)
        resampled_neg = rng.choice(negatives, size=len(negatives), replace=True)
        resampled_idx = np.concatenate([resampled_pos, resampled_neg])
        aucs[idx] = float(roc_auc_score(labels[resampled_idx], scores[resampled_idx]))

    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
    ), aucs


def stratified_bootstrap_auc_delta(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[tuple[float, float], np.ndarray]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        resampled_pos = rng.choice(positives, size=len(positives), replace=True)
        resampled_neg = rng.choice(negatives, size=len(negatives), replace=True)
        resampled_idx = np.concatenate([resampled_pos, resampled_neg])
        auc_a = roc_auc_score(labels[resampled_idx], score_a[resampled_idx])
        auc_b = roc_auc_score(labels[resampled_idx], score_b[resampled_idx])
        deltas[idx] = float(auc_a - auc_b)

    return (
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    ), deltas


def rank_frame(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    ranked_df = eval_df[["gene", "label", "num_studies"]].copy()
    ranked_df["score"] = scores
    return ranked_df.sort_values(
        by=["score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_rank_metrics(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    kind: str,
) -> dict:
    prevalence = float(eval_df["label"].mean())
    if kind == "constant":
        metrics = {
            "ranking_mode": "random_tie_expectation",
            "base_rate_within_secreted_subset": prevalence,
            "top_genes": None,
        }
        for k in TOP_K_VALUES:
            precision = prevalence
            ef = 1.0 if prevalence > 0 else float("nan")
            metrics[f"p_at_{k}"] = precision
            metrics[f"ef_at_{k}"] = ef
            metrics[f"expected_hits_at_{k}"] = float(prevalence * k)
        return metrics

    ranked_df = rank_frame(eval_df, scores)
    metrics = {
        "ranking_mode": "score_desc_num_studies_desc_gene_asc",
        "base_rate_within_secreted_subset": prevalence,
        "top_genes": ranked_df.head(max(TOP_K_VALUES))["gene"].tolist(),
    }
    for k in TOP_K_VALUES:
        top_df = ranked_df.head(k)
        hits = int(top_df["label"].sum())
        precision = float(hits / k)
        ef = float(precision / prevalence) if prevalence > 0 else float("nan")
        metrics[f"p_at_{k}"] = precision
        metrics[f"ef_at_{k}"] = ef
        metrics[f"hits_at_{k}"] = hits
    return metrics


def summarize_distribution(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
    }


def plot_aurocs(method_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    labels = [row["label"] for row in method_results]
    aucs = [row["cv_auroc"] for row in method_results]
    lower_errors = [row["cv_auroc"] - row["bootstrap_ci"]["lower"] for row in method_results]
    upper_errors = [row["bootstrap_ci"]["upper"] - row["cv_auroc"] for row in method_results]
    colors = ["#999999", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#0072B2", "#CC79A7"]

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        aucs,
        yerr=np.vstack([lower_errors, upper_errors]),
        capsize=5,
        color=colors[: len(labels)],
        edgecolor="#1f2933",
        linewidth=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Secreted-subset AUROC (5-fold CV)")
    ax.set_title("Fold C secreted-gene ablation against the external NM corpus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, method_results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(row["bootstrap_ci"]["upper"] + 0.03, 1.02),
            f"{row['cv_auroc']:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    secreted_df, dataset_summary = build_fold_c_frame()
    if secreted_df["label"].nunique() < 2:
        raise RuntimeError("Secreted subset must contain both positives and negatives.")

    labels = secreted_df["label"].to_numpy(dtype=int)
    method_results: list[dict] = []
    cv_scores_by_method: dict[str, np.ndarray] = {}

    for method_idx, spec in enumerate(METHOD_SPECS):
        cv_scores, fold_aurocs = compute_cv_scores(secreted_df, spec["kind"], spec["features"])
        fit_scores = compute_full_fit_scores(secreted_df, spec["kind"], spec["features"])
        bootstrap_ci, bootstrap_values = stratified_bootstrap_auc(
            labels=labels,
            scores=cv_scores,
            seed=RANDOM_SEED + method_idx,
        )
        cv_auc = float(roc_auc_score(labels, cv_scores))
        rank_metrics = compute_rank_metrics(secreted_df, fit_scores, spec["kind"])

        row = {
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
            "ranking_metrics": rank_metrics,
        }
        if "ranking_note" in spec:
            row["ranking_note"] = spec["ranking_note"]

        method_results.append(row)
        cv_scores_by_method[spec["name"]] = cv_scores

    full_model_scores = cv_scores_by_method["full_model"]
    full_model_auc = next(row["cv_auroc"] for row in method_results if row["method"] == "full_model")
    full_vs_baselines = {}
    for spec in METHOD_SPECS:
        if spec["name"] == "full_model":
            continue
        delta = float(full_model_auc - next(row["cv_auroc"] for row in method_results if row["method"] == spec["name"]))
        delta_ci, delta_values = stratified_bootstrap_auc_delta(
            labels=labels,
            score_a=full_model_scores,
            score_b=cv_scores_by_method[spec["name"]],
            seed=RANDOM_SEED + 100 + len(full_vs_baselines),
        )
        full_vs_baselines[spec["name"]] = {
            "observed_auroc_delta": delta,
            "bootstrap_ci": {
                "lower": delta_ci[0],
                "upper": delta_ci[1],
            },
            "bootstrap_summary": summarize_distribution(delta_values),
        }

    plot_aurocs(method_results)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "n_splits": N_SPLITS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "nm_corpus_requested": NM_CORPUS_REQUESTED,
        "dataset_summary": dataset_summary,
        "method_order": [spec["name"] for spec in METHOD_SPECS],
        "methods": method_results,
        "full_model_vs_baselines": full_vs_baselines,
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("NM Experiment P6: Secreted-gene ablation")
    print(
        f"Secreted subset: {dataset_summary['n_secreted_subset']} genes "
        f"({dataset_summary['n_secreted_positives']} positives, {dataset_summary['n_secreted_negatives']} negatives)"
    )
    print(
        f"NM genes requested / in universe: {dataset_summary['n_nm_requested']} / {dataset_summary['n_nm_in_universe']}"
    )
    print()
    for row in method_results:
        print(
            f"{row['method']}: AUROC={row['cv_auroc']:.3f} "
            f"[{row['bootstrap_ci']['lower']:.3f}, {row['bootstrap_ci']['upper']:.3f}]"
        )
        print(
            f"  P@5={row['ranking_metrics']['p_at_5']:.3f}, "
            f"P@10={row['ranking_metrics']['p_at_10']:.3f}, "
            f"P@20={row['ranking_metrics']['p_at_20']:.3f}"
        )
        print(
            f"  EF@5={row['ranking_metrics']['ef_at_5']:.3f}, "
            f"EF@10={row['ranking_metrics']['ef_at_10']:.3f}"
        )
    print()
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
