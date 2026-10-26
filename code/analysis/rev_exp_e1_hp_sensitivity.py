#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from rev_exp_common import (
    BASE_DIR,
    FULL_FEATURES,
    RESULTS_DIR,
    build_external_fold_dataset,
    build_within_secreted_fold_c_frame,
    compute_rank_metrics_from_scores,
    cross_validated_scores,
    ensure_results_dir,
    full_fit_scores,
    make_fixed_xgb,
    make_nm_exp5_style_xgb,
    provenance,
    summarize_distribution,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
GRID = [
    {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
    for n_estimators, max_depth, learning_rate in itertools.product(
        [100, 200, 400],
        [2, 3, 4, 5, 6],
        [0.03, 0.05, 0.1, 0.2],
    )
]
MANUSCRIPT_CONFIG = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
PUBLISHED_WITHIN_SECRETED_AUROC = 0.759
PUBLISHED_EXTERNAL_FOLD_C_AUROC = 0.983


def config_key(config: dict) -> str:
    return f"n{config['n_estimators']}_d{config['max_depth']}_lr{config['learning_rate']}"


def builder_from_config(config: dict):
    return lambda _y=None: make_fixed_xgb(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
    )


def sort_within_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -row["auroc_oof"],
            -row["p_at_10"],
            -row["p_at_5"],
            row["n_estimators"],
            row["max_depth"],
            row["learning_rate"],
        ),
    )


def sort_external_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -row.get("auroc_mean", row.get("inner_mean_auroc")),
            -row.get("auroc_oof", row.get("inner_mean_auroc")),
            row["n_estimators"],
            row["max_depth"],
            row["learning_rate"],
        ),
    )


def add_rank(rows: list[dict], sorted_rows: list[dict], rank_field: str) -> dict[str, int]:
    rank_lookup = {config_key(row): idx + 1 for idx, row in enumerate(sorted_rows)}
    for row in rows:
        row[rank_field] = int(rank_lookup[config_key(row)])
    return rank_lookup


def describe_grid(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def evaluate_grid(secreted_df, external_df) -> tuple[list[dict], list[dict]]:
    within_rows: list[dict] = []
    external_rows: list[dict] = []
    for config in GRID:
        builder = builder_from_config(config)

        within_cv = cross_validated_scores(secreted_df, FULL_FEATURES, builder)
        within_fit_scores = full_fit_scores(secreted_df, FULL_FEATURES, builder)
        within_rank = compute_rank_metrics_from_scores(secreted_df, within_fit_scores, ks=(5, 10))
        within_rows.append(
            {
                **config,
                "config_key": config_key(config),
                "auroc_oof": within_cv["auroc_oof"],
                "auroc_mean": within_cv["auroc_mean"],
                "auroc_std": within_cv["auroc_std"],
                "p_at_5": within_rank["p_at_5"],
                "p_at_10": within_rank["p_at_10"],
                "top10_genes": within_rank["top_genes"][:10],
            }
        )

        external_cv = cross_validated_scores(external_df, FULL_FEATURES, builder)
        external_rows.append(
            {
                **config,
                "config_key": config_key(config),
                "auroc_oof": external_cv["auroc_oof"],
                "auroc_mean": external_cv["auroc_mean"],
                "auroc_std": external_cv["auroc_std"],
                "auprc_oof": external_cv["auprc_oof"],
            }
        )
    return within_rows, external_rows


def evaluate_fixed_config_nested_cv(eval_df) -> dict:
    X = eval_df[FULL_FEATURES].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = np.full(len(eval_df), np.nan, dtype=float)
    outer_rows: list[dict] = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + outer_idx)
        inner_rows: list[dict] = []

        for config in GRID:
            fold_aurocs: list[float] = []
            for inner_train_idx, inner_test_idx in inner.split(X_train, y_train):
                model = make_fixed_xgb(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    learning_rate=config["learning_rate"],
                )
                model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                inner_scores = model.predict_proba(X_train[inner_test_idx])[:, 1]
                fold_aurocs.append(float(roc_auc_score(y_train[inner_test_idx], inner_scores)))
            inner_rows.append(
                {
                    **config,
                    "config_key": config_key(config),
                    "inner_fold_aurocs": fold_aurocs,
                    "inner_mean_auroc": float(np.mean(fold_aurocs)),
                }
            )

        inner_rows = sort_external_rows(inner_rows)
        best = inner_rows[0]
        best_model = make_fixed_xgb(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
        )
        best_model.fit(X[train_idx], y[train_idx])
        outer_scores = best_model.predict_proba(X[test_idx])[:, 1]
        oof_scores[test_idx] = outer_scores
        outer_rows.append(
            {
                "outer_fold": outer_idx,
                "selected_config": {
                    "n_estimators": best["n_estimators"],
                    "max_depth": best["max_depth"],
                    "learning_rate": best["learning_rate"],
                    "config_key": best["config_key"],
                    "inner_mean_auroc": best["inner_mean_auroc"],
                },
                "inner_grid": inner_rows,
                "outer_fold_auroc": float(roc_auc_score(y[test_idx], outer_scores)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("Nested CV failed to assign every held-out score.")

    selected_counter = {}
    for row in outer_rows:
        key = row["selected_config"]["config_key"]
        selected_counter[key] = selected_counter.get(key, 0) + 1

    fold_aurocs = [row["outer_fold_auroc"] for row in outer_rows]
    return {
        "outer_folds": outer_rows,
        "selected_config_counts": selected_counter,
        "auroc_oof": float(roc_auc_score(y, oof_scores)),
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs, ddof=1)),
        "fold_aurocs": fold_aurocs,
    }


def run_reproducibility_checks(secreted_df, external_df) -> dict:
    checks = [
        {
            "name": "nm_exp4_checked_in_make_xgb",
            "builder": lambda _y=None: make_fixed_xgb(200, 4, 0.05),
            "notes": "learning_rate=0.05, no scale_pos_weight",
        },
        {
            "name": "lr_0.1_same_otherwise_as_nm_exp4",
            "builder": lambda _y=None: make_fixed_xgb(200, 4, 0.1),
            "notes": "learning_rate=0.1, no scale_pos_weight",
        },
        {
            "name": "nm_exp5_checked_in_make_xgb",
            "builder": make_nm_exp5_style_xgb,
            "notes": "learning_rate=0.1 with fold-specific scale_pos_weight",
        },
    ]

    result = {"within_secreted": [], "external_fold_c": []}
    for check in checks:
        within_cv = cross_validated_scores(secreted_df, FULL_FEATURES, check["builder"])
        external_cv = cross_validated_scores(external_df, FULL_FEATURES, check["builder"])
        result["within_secreted"].append(
            {
                "name": check["name"],
                "notes": check["notes"],
                "auroc_oof": within_cv["auroc_oof"],
                "abs_delta_to_published_0p759": abs(within_cv["auroc_oof"] - PUBLISHED_WITHIN_SECRETED_AUROC),
            }
        )
        result["external_fold_c"].append(
            {
                "name": check["name"],
                "notes": check["notes"],
                "auroc_mean": external_cv["auroc_mean"],
                "auroc_std": external_cv["auroc_std"],
                "auroc_oof": external_cv["auroc_oof"],
                "abs_delta_to_published_0p983": abs(external_cv["auroc_mean"] - PUBLISHED_EXTERNAL_FOLD_C_AUROC),
            }
        )

    result["within_secreted"] = sorted(result["within_secreted"], key=lambda row: row["abs_delta_to_published_0p759"])
    result["external_fold_c"] = sorted(result["external_fold_c"], key=lambda row: row["abs_delta_to_published_0p983"])
    result["headline_conclusion"] = {
        "within_secreted_closest_to_published_0p759": result["within_secreted"][0]["name"],
        "external_fold_c_closest_to_published_0p983": result["external_fold_c"][0]["name"],
    }
    return result


def main() -> None:
    start_time = time.time()
    ensure_results_dir()

    secreted_df, secreted_summary = build_within_secreted_fold_c_frame()
    external_df, external_summary, _ = build_external_fold_dataset("C")
    within_rows, external_rows = evaluate_grid(secreted_df, external_df)

    within_sorted = sort_within_rows(within_rows)
    external_sorted = sort_external_rows(external_rows)
    add_rank(within_rows, within_sorted, "rank_within_secreted")
    add_rank(external_rows, external_sorted, "rank_external_fold_c")

    within_manuscript_row = next(row for row in within_rows if row["config_key"] == config_key(MANUSCRIPT_CONFIG))
    external_manuscript_row = next(row for row in external_rows if row["config_key"] == config_key(MANUSCRIPT_CONFIG))

    fixed_external_cv = cross_validated_scores(external_df, FULL_FEATURES, builder_from_config(MANUSCRIPT_CONFIG))
    nested_external_cv = evaluate_fixed_config_nested_cv(external_df)
    reproducibility = run_reproducibility_checks(secreted_df, external_df)

    within_distribution = describe_grid(np.asarray([row["auroc_oof"] for row in within_rows], dtype=float))
    external_distribution = describe_grid(np.asarray([row["auroc_mean"] for row in external_rows], dtype=float))

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "grid": GRID,
                "manuscript_config_under_test": MANUSCRIPT_CONFIG,
                "within_secreted_metric_for_ranking": "auroc_oof",
                "external_fold_c_metric_for_ranking": "auroc_mean",
                "nested_cv_inner_splits": 4,
                "nested_cv_outer_splits": 5,
                "feature_columns_used": FULL_FEATURES,
                "feature_set_note": (
                    "The checked-in pipeline exposes 15 dynamic FEATURE_COLUMNS and this script adds is_secreted, "
                    "so the actual reviewer experiment runs 16 input columns rather than the manuscript's 10-feature description."
                ),
            },
            input_paths=[
                BASE_DIR / "nm_exp4_full_universe.py",
                BASE_DIR / "nm_exp5_llm_vs_ml_benchmark.py",
                BASE_DIR / "nm_exp_p6_secreted_ablation.py",
                BASE_DIR / "paper" / "supplementary_table_s1.tex",
            ],
        ),
        "outputs": {"json": str(OUTPUT_JSON)},
        "dataset_summaries": {
            "within_secreted": secreted_summary,
            "external_fold_c": external_summary,
        },
        "within_secreted_grid": {
            "rows": within_rows,
            "best_config": within_sorted[0],
            "manuscript_config": within_manuscript_row,
            "manuscript_config_rank": int(within_manuscript_row["rank_within_secreted"]),
            "distribution": within_distribution,
            "distribution_source_values": summarize_distribution(
                np.asarray([row["auroc_oof"] for row in within_rows], dtype=float)
            ),
        },
        "external_fold_c_grid": {
            "rows": external_rows,
            "best_config": external_sorted[0],
            "manuscript_config": external_manuscript_row,
            "manuscript_config_rank": int(external_manuscript_row["rank_external_fold_c"]),
            "distribution": external_distribution,
            "distribution_source_values": summarize_distribution(
                np.asarray([row["auroc_mean"] for row in external_rows], dtype=float)
            ),
        },
        "nested_model_selection_external_fold_c": {
            "fixed_checked_in_config": {
                **MANUSCRIPT_CONFIG,
                "auroc_mean": fixed_external_cv["auroc_mean"],
                "auroc_std": fixed_external_cv["auroc_std"],
                "auroc_oof": fixed_external_cv["auroc_oof"],
            },
            "nested_cv": nested_external_cv,
            "delta_nested_minus_fixed_auroc_mean": nested_external_cv["auroc_mean"] - fixed_external_cv["auroc_mean"],
            "delta_nested_minus_fixed_auroc_oof": nested_external_cv["auroc_oof"] - fixed_external_cv["auroc_oof"],
        },
        "published_headline_reproducibility_checks": reproducibility,
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, results)

    print(
        f"E1 hyperparameter sensitivity: within-secreted best AUROC={within_sorted[0]['auroc_oof']:.3f} "
        f"({within_sorted[0]['config_key']}), manuscript config rank={within_manuscript_row['rank_within_secreted']}/60."
    )
    print(
        f"External Fold C best mean AUROC={external_sorted[0]['auroc_mean']:.3f}±{external_sorted[0]['auroc_std']:.3f} "
        f"({external_sorted[0]['config_key']}), manuscript config rank={external_manuscript_row['rank_external_fold_c']}/60."
    )
    print(
        f"Nested vs fixed external Fold C: nested={nested_external_cv['auroc_mean']:.3f}±{nested_external_cv['auroc_std']:.3f}, "
        f"fixed={fixed_external_cv['auroc_mean']:.3f}±{fixed_external_cv['auroc_std']:.3f}."
    )
    print(
        f"Closest to published 0.759: {reproducibility['headline_conclusion']['within_secreted_closest_to_published_0p759']}; "
        f"closest to published 0.983: {reproducibility['headline_conclusion']['external_fold_c_closest_to_published_0p983']}."
    )
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
