#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rev_exp_common import (
    BASE_DIR,
    FEATURE_LAYER_ORDER,
    FULL_FEATURES,
    RESULTS_DIR,
    annotate_tables_with_study_metadata,
    build_external_fold_dataset,
    build_feature_context,
    build_within_secreted_fold_c_frame,
    compute_rank_metrics_from_scores,
    cross_validated_scores,
    ensure_results_dir,
    full_fit_scores,
    make_fixed_xgb,
    provenance,
    subset_feature_tables_by_studies,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_REPLICATES = 20


def builder(_y=None):
    return make_fixed_xgb(200, 4, 0.05)


def paired_t_test(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    deltas = values_b - values_a
    t_stat, p_value = stats.ttest_rel(values_b, values_a)
    mean_delta = float(np.mean(deltas))
    if len(deltas) > 1:
        sem = float(stats.sem(deltas))
        ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean_delta, scale=sem)
        ci95 = {"lower": float(ci[0]), "upper": float(ci[1])}
    else:
        ci95 = {"lower": mean_delta, "upper": mean_delta}
    return {
        "mean_delta_b_minus_a": mean_delta,
        "t_statistic": float(t_stat),
        "p_value_two_sided": float(p_value),
        "ci95": ci95,
    }


def summarize_fraction_rows(rows: list[dict], metric_key: str) -> dict:
    values = np.asarray([row[metric_key] for row in rows], dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    start_time = time.time()
    ensure_results_dir()

    full_feature_context = build_feature_context(2021)
    full_gene_universe = full_feature_context["gene_universe"]
    annotated_tables, _, _ = annotate_tables_with_study_metadata(full_feature_context["tables"])
    feature_study_ids = sorted(
        set().union(*(set(annotated_tables[layer]["_study_id"]) for layer in FEATURE_LAYER_ORDER))
    )

    secreted_full_df, secreted_summary = build_within_secreted_fold_c_frame()
    secreted_gene_order = secreted_full_df["gene"].tolist()
    secreted_label_map = dict(zip(secreted_full_df["gene"], secreted_full_df["label"], strict=True))

    external_full_df, external_summary, _ = build_external_fold_dataset("C")
    external_gene_order = external_full_df["gene"].tolist()
    external_label_map = dict(zip(external_full_df["gene"], external_full_df["label"], strict=True))

    all_fraction_results: list[dict] = []
    fraction_aggregates = {}

    for fraction_idx, fraction in enumerate(FRACTIONS):
        fraction_rows: list[dict] = []
        n_to_sample = int(round(len(feature_study_ids) * fraction))
        if fraction == 1.0:
            n_to_sample = len(feature_study_ids)
        n_to_sample = max(1, n_to_sample)

        for replicate_idx in range(N_REPLICATES):
            rng = np.random.default_rng(42 + fraction_idx * 1000 + replicate_idx)
            if n_to_sample == len(feature_study_ids):
                sampled = feature_study_ids
            else:
                sampled = sorted(rng.choice(feature_study_ids, size=n_to_sample, replace=False).tolist())
            sampled_set = set(sampled)

            subset_tables = subset_feature_tables_by_studies(annotated_tables, sampled_set)
            subset_feature_context = build_feature_context(
                cutoff_year=2021,
                tables=subset_tables,
                gene_universe=full_gene_universe,
            )
            feature_df = subset_feature_context["feature_df"]

            secreted_eval = feature_df.loc[feature_df["gene"].isin(secreted_gene_order)].copy()
            secreted_eval["label"] = secreted_eval["gene"].map(secreted_label_map).astype(int)
            secreted_eval = secreted_eval.set_index("gene").loc[secreted_gene_order].reset_index()
            secreted_cv = cross_validated_scores(secreted_eval, FULL_FEATURES, builder)
            secreted_fit_scores = full_fit_scores(secreted_eval, FULL_FEATURES, builder)
            secreted_rank = compute_rank_metrics_from_scores(secreted_eval, secreted_fit_scores, ks=(10,))

            external_eval = feature_df.loc[feature_df["gene"].isin(external_gene_order)].copy()
            external_eval["label"] = external_eval["gene"].map(external_label_map).astype(int)
            external_eval = external_eval.set_index("gene").loc[external_gene_order].reset_index()
            external_cv = cross_validated_scores(external_eval, FULL_FEATURES, builder)

            retained_rows = {layer: int(len(subset_tables[layer])) for layer in FEATURE_LAYER_ORDER}
            fraction_rows.append(
                {
                    "fraction": fraction,
                    "replicate_index": replicate_idx,
                    "replicate_seed": 42 + fraction_idx * 1000 + replicate_idx,
                    "n_sampled_studies": int(len(sampled)),
                    "sampled_studies_preview": sampled[:20],
                    "retained_raw_rows_feature_layers": int(sum(retained_rows.values())),
                    "retained_raw_rows_by_layer": retained_rows,
                    "retained_genes_with_cutoff_evidence": int(feature_df["has_evidence_by_cutoff"].sum()),
                    "retained_genes_in_full_universe_with_any_nonzero_feature": int(
                        (feature_df[FULL_FEATURES].sum(axis=1) > 0).sum()
                    ),
                    "within_secreted_auroc": secreted_cv["auroc_oof"],
                    "within_secreted_p_at_10": secreted_rank["p_at_10"],
                    "external_fold_c_auroc": external_cv["auroc_oof"],
                }
            )

        all_fraction_results.extend(fraction_rows)
        fraction_aggregates[str(fraction)] = {
            "n_replicates": N_REPLICATES,
            "study_count": n_to_sample,
            "within_secreted_auroc": summarize_fraction_rows(fraction_rows, "within_secreted_auroc"),
            "within_secreted_p_at_10": summarize_fraction_rows(fraction_rows, "within_secreted_p_at_10"),
            "external_fold_c_auroc": summarize_fraction_rows(fraction_rows, "external_fold_c_auroc"),
            "retained_raw_rows_feature_layers": summarize_fraction_rows(fraction_rows, "retained_raw_rows_feature_layers"),
            "retained_genes_with_cutoff_evidence": summarize_fraction_rows(
                fraction_rows, "retained_genes_with_cutoff_evidence"
            ),
        }

    fraction_lookup: dict[float, list[dict]] = {}
    for row in all_fraction_results:
        fraction_lookup.setdefault(row["fraction"], []).append(row)

    rows_08 = fraction_lookup[0.8]
    rows_10 = fraction_lookup[1.0]
    plateau_tests = {
        "within_secreted_auroc": paired_t_test(
            np.asarray([row["within_secreted_auroc"] for row in rows_08], dtype=float),
            np.asarray([row["within_secreted_auroc"] for row in rows_10], dtype=float),
        ),
        "within_secreted_p_at_10": paired_t_test(
            np.asarray([row["within_secreted_p_at_10"] for row in rows_08], dtype=float),
            np.asarray([row["within_secreted_p_at_10"] for row in rows_10], dtype=float),
        ),
        "external_fold_c_auroc": paired_t_test(
            np.asarray([row["external_fold_c_auroc"] for row in rows_08], dtype=float),
            np.asarray([row["external_fold_c_auroc"] for row in rows_10], dtype=float),
        ),
    }

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "fractions": FRACTIONS,
                "n_replicates_per_fraction": N_REPLICATES,
                "sampling_unit": "canonical studies across the six feature-generating ABOR layers used by build_features_and_evidence",
                "fixed_secreted_subset_rule": (
                    "The 35-gene within-secreted benchmark was defined once from the full corpus and held fixed across replicates; "
                    "genes that lost all evidence under subsampling were retained with zero features."
                ),
                "fixed_external_eval_rule": (
                    "The full-corpus External Fold C candidate set was held fixed across replicates; "
                    "genes that lost all evidence under subsampling were retained with zero features."
                ),
                "feature_columns_used": FULL_FEATURES,
            },
            input_paths=[
                BASE_DIR / "nm_exp4_full_universe.py",
                BASE_DIR / "nm_exp_p6_secreted_ablation.py",
                BASE_DIR / "paper" / "supplementary_table_s1.tex",
            ],
        ),
        "outputs": {"json": str(OUTPUT_JSON)},
        "dataset_summaries": {
            "within_secreted_full_corpus": secreted_summary,
            "external_fold_c_full_corpus": external_summary,
            "n_feature_studies_available": int(len(feature_study_ids)),
        },
        "replicates": all_fraction_results,
        "fraction_aggregates": fraction_aggregates,
        "plateau_tests_0p8_vs_1p0": plateau_tests,
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, results)

    print(
        f"E2 corpus saturation: {len(feature_study_ids)} feature-layer studies, "
        f"0.8→1.0 within-secreted AUROC delta={plateau_tests['within_secreted_auroc']['mean_delta_b_minus_a']:+.3f} "
        f"(p={plateau_tests['within_secreted_auroc']['p_value_two_sided']:.3g})."
    )
    print(
        f"0.8→1.0 external Fold C AUROC delta={plateau_tests['external_fold_c_auroc']['mean_delta_b_minus_a']:+.3f} "
        f"(p={plateau_tests['external_fold_c_auroc']['p_value_two_sided']:.3g})."
    )
    print(
        f"At fraction 1.0: within-secreted AUROC={fraction_aggregates['1.0']['within_secreted_auroc']['mean']:.3f}±"
        f"{fraction_aggregates['1.0']['within_secreted_auroc']['std']:.3f}, "
        f"external Fold C AUROC={fraction_aggregates['1.0']['external_fold_c_auroc']['mean']:.3f}±"
        f"{fraction_aggregates['1.0']['external_fold_c_auroc']['std']:.3f}."
    )
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
