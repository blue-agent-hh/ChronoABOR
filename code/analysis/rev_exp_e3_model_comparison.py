#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rev_exp_common import (
    BASE_DIR,
    FULL_FEATURES,
    RESULTS_DIR,
    build_external_fold_dataset,
    build_within_secreted_fold_c_frame,
    compute_rank_metrics_from_scores,
    cross_validated_scores,
    delong_test,
    ensure_results_dir,
    full_fit_scores,
    make_elastic_net_pipeline,
    make_fixed_xgb,
    make_l2_logistic,
    paired_stratified_bootstrap_auc_delta,
    provenance,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
MODEL_SPECS = [
    {
        "name": "xgboost_checked_in",
        "label": "XGBoost",
        "builder": lambda _y=None: make_fixed_xgb(200, 4, 0.05),
        "notes": "Checked-in nm_exp4_full_universe make_xgb configuration.",
    },
    {
        "name": "logistic_l2",
        "label": "Logistic regression (L2)",
        "builder": make_l2_logistic,
        "notes": "Standardized features, L2 penalty, C=1.0.",
    },
    {
        "name": "elastic_net",
        "label": "Elastic net",
        "builder": make_elastic_net_pipeline,
        "notes": "Reused nm_exp5_llm_vs_ml_benchmark elastic-net pipeline.",
    },
]


def evaluate_endpoint(
    endpoint_name: str,
    eval_df,
    ranking_scores_source: str,
    seed_offset: int,
    include_delong: bool,
) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    model_results: list[dict] = []
    cv_scores_by_model: dict[str, np.ndarray] = {}
    rank_metrics_by_model: dict[str, dict] = {}

    for spec in MODEL_SPECS:
        cv = cross_validated_scores(eval_df, FULL_FEATURES, spec["builder"])
        if ranking_scores_source == "full_fit":
            ranking_scores = full_fit_scores(eval_df, FULL_FEATURES, spec["builder"])
        elif ranking_scores_source == "oof":
            ranking_scores = cv["oof_scores"]
        else:
            raise ValueError(f"Unsupported ranking score source: {ranking_scores_source}")
        rank_metrics = compute_rank_metrics_from_scores(eval_df, ranking_scores, ks=(5, 10, 50))

        model_results.append(
            {
                "model": spec["name"],
                "label": spec["label"],
                "notes": spec["notes"],
                "auroc_oof": cv["auroc_oof"],
                "auroc_mean": cv["auroc_mean"],
                "auroc_std": cv["auroc_std"],
                "auprc_oof": cv["auprc_oof"],
                "fold_aurocs": cv["fold_aurocs"],
                "ranking_score_source": ranking_scores_source,
                "ranking_metrics": rank_metrics,
            }
        )
        cv_scores_by_model[spec["name"]] = cv["oof_scores"]
        rank_metrics_by_model[spec["name"]] = rank_metrics

    xgb_scores = cv_scores_by_model["xgboost_checked_in"]
    xgb_rank = rank_metrics_by_model["xgboost_checked_in"]
    pairwise = {}
    for pair_idx, baseline_name in enumerate(["logistic_l2", "elastic_net"]):
        delta = paired_stratified_bootstrap_auc_delta(
            labels,
            xgb_scores,
            cv_scores_by_model[baseline_name],
            n_bootstrap=1000,
            seed=42 + seed_offset + pair_idx,
        )
        row = {
            "comparison": f"xgboost_checked_in_vs_{baseline_name}",
            "delta_auroc_bootstrap": {
                "observed_delta": delta["observed_delta"],
                "ci95": delta["ci95"],
                "p_value_two_sided": delta["p_value_two_sided"],
            },
            "delta_p_at_5": xgb_rank.get("p_at_5", float("nan")) - rank_metrics_by_model[baseline_name].get("p_at_5", float("nan")),
            "delta_p_at_10": xgb_rank.get("p_at_10", float("nan")) - rank_metrics_by_model[baseline_name].get("p_at_10", float("nan")),
            "delta_ef_at_10": xgb_rank.get("ef_at_10", float("nan")) - rank_metrics_by_model[baseline_name].get("ef_at_10", float("nan")),
            "statistically_distinguishable_by_bootstrap_95ci": not (
                delta["ci95"]["lower"] <= 0.0 <= delta["ci95"]["upper"]
            ),
        }
        if include_delong:
            delong = delong_test(labels, xgb_scores, cv_scores_by_model[baseline_name])
            row["delong"] = delong
            row["statistically_distinguishable_by_delong_p_lt_0p05"] = delong["p_value_two_sided"] < 0.05
        pairwise[baseline_name] = row

    conclusion_bits = []
    for baseline_name in ["logistic_l2", "elastic_net"]:
        pair = pairwise[baseline_name]
        xgb_better_on_auroc = pair["delta_auroc_bootstrap"]["observed_delta"] > 0
        distinguishable = pair["statistically_distinguishable_by_bootstrap_95ci"]
        if include_delong:
            distinguishable = distinguishable and pair["statistically_distinguishable_by_delong_p_lt_0p05"]
        if distinguishable:
            conclusion_bits.append(
                f"XGBoost {'exceeds' if xgb_better_on_auroc else 'trails'} {baseline_name} on AUROC"
            )
        else:
            conclusion_bits.append(f"XGBoost is not statistically distinguishable from {baseline_name} on AUROC")

    return {
        "endpoint_name": endpoint_name,
        "n_samples": int(len(eval_df)),
        "n_positives": int(labels.sum()),
        "n_negatives": int(len(labels) - labels.sum()),
        "feature_columns_used": FULL_FEATURES,
        "models": model_results,
        "pairwise_xgb_vs_linear": pairwise,
        "honest_conclusion": "; ".join(conclusion_bits),
    }


def main() -> None:
    start_time = time.time()
    ensure_results_dir()

    fold_c_df, fold_c_summary, _ = build_external_fold_dataset("C")
    fold_d_df, fold_d_summary, _ = build_external_fold_dataset("D")
    secreted_df, secreted_summary = build_within_secreted_fold_c_frame()

    fold_c = evaluate_endpoint(
        endpoint_name="external_fold_c",
        eval_df=fold_c_df,
        ranking_scores_source="oof",
        seed_offset=100,
        include_delong=True,
    )
    fold_d = evaluate_endpoint(
        endpoint_name="external_fold_d",
        eval_df=fold_d_df,
        ranking_scores_source="oof",
        seed_offset=200,
        include_delong=True,
    )
    within = evaluate_endpoint(
        endpoint_name="within_secreted_fold_c",
        eval_df=secreted_df,
        ranking_scores_source="full_fit",
        seed_offset=300,
        include_delong=True,
    )

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "feature_columns_used": FULL_FEATURES,
                "feature_set_note": (
                    "The checked-in pipeline exposes 15 dynamic FEATURE_COLUMNS and this script adds is_secreted, "
                    "so the actual comparison uses 16 input columns rather than the manuscript's 10-feature external model description."
                ),
                "bootstrap_iterations": 1000,
                "external_ranking_score_source": "oof",
                "within_secreted_ranking_score_source": "full_fit to match the existing p6 early-precision path",
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
            "external_fold_c": fold_c_summary,
            "external_fold_d": fold_d_summary,
            "within_secreted": secreted_summary,
        },
        "endpoints": {
            "external_fold_c": fold_c,
            "external_fold_d": fold_d,
            "within_secreted": within,
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, results)

    fold_c_xgb = next(row for row in fold_c["models"] if row["model"] == "xgboost_checked_in")
    fold_d_xgb = next(row for row in fold_d["models"] if row["model"] == "xgboost_checked_in")
    within_xgb = next(row for row in within["models"] if row["model"] == "xgboost_checked_in")

    print(
        f"E3 model comparison: Fold C XGB AUROC={fold_c_xgb['auroc_mean']:.3f}±{fold_c_xgb['auroc_std']:.3f}; "
        f"Fold D XGB AUROC={fold_d_xgb['auroc_mean']:.3f}±{fold_d_xgb['auroc_std']:.3f}."
    )
    print(
        f"Within-secreted XGB AUROC={within_xgb['auroc_oof']:.3f}, "
        f"P@5={within_xgb['ranking_metrics']['p_at_5']:.3f}, "
        f"P@10={within_xgb['ranking_metrics']['p_at_10']:.3f}, "
        f"EF@10={within_xgb['ranking_metrics']['ef_at_10']:.3f}."
    )
    print(f"Fold C conclusion: {fold_c['honest_conclusion']}.")
    print(f"Fold D conclusion: {fold_d['honest_conclusion']}.")
    print(f"Within-secreted conclusion: {within['honest_conclusion']}.")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
