#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_common as rc

SCRIPT_NAME = Path(__file__).name
RESULTS_DIR = rc.BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e19_no_prior_ablation.json"
OLD_E12_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"
ANNOTATION_BUNDLE_PATH = RESULTS_DIR / "rev_exp_annotations.json"

BENCHMARK_KEYS = ("strict_fold_c", "strict_fold_d", "within_secreted_strict")
MODEL_KEYS = ("xgb", "logistic")
NO_PRIOR_FEATURES = list(rc.FEATURE_COLUMNS)
EXPECTED_TEMPORAL_FEATURE_COUNT = 15


def load_reference_e12_results() -> dict:
    if not OLD_E12_PATH.exists():
        raise FileNotFoundError(f"Missing reference E12 results: {OLD_E12_PATH}")
    payload = json.loads(OLD_E12_PATH.read_text())
    missing = [key for key in BENCHMARK_KEYS if key not in payload.get("benchmarks", {})]
    if missing:
        raise RuntimeError(f"Reference E12 results are missing benchmarks: {missing}")
    return payload


def load_cached_annotation_bundle() -> tuple[dict, set[str], Path]:
    if not ANNOTATION_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing cached annotation bundle: {ANNOTATION_BUNDLE_PATH}")
    payload = json.loads(ANNOTATION_BUNDLE_PATH.read_text())
    provenance = payload.get("provenance", {})
    if provenance.get("script_name") != "rev_exp_annotations.py":
        raise RuntimeError(f"Unexpected annotation provenance in {ANNOTATION_BUNDLE_PATH}: {provenance}")
    bundle = payload.get("annotation_bundle")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"annotation_bundle missing from {ANNOTATION_BUNDLE_PATH}")
    kw_genes = set(bundle["uniprot"]["secreted_kw_genes"])
    return bundle, kw_genes, ANNOTATION_BUNDLE_PATH


def apply_no_prior_feature_patch():
    import rev_exp_e9_nested_tuning as e9
    import rev_exp_e12_strict_windows as e12

    if len(NO_PRIOR_FEATURES) != EXPECTED_TEMPORAL_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TEMPORAL_FEATURE_COUNT} temporal features in rev_exp_common.FEATURE_COLUMNS, "
            f"found {len(NO_PRIOR_FEATURES)}."
        )
    if "is_secreted" in NO_PRIOR_FEATURES:
        raise RuntimeError("rev_exp_common.FEATURE_COLUMNS unexpectedly contains is_secreted.")

    e9.FULL_FEATURES = list(NO_PRIOR_FEATURES)
    e12.FULL_FEATURES = list(NO_PRIOR_FEATURES)

    if "is_secreted" in e9.FULL_FEATURES:
        raise RuntimeError("is_secreted is still present in rev_exp_e9_nested_tuning.FULL_FEATURES after rebinding.")
    if "is_secreted" in e12.FULL_FEATURES:
        raise RuntimeError("is_secreted is still present in rev_exp_e12_strict_windows.FULL_FEATURES after rebinding.")

    audit = {
        "rev_exp_common_feature_columns": list(NO_PRIOR_FEATURES),
        "rev_exp_common_full_features_unchanged": list(rc.FULL_FEATURES),
        "rev_exp_e9_full_features_after_rebinding": list(e9.FULL_FEATURES),
        "rev_exp_e12_full_features_after_rebinding": list(e12.FULL_FEATURES),
    }
    return e9, e12, audit


class FeatureUsageCaptured(RuntimeError):
    def __init__(self, consumer: str, columns: list[str]):
        self.consumer = consumer
        self.columns = list(columns)
        super().__init__(f"{consumer} consumed columns: {columns}")


class AuditedFeatureSlice(pd.DataFrame):
    _metadata = ["_audit_consumer"]

    @property
    def _constructor(self):
        return AuditedFeatureSlice

    def to_numpy(self, *args, **kwargs):
        raise FeatureUsageCaptured(getattr(self, "_audit_consumer", "unknown"), list(self.columns))


class AuditedEvalFrame(pd.DataFrame):
    _metadata = ["_audit_consumer"]

    @property
    def _constructor(self):
        return AuditedEvalFrame

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, list) and isinstance(result, pd.DataFrame):
            audited = AuditedFeatureSlice(result.copy())
            audited._audit_consumer = getattr(self, "_audit_consumer", "unknown")
            return audited
        return result


def capture_consumed_columns(eval_df: pd.DataFrame, consumer: str, runner) -> list[str]:
    audit_df = AuditedEvalFrame(eval_df.copy())
    audit_df._audit_consumer = consumer
    try:
        runner(audit_df)
    except FeatureUsageCaptured as exc:
        if exc.consumer != consumer:
            raise RuntimeError(f"Feature audit consumer mismatch: expected {consumer}, saw {exc.consumer}.")
        return exc.columns
    raise RuntimeError(f"Feature audit for {consumer} did not trigger; rebinding could not be verified.")


def verify_rebinding_takes_effect(e12_module) -> dict:
    fold_c_df, fold_c_summary = e12_module.build_eval(e12_module.STRICT_SPECS["strict_fold_c"])
    nested_consumed = capture_consumed_columns(
        fold_c_df,
        consumer="repeated_nested_cv",
        runner=lambda df: e12_module.repeated_nested_cv(df, "logistic", "strict_fold_c_feature_audit"),
    )
    ranking_consumed = capture_consumed_columns(
        fold_c_df,
        consumer="full_fit_ranking",
        runner=lambda df: e12_module.full_fit_ranking(df, "logistic"),
    )

    for consumer, columns in {
        "repeated_nested_cv": nested_consumed,
        "full_fit_ranking": ranking_consumed,
    }.items():
        if columns != NO_PRIOR_FEATURES:
            raise RuntimeError(
                f"{consumer} consumed the wrong feature list: expected {NO_PRIOR_FEATURES}, got {columns}. "
                "Stopping instead of running a misleading no-prior ablation."
            )
        if "is_secreted" in columns:
            raise RuntimeError(
                f"{consumer} still consumed is_secreted after rebinding. "
                "Stopping instead of running a misleading no-prior ablation."
            )

    return {
        "verification_benchmark": "strict_fold_c",
        "verification_n_genes": int(len(fold_c_df)),
        "verification_n_positives": int(fold_c_summary["n_positives"]),
        "feature_columns_consumed": {
            "repeated_nested_cv": nested_consumed,
            "full_fit_ranking": ranking_consumed,
        },
    }


def collect_comparison(with_prior_payload: dict, no_prior_payload: dict) -> dict:
    comparison = {
        "reference_result_path": str(OLD_E12_PATH),
        "benchmark_model_metrics": {},
        "strict_fold_c_full_fit_ranking": {},
    }

    for benchmark in BENCHMARK_KEYS:
        comparison["benchmark_model_metrics"][benchmark] = {}
        for model in MODEL_KEYS:
            with_prior = with_prior_payload["benchmarks"][benchmark][model]
            no_prior = no_prior_payload["benchmarks"][benchmark][model]
            comparison["benchmark_model_metrics"][benchmark][model] = {
                "with_prior": {
                    "auroc_repeat_mean": with_prior["auroc_repeat_mean"],
                    "auroc_repeat_sd": with_prior["auroc_repeat_sd"],
                    "auroc_oof_mean": with_prior["auroc_oof_mean"],
                },
                "no_prior": {
                    "auroc_repeat_mean": no_prior["auroc_repeat_mean"],
                    "auroc_repeat_sd": no_prior["auroc_repeat_sd"],
                    "auroc_oof_mean": no_prior["auroc_oof_mean"],
                },
            }

    for model in MODEL_KEYS:
        with_prior_rank = with_prior_payload["benchmarks"]["strict_fold_c"][model]["full_fit_rank_metrics"]
        no_prior_rank = no_prior_payload["benchmarks"]["strict_fold_c"][model]["full_fit_rank_metrics"]
        comparison["strict_fold_c_full_fit_ranking"][model] = {
            "with_prior": {
                "p_at_5": with_prior_rank["top5"]["precision"],
                "p_at_10": with_prior_rank["top10"]["precision"],
                "ef_at_10": with_prior_rank["top10"]["enrichment_factor"],
                "top_10_genes": with_prior_rank["top10"]["top_genes"],
            },
            "no_prior": {
                "p_at_5": no_prior_rank["top5"]["precision"],
                "p_at_10": no_prior_rank["top10"]["precision"],
                "ef_at_10": no_prior_rank["top10"]["enrichment_factor"],
                "top_10_genes": no_prior_rank["top10"]["top_genes"],
            },
        }
    return comparison


def fmt_float(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def fmt_mean_sd(mean_value, sd_value) -> str:
    return f"{fmt_float(mean_value)} +/- {fmt_float(sd_value)}"


def print_comparison_table(comparison: dict) -> None:
    print("\nWith-prior vs no-prior AUROC comparison", flush=True)
    print(
        f"{'benchmark':<24} {'model':<10} {'WITH repeat +/- sd':<20} {'NO repeat +/- sd':<20} {'WITH oof':<10} {'NO oof':<10}",
        flush=True,
    )
    for benchmark in BENCHMARK_KEYS:
        for model in MODEL_KEYS:
            entry = comparison["benchmark_model_metrics"][benchmark][model]
            print(
                f"{benchmark:<24} {model:<10} "
                f"{fmt_mean_sd(entry['with_prior']['auroc_repeat_mean'], entry['with_prior']['auroc_repeat_sd']):<20} "
                f"{fmt_mean_sd(entry['no_prior']['auroc_repeat_mean'], entry['no_prior']['auroc_repeat_sd']):<20} "
                f"{fmt_float(entry['with_prior']['auroc_oof_mean']):<10} "
                f"{fmt_float(entry['no_prior']['auroc_oof_mean']):<10}",
                flush=True,
            )

    print("\nStrict Fold C full-fit ranking comparison", flush=True)
    print(
        f"{'model':<10} {'WITH P@5':<10} {'NO P@5':<10} {'WITH P@10':<11} {'NO P@10':<11} {'WITH EF@10':<12} {'NO EF@10':<12}",
        flush=True,
    )
    for model in MODEL_KEYS:
        entry = comparison["strict_fold_c_full_fit_ranking"][model]
        print(
            f"{model:<10} "
            f"{fmt_float(entry['with_prior']['p_at_5']):<10} "
            f"{fmt_float(entry['no_prior']['p_at_5']):<10} "
            f"{fmt_float(entry['with_prior']['p_at_10']):<11} "
            f"{fmt_float(entry['no_prior']['p_at_10']):<11} "
            f"{fmt_float(entry['with_prior']['ef_at_10']):<12} "
            f"{fmt_float(entry['no_prior']['ef_at_10']):<12}",
            flush=True,
        )
        print(f"  WITH top10: {', '.join(entry['with_prior']['top_10_genes'])}", flush=True)
        print(f"  NO   top10: {', '.join(entry['no_prior']['top_10_genes'])}", flush=True)


def main() -> None:
    rc.ensure_results_dir()
    start = time.time()
    print(f"[{rc.utc_timestamp()}] E19 no-prior ablation starting", flush=True)

    with_prior_payload = load_reference_e12_results()
    annotation_bundle, kw_genes, annotation_bundle_path = load_cached_annotation_bundle()
    e9, e12, feature_rebinding_audit = apply_no_prior_feature_patch()
    feature_rebinding_audit["actual_consumption_verification"] = verify_rebinding_takes_effect(e12)

    print(f"Loaded with-prior reference from {OLD_E12_PATH}", flush=True)
    print(f"Loaded cached annotation bundle from {annotation_bundle_path}", flush=True)
    print(f"No-prior feature set ({len(NO_PRIOR_FEATURES)}): {', '.join(NO_PRIOR_FEATURES)}", flush=True)

    payload = rc.provenance(
        SCRIPT_NAME,
        parameters={
            "reference_with_prior_result_path": str(OLD_E12_PATH),
            "annotation_bundle_path": str(annotation_bundle_path),
            "annotation_source_mode": (
                "read-only load from results/rev_exp_annotations.json because "
                "build_annotation_bundle() rewrites cached annotation artifacts"
            ),
            "same_nested_cv_as_e12": True,
            "strict_specs": e12.STRICT_SPECS,
            "within_secreted_spec": e12.WITHIN_SECRETED_SPEC,
            "n_outer": e12.N_OUTER,
            "n_inner": e12.N_INNER,
            "outer_seeds": e12.OUTER_SEEDS,
            "inner_seed": e12.INNER_SEED,
            "xgb_grid": e12.XGB_GRID,
            "logistic_grid": e12.LOGISTIC_GRID,
            "with_prior_feature_columns": list(rc.FULL_FEATURES),
            "no_prior_feature_columns": list(NO_PRIOR_FEATURES),
            "ablation_question": "how much of the deployed model's performance survives when the static prior is dropped entirely",
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_e9_nested_tuning.py",
            "rev_exp_e12_strict_windows.py",
            "rev_exp_e18_uniprot_prior_probe.py",
            "results/rev_exp_e12_strict_windows.json",
            "results/rev_exp_annotations.json",
            "paper/supplementary_table_s1.tex",
        ],
    )
    payload["python_executable"] = sys.executable
    payload["feature_rebinding_audit"] = feature_rebinding_audit
    payload["annotation_bundle_manifest"] = annotation_bundle["uniprot"]["manifest"]
    payload["benchmarks"] = {}

    for key, spec in e12.STRICT_SPECS.items():
        df, summary = e12.build_eval(spec)
        print(f"{key}: n={summary['n_genes']} pos={summary['n_positives']} ({summary['positives_in_eval']})", flush=True)
        entry = {"summary": summary, "baselines": e12.baselines(df)}
        for kind in MODEL_KEYS:
            entry[kind] = e12.repeated_nested_cv(df, kind, key)
            scores, cfg = e12.full_fit_ranking(df, kind)
            entry[kind]["full_fit_rank_metrics"] = {"config": cfg, **e12.rank_metrics(df, scores)}
        payload["benchmarks"][key] = entry

    ws_df, ws_summary = e12.build_eval(e12.WITHIN_SECRETED_SPEC, kw_genes=kw_genes)
    print(
        f"within-secreted strict: n={ws_summary['n_genes']} pos={ws_summary['n_positives']} ({ws_summary['positives_in_eval']})",
        flush=True,
    )
    ws_entry = {"summary": ws_summary, "baselines": e12.baselines(ws_df)}
    for col in ["n_evidence_total", "num_modalities", "num_studies", "layer_count", "sig_max_auc", "deg_count"]:
        if col in ws_df.columns and ws_df[col].nunique() > 1:
            ws_entry["baselines"][col] = {
                "auroc": float(e12.roc_auc_score(ws_df["label"], ws_df[col])),
                "auprc": float(e12.average_precision_score(ws_df["label"], ws_df[col])),
            }
    for kind in MODEL_KEYS:
        ws_entry[kind] = e12.repeated_nested_cv(ws_df, kind, "within_secreted_strict")
        scores, cfg = e12.full_fit_ranking(ws_df, kind)
        ws_entry[kind]["full_fit_rank_metrics"] = {"config": cfg, **e12.rank_metrics(ws_df, scores, ks=(5, 10, 20))}
    payload["benchmarks"]["within_secreted_strict"] = ws_entry

    payload["comparison_to_rev_exp_e12_strict_windows"] = collect_comparison(with_prior_payload, payload)
    payload["runtime_seconds"] = float(time.time() - start)

    rc.write_json(OUT_PATH, payload)
    print_comparison_table(payload["comparison_to_rev_exp_e12_strict_windows"])
    print(f"\nWrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
