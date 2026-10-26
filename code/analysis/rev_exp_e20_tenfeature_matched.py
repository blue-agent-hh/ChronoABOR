#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_common as rc
import rev_exp_e8_spec_model as e8

SCRIPT_NAME = Path(__file__).name
RESULTS_DIR = rc.BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e20_tenfeature_matched.json"
REFERENCE_E12_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"
ANNOTATION_BUNDLE_PATH = RESULTS_DIR / "rev_exp_annotations.json"

MODEL_KEYS = ("xgb", "logistic")
TEN_FEATURES = list(e8.MANUSCRIPT_TEN_FEATURES)
BENCHMARK_SPECS = {
    "strict_fold_c": {
        "cutoff_year": 2020,
        "label_year_start": 2021,
        "label_year_end": 2025,
        "subset_mode": "temporal",
        "rank_ks": (5, 10, 20, 50),
    },
    "strict_fold_d": {
        "cutoff_year": 2021,
        "label_year_start": 2022,
        "label_year_end": 2025,
        "subset_mode": "temporal",
        "rank_ks": (5, 10, 20, 50),
    },
    "within_secreted_strict": {
        "cutoff_year": 2020,
        "label_year_start": 2021,
        "label_year_end": 2025,
        "subset_mode": "within_secreted",
        "rank_ks": (5, 10, 20),
    },
}
EXPECTED_MATCHED_COUNTS = {
    "strict_fold_c": {"n_genes": 2403, "n_positives": 16},
    "strict_fold_d": {"n_genes": 2536, "n_positives": 8},
    "within_secreted_strict": {"n_genes": 246, "n_positives": 4},
}
EXPECTED_SECRETED_COUNT_2020 = 246


def load_reference_e12_results() -> dict:
    if not REFERENCE_E12_PATH.exists():
        raise FileNotFoundError(f"Missing E12 reference result: {REFERENCE_E12_PATH}")
    payload = json.loads(REFERENCE_E12_PATH.read_text())
    missing = [key for key in BENCHMARK_SPECS if key not in payload.get("benchmarks", {})]
    if missing:
        raise RuntimeError(f"E12 reference is missing benchmarks: {missing}")
    return payload


def load_cached_annotation_bundle() -> tuple[dict, set[str], set[str], Path]:
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
    ad_driver_genes = set(bundle["kegg"]["ad_driver_genes"])
    return bundle, kw_genes, ad_driver_genes, ANNOTATION_BUNDLE_PATH


def apply_ten_feature_rebinding():
    import rev_exp_e9_nested_tuning as e9
    import rev_exp_e12_strict_windows as e12

    e9.FULL_FEATURES = list(TEN_FEATURES)
    e12.FULL_FEATURES = list(TEN_FEATURES)

    if list(e9.FULL_FEATURES) != TEN_FEATURES:
        raise RuntimeError(
            "rev_exp_e9_nested_tuning.FULL_FEATURES rebinding did not stick. "
            "Stopping instead of running the wrong protocol."
        )
    if list(e12.FULL_FEATURES) != TEN_FEATURES:
        raise RuntimeError(
            "rev_exp_e12_strict_windows.FULL_FEATURES rebinding did not stick. "
            "Stopping instead of ranking with the wrong feature set."
        )

    audit = {
        "rev_exp_common_full_features_16f_reference": list(rc.FULL_FEATURES),
        "ten_feature_columns": list(TEN_FEATURES),
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


def verify_rebinding_takes_effect(eval_df: pd.DataFrame, e9_module, e12_module) -> dict:
    nested_consumed = capture_consumed_columns(
        eval_df,
        consumer="repeated_nested_cv",
        runner=lambda df: e9_module.repeated_nested_cv(df, "logistic", "strict_fold_c_feature_audit"),
    )
    ranking_consumed = capture_consumed_columns(
        eval_df,
        consumer="full_fit_ranking",
        runner=lambda df: e12_module.full_fit_ranking(df, "logistic"),
    )

    for consumer, columns in {
        "repeated_nested_cv": nested_consumed,
        "full_fit_ranking": ranking_consumed,
    }.items():
        if columns != TEN_FEATURES:
            raise RuntimeError(
                f"{consumer} consumed {columns}, expected {TEN_FEATURES}. "
                "Stopping instead of running a misleading matched comparison."
            )

    return {
        "verification_benchmark": "strict_fold_c",
        "verification_n_genes": int(len(eval_df)),
        "verification_n_positives": int(eval_df["label"].sum()),
        "feature_columns_consumed": {
            "repeated_nested_cv": nested_consumed,
            "full_fit_ranking": ranking_consumed,
        },
    }


def build_ten_feature_frames(kw_genes: set[str], ad_driver_genes: set[str]) -> tuple[dict[int, pd.DataFrame], dict]:
    annotated_tables, _, _ = e8.prepare_annotated_tables()
    scored_universe = rc.build_gene_universe(rc.load_all_tables())

    frames = {}
    audit = {
        "construction_path": {
            "prepare_annotated_tables": "rev_exp_e8_spec_model.prepare_annotated_tables()",
            "scored_universe": "rev_exp_common.build_gene_universe(rev_exp_common.load_all_tables())",
            "base_matrix_builder": "rev_exp_e8_spec_model.build_base_feature_matrix(...)",
            "secreted_attachment": "rev_exp_e8_spec_model.attach_secreted_definition(..., kw_genes)",
        },
        "cutoffs": {},
    }
    for cutoff_year in sorted({spec["cutoff_year"] for spec in BENCHMARK_SPECS.values()}):
        base = e8.build_base_feature_matrix(cutoff_year, annotated_tables, scored_universe, ad_driver_genes)
        feature_df = e8.attach_secreted_definition(base["feature_df"], kw_genes)
        frames[cutoff_year] = feature_df
        audit["cutoffs"][str(cutoff_year)] = {
            "n_genes": int(len(feature_df)),
            "n_secreted_kw0964": int(feature_df["is_secreted"].sum()),
            "n_with_temporal_evidence_by_cutoff": int(feature_df["has_temporal_evidence_by_cutoff"].sum()),
            "n_with_feature_layer_evidence_by_cutoff": int(feature_df["has_feature_layer_evidence_by_cutoff"].sum()),
            "n_added_via_target_only": int(base["n_added_via_target_only"]),
            "genes_added_via_target_only": list(base["genes_added_via_target_only"]),
        }
    return frames, audit


def build_native_eval_df(feature_df: pd.DataFrame, benchmark_key: str, positives: set[str]) -> pd.DataFrame:
    subset_mode = BENCHMARK_SPECS[benchmark_key]["subset_mode"]
    if subset_mode == "within_secreted":
        eval_df = feature_df.loc[feature_df["is_secreted"] == 1].copy()
    else:
        eval_df = feature_df.loc[feature_df["has_temporal_evidence_by_cutoff"]].copy()
    eval_df = eval_df.reset_index(drop=True)
    if not eval_df["gene"].is_unique:
        raise RuntimeError(f"{benchmark_key}: native ten-feature eval frame has duplicate gene rows.")
    eval_df["label"] = eval_df["gene"].isin(positives).astype(int)
    return eval_df


def build_reference_eval_df(benchmark_key: str, e12_module, kw_genes: set[str]) -> tuple[pd.DataFrame, dict]:
    if benchmark_key == "within_secreted_strict":
        return e12_module.build_eval(e12_module.WITHIN_SECRETED_SPEC, kw_genes=kw_genes)
    return e12_module.build_eval(e12_module.STRICT_SPECS[benchmark_key])


def match_to_reference_eval(
    benchmark_key: str,
    native_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    if not reference_df["gene"].is_unique:
        raise RuntimeError(f"{benchmark_key}: reconstructed E12 eval frame has duplicate gene rows.")

    reference_genes = reference_df["gene"].astype(str).tolist()
    reference_gene_set = set(reference_genes)
    native_gene_set = set(native_df["gene"].astype(str))
    missing_required_genes = [gene for gene in reference_genes if gene not in native_gene_set]
    intersection_genes = [gene for gene in reference_genes if gene in native_gene_set]
    native_only_genes = sorted(native_gene_set - reference_gene_set)

    matched_df = native_df.set_index("gene", drop=False).loc[intersection_genes].reset_index(drop=True)
    matched_df = matched_df.reset_index(drop=True)

    reference_label_map = reference_df.set_index("gene")["label"].to_dict()
    matched_reference_labels = matched_df["gene"].map(reference_label_map).astype(int)
    if not matched_df["label"].equals(matched_reference_labels):
        raise RuntimeError(f"{benchmark_key}: matched ten-feature labels do not agree with reconstructed E12 labels.")

    match_audit = {
        "native_eval_gene_count": int(len(native_df)),
        "native_eval_positive_count": int(native_df["label"].sum()),
        "reference_eval_gene_count": int(reference_summary["n_genes"]),
        "reference_eval_positive_count": int(reference_summary["n_positives"]),
        "reference_eval_genes": reference_genes,
        "native_only_gene_count": int(len(native_only_genes)),
        "native_only_genes": native_only_genes,
        "missing_required_gene_count": int(len(missing_required_genes)),
        "missing_required_genes": missing_required_genes,
        "intersection_gene_count": int(len(intersection_genes)),
        "intersection_positive_count": int(matched_df["label"].sum()),
        "exact_reference_gene_set_recovered": len(missing_required_genes) == 0,
    }

    expected = EXPECTED_MATCHED_COUNTS[benchmark_key]
    if int(reference_summary["n_genes"]) != expected["n_genes"] or int(reference_summary["n_positives"]) != expected["n_positives"]:
        raise RuntimeError(
            f"{benchmark_key}: reconstructed E12 reference counts "
            f"({reference_summary['n_genes']}, {reference_summary['n_positives']}) do not match expected "
            f"({expected['n_genes']}, {expected['n_positives']})."
        )
    if len(missing_required_genes) > 0:
        raise RuntimeError(
            f"{benchmark_key}: the ten-feature frame is missing {len(missing_required_genes)} required E12 genes; "
            f"intersection size={len(intersection_genes)}. Stopping instead of comparing different universes."
        )
    if len(matched_df) != expected["n_genes"] or int(matched_df["label"].sum()) != expected["n_positives"]:
        raise RuntimeError(
            f"{benchmark_key}: matched ten-feature counts ({len(matched_df)}, {int(matched_df['label'].sum())}) "
            f"do not match expected ({expected['n_genes']}, {expected['n_positives']})."
        )

    return matched_df, match_audit


def collect_comparison(reference_payload: dict, new_payload: dict) -> dict:
    comparison = {
        "reference_result_path": str(REFERENCE_E12_PATH),
        "benchmark_model_metrics": {},
        "strict_fold_c_full_fit_ranking": {},
    }
    for benchmark_key in BENCHMARK_SPECS:
        comparison["benchmark_model_metrics"][benchmark_key] = {}
        for model_key in MODEL_KEYS:
            old_model = reference_payload["benchmarks"][benchmark_key][model_key]
            new_model = new_payload["benchmarks"][benchmark_key][model_key]
            comparison["benchmark_model_metrics"][benchmark_key][model_key] = {
                "deployed_16_feature": {
                    "auroc_repeat_mean": old_model["auroc_repeat_mean"],
                    "auroc_repeat_sd": old_model["auroc_repeat_sd"],
                    "auroc_oof_mean": old_model["auroc_oof_mean"],
                },
                "matched_10_feature": {
                    "auroc_repeat_mean": new_model["auroc_repeat_mean"],
                    "auroc_repeat_sd": new_model["auroc_repeat_sd"],
                    "auroc_oof_mean": new_model["auroc_oof_mean"],
                },
            }

    for model_key in MODEL_KEYS:
        old_rank = reference_payload["benchmarks"]["strict_fold_c"][model_key]["full_fit_rank_metrics"]
        new_rank = new_payload["benchmarks"]["strict_fold_c"][model_key]["full_fit_rank_metrics"]
        comparison["strict_fold_c_full_fit_ranking"][model_key] = {
            "deployed_16_feature": {
                "p_at_5": old_rank["top5"]["precision"],
                "p_at_10": old_rank["top10"]["precision"],
                "ef_at_10": old_rank["top10"]["enrichment_factor"],
                "top_10_genes": old_rank["top10"]["top_genes"],
            },
            "matched_10_feature": {
                "p_at_5": new_rank["top5"]["precision"],
                "p_at_10": new_rank["top10"]["precision"],
                "ef_at_10": new_rank["top10"]["enrichment_factor"],
                "top_10_genes": new_rank["top10"]["top_genes"],
            },
        }
    return comparison


def fmt_float(value) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.3f}"


def fmt_mean_sd(mean_value, sd_value) -> str:
    return f"{fmt_float(mean_value)} +/- {fmt_float(sd_value)}"


def print_comparison_table(comparison: dict) -> None:
    print("\nBenchmark x model AUROC comparison", flush=True)
    print(
        f"{'benchmark':<24} {'model':<10} {'16F repeat +/- sd':<20} {'10F repeat +/- sd':<20} {'16F oof':<10} {'10F oof':<10}",
        flush=True,
    )
    for benchmark_key in BENCHMARK_SPECS:
        for model_key in MODEL_KEYS:
            entry = comparison["benchmark_model_metrics"][benchmark_key][model_key]
            print(
                f"{benchmark_key:<24} {model_key:<10} "
                f"{fmt_mean_sd(entry['deployed_16_feature']['auroc_repeat_mean'], entry['deployed_16_feature']['auroc_repeat_sd']):<20} "
                f"{fmt_mean_sd(entry['matched_10_feature']['auroc_repeat_mean'], entry['matched_10_feature']['auroc_repeat_sd']):<20} "
                f"{fmt_float(entry['deployed_16_feature']['auroc_oof_mean']):<10} "
                f"{fmt_float(entry['matched_10_feature']['auroc_oof_mean']):<10}",
                flush=True,
            )

    print("\nStrict Fold C full-fit ranking comparison", flush=True)
    print(
        f"{'model':<10} {'16F P@5':<10} {'10F P@5':<10} {'16F P@10':<11} {'10F P@10':<11} {'16F EF@10':<12} {'10F EF@10':<12}",
        flush=True,
    )
    for model_key in MODEL_KEYS:
        entry = comparison["strict_fold_c_full_fit_ranking"][model_key]
        print(
            f"{model_key:<10} "
            f"{fmt_float(entry['deployed_16_feature']['p_at_5']):<10} "
            f"{fmt_float(entry['matched_10_feature']['p_at_5']):<10} "
            f"{fmt_float(entry['deployed_16_feature']['p_at_10']):<11} "
            f"{fmt_float(entry['matched_10_feature']['p_at_10']):<11} "
            f"{fmt_float(entry['deployed_16_feature']['ef_at_10']):<12} "
            f"{fmt_float(entry['matched_10_feature']['ef_at_10']):<12}",
            flush=True,
        )
        print(f"  16F top10: {', '.join(entry['deployed_16_feature']['top_10_genes'])}", flush=True)
        print(f"  10F top10: {', '.join(entry['matched_10_feature']['top_10_genes'])}", flush=True)


def main() -> None:
    rc.ensure_results_dir()
    start = time.time()
    print(f"[{rc.utc_timestamp()}] E20 ten-feature matched comparison starting", flush=True)

    reference_payload = load_reference_e12_results()
    annotation_bundle, kw_genes, ad_driver_genes, annotation_bundle_path = load_cached_annotation_bundle()
    e9, e12, feature_rebinding_audit = apply_ten_feature_rebinding()
    ten_feature_frames, ten_feature_build_audit = build_ten_feature_frames(kw_genes, ad_driver_genes)

    secreted_count_2020 = int(ten_feature_frames[2020]["is_secreted"].sum())
    print(f"Ten-feature cutoff-2020 is_secreted=1 count: {secreted_count_2020}", flush=True)
    if secreted_count_2020 != EXPECTED_SECRETED_COUNT_2020:
        raise RuntimeError(
            f"Expected {EXPECTED_SECRETED_COUNT_2020} cutoff-2020 UniProt KW-0964 genes, found {secreted_count_2020}."
        )

    nm_rows = e12.load_nm_corpus_table()
    benchmark_frames: dict[str, pd.DataFrame] = {}
    benchmark_payloads: dict[str, dict] = {}

    for benchmark_key, spec in BENCHMARK_SPECS.items():
        positives = e12.positives_for_window(nm_rows, spec["label_year_start"], spec["label_year_end"])
        feature_df = ten_feature_frames[spec["cutoff_year"]]
        native_df = build_native_eval_df(feature_df, benchmark_key, positives)
        reference_df, reference_summary = build_reference_eval_df(benchmark_key, e12, kw_genes)
        matched_df, match_audit = match_to_reference_eval(benchmark_key, native_df, reference_df, reference_summary)

        print(
            f"{benchmark_key}: native n={len(native_df)} pos={int(native_df['label'].sum())}; "
            f"matched n={len(matched_df)} pos={int(matched_df['label'].sum())}; "
            f"missing_required={match_audit['missing_required_gene_count']}; "
            f"intersection={match_audit['intersection_gene_count']}",
            flush=True,
        )

        benchmark_frames[benchmark_key] = matched_df
        benchmark_payloads[benchmark_key] = {
            "spec": dict(spec),
            "summary": {
                **spec,
                "positives_requested": sorted(positives),
                "n_positives_requested": int(len(positives)),
                "positives_in_eval": sorted(set(matched_df.loc[matched_df["label"] == 1, "gene"])),
                "positives_missing_from_eval": sorted(positives - set(matched_df["gene"])),
                "n_genes": int(len(matched_df)),
                "n_positives": int(matched_df["label"].sum()),
            },
            "native_ten_feature_eval": {
                "n_genes": int(len(native_df)),
                "n_positives": int(native_df["label"].sum()),
            },
            "reference_e12_eval": reference_summary,
            "match_audit": match_audit,
        }

    feature_rebinding_audit["actual_consumption_verification"] = verify_rebinding_takes_effect(
        benchmark_frames["strict_fold_c"],
        e9,
        e12,
    )

    payload = rc.provenance(
        SCRIPT_NAME,
        parameters={
            "reference_e12_result_path": str(REFERENCE_E12_PATH),
            "annotation_bundle_path": str(annotation_bundle_path),
            "annotation_source_mode": (
                "read-only load from results/rev_exp_annotations.json because "
                "build_annotation_bundle() rewrites cached annotation artifacts"
            ),
            "ten_feature_builder_path": "rev_exp_e8_spec_model.build_base_feature_matrix + attach_secreted_definition",
            "ten_feature_columns": list(TEN_FEATURES),
            "deployed_16_feature_columns": list(rc.FULL_FEATURES),
            "strict_specs": {
                key: {k: v for k, v in spec.items() if k != "subset_mode" and k != "rank_ks"}
                for key, spec in BENCHMARK_SPECS.items()
            },
            "same_nested_cv_as_e12": True,
            "n_outer": e12.N_OUTER,
            "n_inner": e12.N_INNER,
            "outer_seeds": e12.OUTER_SEEDS,
            "inner_seed": e12.INNER_SEED,
            "xgb_grid": e12.XGB_GRID,
            "logistic_grid": e12.LOGISTIC_GRID,
            "matched_gene_set_required": True,
            "expected_matched_counts": EXPECTED_MATCHED_COUNTS,
            "strict_fold_c_comparison_question": (
                "How does the manuscript ten-feature set perform against the deployed 16-feature model "
                "when both are evaluated on the exact E12 gene universes, labels, seeds, and tuning protocol?"
            ),
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_annotations.py",
            "rev_exp_e8_spec_model.py",
            "rev_exp_e9_nested_tuning.py",
            "rev_exp_e12_strict_windows.py",
            "rev_exp_e13_strict_spec_ablation.py",
            "rev_exp_e18_uniprot_prior_probe.py",
            "rev_exp_e19_no_prior_ablation.py",
            "results/rev_exp_annotations.json",
            "results/rev_exp_e12_strict_windows.json",
            "paper/supplementary_table_s1.tex",
        ],
    )
    payload["python_executable"] = sys.executable
    payload["annotation_bundle_manifest"] = annotation_bundle["uniprot"]["manifest"]
    payload["feature_rebinding_audit"] = feature_rebinding_audit
    payload["ten_feature_build_audit"] = ten_feature_build_audit
    payload["sanity_checks"] = {
        "cutoff_2020_uniprot_kw0964_secreted_count": secreted_count_2020,
        "expected_cutoff_2020_uniprot_kw0964_secreted_count": EXPECTED_SECRETED_COUNT_2020,
    }
    payload["benchmarks"] = {}

    for benchmark_key, spec in BENCHMARK_SPECS.items():
        matched_df = benchmark_frames[benchmark_key]
        entry = benchmark_payloads[benchmark_key]
        for model_key in MODEL_KEYS:
            entry[model_key] = e9.repeated_nested_cv(matched_df, model_key, benchmark_key)
            full_scores, cfg = e12.full_fit_ranking(matched_df, model_key)
            entry[model_key]["full_fit_rank_metrics"] = {
                "config": cfg,
                **e12.rank_metrics(matched_df, full_scores, ks=spec["rank_ks"]),
            }
        payload["benchmarks"][benchmark_key] = entry

    payload["comparison_to_rev_exp_e12_strict_windows"] = collect_comparison(reference_payload, payload)
    payload["runtime_seconds"] = float(time.time() - start)

    rc.write_json(OUT_PATH, payload)
    print_comparison_table(payload["comparison_to_rev_exp_e12_strict_windows"])
    print(f"\nWrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
