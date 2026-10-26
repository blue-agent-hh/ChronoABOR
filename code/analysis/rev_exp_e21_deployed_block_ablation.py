#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_common as rc

SCRIPT_NAME = Path(__file__).name
RESULTS_DIR = rc.BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e21_deployed_block_ablation.json"
E12_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"
E19_PATH = RESULTS_DIR / "rev_exp_e19_no_prior_ablation.json"

STRICT_FOLD_C_KEY = "strict_fold_c"
EXPECTED_STRICT_FOLD_C = {
    "cutoff_year": 2020,
    "label_year_start": 2021,
    "label_year_end": 2025,
    "n_genes": 2403,
    "n_positives": 16,
}
MODEL_KEYS = ("xgb", "logistic")
FULL_FEATURES = list(rc.FULL_FEATURES)
TEMPORAL_FEATURES = list(rc.FEATURE_COLUMNS)
STATIC_PRIOR_FEATURE = ["is_secreted"]

FEATURE_BLOCKS = {
    "deg_block": [
        "deg_count",
        "deg_up_count",
        "deg_down_count",
        "deg_tissue_breadth",
        "num_deg_studies",
    ],
    "signature_block": [
        "sig_count_all",
        "sig_count_peripheral",
        "sig_count_high_auc",
        "sig_max_auc",
    ],
    "other_layer_block": [
        "snp_count",
        "cpg_count",
        "hub_count",
        "mirna_count",
    ],
    "aggregate_block": [
        "layer_count",
        "num_studies",
    ],
    "static_prior": STATIC_PRIOR_FEATURE,
}

EXPECTED_CONFIGURATION_COUNTS = {
    "full_model": 16,
    "no_deg_block": 11,
    "no_signature_block": 12,
    "no_other_layer_block": 12,
    "no_aggregate_block": 14,
    "prior_only": 1,
    "temporal_only": 15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate sources, strict Fold C counts, and per-configuration rebinding audits without running nested CV.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text())


def ordered_feature_subset(excluded_features: list[str]) -> list[str]:
    excluded = set(excluded_features)
    return [feature for feature in FULL_FEATURES if feature not in excluded]


def build_configuration_specs() -> list[dict]:
    specs = [
        {
            "name": "full_model",
            "feature_columns": list(FULL_FEATURES),
            "dropped_blocks": [],
            "source_label": "reused:e12",
            "reuse_source": "e12",
        },
        {
            "name": "no_deg_block",
            "feature_columns": ordered_feature_subset(FEATURE_BLOCKS["deg_block"]),
            "dropped_blocks": ["deg_block"],
            "source_label": "recomputed",
            "reuse_source": None,
        },
        {
            "name": "no_signature_block",
            "feature_columns": ordered_feature_subset(FEATURE_BLOCKS["signature_block"]),
            "dropped_blocks": ["signature_block"],
            "source_label": "recomputed",
            "reuse_source": None,
        },
        {
            "name": "no_other_layer_block",
            "feature_columns": ordered_feature_subset(FEATURE_BLOCKS["other_layer_block"]),
            "dropped_blocks": ["other_layer_block"],
            "source_label": "recomputed",
            "reuse_source": None,
        },
        {
            "name": "no_aggregate_block",
            "feature_columns": ordered_feature_subset(FEATURE_BLOCKS["aggregate_block"]),
            "dropped_blocks": ["aggregate_block"],
            "source_label": "recomputed",
            "reuse_source": None,
        },
        {
            "name": "prior_only",
            "feature_columns": list(STATIC_PRIOR_FEATURE),
            "dropped_blocks": ["deg_block", "signature_block", "other_layer_block", "aggregate_block"],
            "source_label": "recomputed",
            "reuse_source": None,
        },
        {
            "name": "temporal_only",
            "feature_columns": list(TEMPORAL_FEATURES),
            "dropped_blocks": ["static_prior"],
            "source_label": "reused:e19",
            "reuse_source": "e19",
        },
    ]

    for spec in specs:
        spec["feature_columns"] = list(spec["feature_columns"])
        spec["n_features"] = len(spec["feature_columns"])
        spec["dropped_features"] = [feature for feature in FULL_FEATURES if feature not in spec["feature_columns"]]
        spec["retained_blocks"] = [
            block_name
            for block_name in FEATURE_BLOCKS
            if all(feature in spec["feature_columns"] for feature in FEATURE_BLOCKS[block_name])
        ]
    return specs


def validate_feature_blocks() -> dict:
    if FULL_FEATURES != TEMPORAL_FEATURES + STATIC_PRIOR_FEATURE:
        raise RuntimeError(
            "rev_exp_common.FULL_FEATURES no longer matches FEATURE_COLUMNS + ['is_secreted']; "
            "stopping instead of running the wrong deployed feature set."
        )

    flattened = [feature for features in FEATURE_BLOCKS.values() for feature in features]
    duplicates = [feature for feature, count in Counter(flattened).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Feature blocks are not disjoint; duplicates: {duplicates}")

    missing = [feature for feature in FULL_FEATURES if feature not in flattened]
    extras = [feature for feature in flattened if feature not in FULL_FEATURES]
    if missing or extras:
        raise RuntimeError(f"Feature-block coverage mismatch. Missing={missing}, extras={extras}")

    return {
        "deployed_full_features": list(FULL_FEATURES),
        "temporal_features": list(TEMPORAL_FEATURES),
        "feature_blocks": copy.deepcopy(FEATURE_BLOCKS),
        "covered_feature_count": len(flattened),
        "expected_feature_count": len(FULL_FEATURES),
        "coverage_exact": True,
        "subset_order_policy": "all configuration feature lists preserve rev_exp_common.FULL_FEATURES order",
    }


def validate_configuration_specs(config_specs: list[dict]) -> dict:
    names = [spec["name"] for spec in config_specs]
    expected_names = list(EXPECTED_CONFIGURATION_COUNTS)
    if names != expected_names:
        raise RuntimeError(f"Unexpected configuration order: {names}; expected {expected_names}")

    for spec in config_specs:
        if spec["n_features"] != EXPECTED_CONFIGURATION_COUNTS[spec["name"]]:
            raise RuntimeError(
                f"{spec['name']} expected {EXPECTED_CONFIGURATION_COUNTS[spec['name']]} features, "
                f"found {spec['n_features']}."
            )
        if len(spec["feature_columns"]) != len(set(spec["feature_columns"])):
            raise RuntimeError(f"{spec['name']} contains duplicate features: {spec['feature_columns']}")
        unknown = [feature for feature in spec["feature_columns"] if feature not in FULL_FEATURES]
        if unknown:
            raise RuntimeError(f"{spec['name']} contains features outside the deployed set: {unknown}")

    return {
        "configuration_order": names,
        "expected_feature_counts": dict(EXPECTED_CONFIGURATION_COUNTS),
    }


def assert_strict_fold_c_summary(summary: dict, where: str) -> dict:
    for key, expected_value in EXPECTED_STRICT_FOLD_C.items():
        actual_value = summary.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"{where}: expected strict Fold C {key}={expected_value}, found {actual_value}. "
                "Stopping instead of running a mismatched benchmark."
            )
    return {key: summary[key] for key in EXPECTED_STRICT_FOLD_C}


def validate_model_row(model_entry: dict, where: str, e12_module) -> dict:
    if int(model_entry.get("n_repeats", -1)) != len(e12_module.OUTER_SEEDS):
        raise RuntimeError(f"{where}: unexpected n_repeats={model_entry.get('n_repeats')}")
    if int(model_entry.get("n_outer_folds", -1)) != e12_module.N_OUTER:
        raise RuntimeError(f"{where}: unexpected n_outer_folds={model_entry.get('n_outer_folds')}")
    if int(model_entry.get("n_inner_folds", -1)) != e12_module.N_INNER:
        raise RuntimeError(f"{where}: unexpected n_inner_folds={model_entry.get('n_inner_folds')}")
    if "full_fit_rank_metrics" not in model_entry:
        raise RuntimeError(f"{where}: missing full_fit_rank_metrics")
    for top_key in ("top5", "top10"):
        if top_key not in model_entry["full_fit_rank_metrics"]:
            raise RuntimeError(f"{where}: missing {top_key} in full_fit_rank_metrics")
    return {
        "n_repeats": model_entry["n_repeats"],
        "n_outer_folds": model_entry["n_outer_folds"],
        "n_inner_folds": model_entry["n_inner_folds"],
        "auroc_repeat_mean": model_entry["auroc_repeat_mean"],
        "auroc_repeat_sd": model_entry["auroc_repeat_sd"],
        "p_at_10": model_entry["full_fit_rank_metrics"]["top10"]["precision"],
    }


def validate_e12_reference(payload: dict, e12_module) -> dict:
    if payload.get("script_name") != "rev_exp_e12_strict_windows.py":
        raise RuntimeError(f"Unexpected E12 reference provenance: {payload.get('script_name')}")
    params = payload.get("parameters", {})
    if params.get("strict_specs") != e12_module.STRICT_SPECS:
        raise RuntimeError("E12 reference strict specs do not match the live E12 module.")
    if params.get("n_outer") != e12_module.N_OUTER or params.get("n_inner") != e12_module.N_INNER:
        raise RuntimeError("E12 reference fold counts do not match the live E12 protocol.")
    if params.get("outer_seeds") != e12_module.OUTER_SEEDS or params.get("inner_seed") != e12_module.INNER_SEED:
        raise RuntimeError("E12 reference seeds do not match the live E12 protocol.")
    if params.get("xgb_grid") != e12_module.XGB_GRID or params.get("logistic_grid") != e12_module.LOGISTIC_GRID:
        raise RuntimeError("E12 reference grids do not match the live E12 protocol.")

    benchmark = payload.get("benchmarks", {}).get(STRICT_FOLD_C_KEY)
    if not isinstance(benchmark, dict):
        raise RuntimeError(f"E12 reference is missing benchmark {STRICT_FOLD_C_KEY}")

    summary_audit = assert_strict_fold_c_summary(benchmark["summary"], "E12 reference")
    model_audit = {
        model_key: validate_model_row(benchmark[model_key], f"E12 reference/{model_key}", e12_module)
        for model_key in MODEL_KEYS
    }

    imported_defaults = list(e12_module.FULL_FEATURES)
    if imported_defaults != FULL_FEATURES:
        raise RuntimeError(
            "Live rev_exp_e12_strict_windows.FULL_FEATURES default import no longer matches the deployed 16-feature list."
        )

    return {
        "reference_result_path": str(E12_PATH),
        "script_name": payload["script_name"],
        "timestamp_utc": payload.get("timestamp_utc"),
        "strict_fold_c_summary_verified": summary_audit,
        "protocol_verified": {
            "n_outer": params["n_outer"],
            "n_inner": params["n_inner"],
            "outer_seeds": params["outer_seeds"],
            "inner_seed": params["inner_seed"],
        },
        "model_rows_verified": model_audit,
        "feature_columns_reference_mode": "canonical_e12_default_module_state",
        "feature_columns_reference": imported_defaults,
        "note": (
            "E12 does not serialize FULL_FEATURES inside the strict Fold C model rows. "
            "This row is accepted because E12 is the canonical strict-window baseline, "
            "its protocol fields match the live module exactly, and the live imported "
            "default FULL_FEATURES equals the deployed 16-feature list before any rebinding."
        ),
    }


def validate_e19_reference(payload: dict, e12_module) -> dict:
    if payload.get("script_name") != "rev_exp_e19_no_prior_ablation.py":
        raise RuntimeError(f"Unexpected E19 reference provenance: {payload.get('script_name')}")
    params = payload.get("parameters", {})
    if params.get("same_nested_cv_as_e12") is not True:
        raise RuntimeError("E19 reference does not declare same_nested_cv_as_e12=True.")
    if params.get("strict_specs") != e12_module.STRICT_SPECS:
        raise RuntimeError("E19 reference strict specs do not match the live E12 module.")
    if params.get("n_outer") != e12_module.N_OUTER or params.get("n_inner") != e12_module.N_INNER:
        raise RuntimeError("E19 reference fold counts do not match the live E12 protocol.")
    if params.get("outer_seeds") != e12_module.OUTER_SEEDS or params.get("inner_seed") != e12_module.INNER_SEED:
        raise RuntimeError("E19 reference seeds do not match the live E12 protocol.")
    if params.get("xgb_grid") != e12_module.XGB_GRID or params.get("logistic_grid") != e12_module.LOGISTIC_GRID:
        raise RuntimeError("E19 reference grids do not match the live E12 protocol.")

    benchmark = payload.get("benchmarks", {}).get(STRICT_FOLD_C_KEY)
    if not isinstance(benchmark, dict):
        raise RuntimeError(f"E19 reference is missing benchmark {STRICT_FOLD_C_KEY}")

    summary_audit = assert_strict_fold_c_summary(benchmark["summary"], "E19 reference")
    model_audit = {
        model_key: validate_model_row(benchmark[model_key], f"E19 reference/{model_key}", e12_module)
        for model_key in MODEL_KEYS
    }

    consumption = payload.get("feature_rebinding_audit", {}).get("actual_consumption_verification", {}).get(
        "feature_columns_consumed"
    )
    if not isinstance(consumption, dict):
        raise RuntimeError("E19 reference is missing serialized feature consumption audit.")
    for consumer in ("repeated_nested_cv", "full_fit_ranking"):
        if consumption.get(consumer) != TEMPORAL_FEATURES:
            raise RuntimeError(
                f"E19 reference {consumer} consumed {consumption.get(consumer)}, expected temporal-only features."
            )

    return {
        "reference_result_path": str(E19_PATH),
        "script_name": payload["script_name"],
        "timestamp_utc": payload.get("timestamp_utc"),
        "strict_fold_c_summary_verified": summary_audit,
        "protocol_verified": {
            "n_outer": params["n_outer"],
            "n_inner": params["n_inner"],
            "outer_seeds": params["outer_seeds"],
            "inner_seed": params["inner_seed"],
        },
        "model_rows_verified": model_audit,
        "feature_columns_reference_mode": "serialized_e19_rebinding_audit",
        "feature_columns_reference": copy.deepcopy(consumption),
        "note": "E19 explicitly serialized the post-rebinding feature consumption for both consumers.",
    }


def load_prior_only_baseline_reference(e12_payload: dict, e19_payload: dict) -> dict:
    e12_baseline = e12_payload["benchmarks"][STRICT_FOLD_C_KEY]["baselines"]["secretion_status"]
    e19_baseline = e19_payload["benchmarks"][STRICT_FOLD_C_KEY]["baselines"]["secretion_status"]
    if e12_baseline != e19_baseline:
        raise RuntimeError(
            f"Strict Fold C secretion_status baseline mismatch between E12 and E19: {e12_baseline} vs {e19_baseline}"
        )
    return {
        "reference_paths": [str(E12_PATH), str(E19_PATH)],
        "baseline_name": "secretion_status",
        "strict_fold_c_baseline": copy.deepcopy(e12_baseline),
        "validated_identical_in_e12_and_e19": True,
        "reuse_decision": (
            "Baseline was read-only loaded for provenance, but prior_only was recomputed here because the cached "
            "JSONs do not serialize an identical repeated_nested_cv + full_fit_rank_metrics row for is_secreted alone."
        ),
    }


def apply_feature_rebinding(e9_module, e12_module, feature_columns: list[str]) -> dict:
    rebound = list(feature_columns)
    e9_module.FULL_FEATURES = rebound
    e12_module.FULL_FEATURES = rebound

    if list(e9_module.FULL_FEATURES) != rebound:
        raise RuntimeError(
            f"rev_exp_e9_nested_tuning.FULL_FEATURES rebinding did not stick for {rebound}. "
            "Stopping instead of running the wrong protocol."
        )
    if list(e12_module.FULL_FEATURES) != rebound:
        raise RuntimeError(
            f"rev_exp_e12_strict_windows.FULL_FEATURES rebinding did not stick for {rebound}. "
            "Stopping instead of ranking with the wrong feature set."
        )

    return {
        "rev_exp_common_full_features_reference": list(FULL_FEATURES),
        "rev_exp_e9_full_features_after_rebinding": list(e9_module.FULL_FEATURES),
        "rev_exp_e12_full_features_after_rebinding": list(e12_module.FULL_FEATURES),
    }


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


def verify_rebinding_takes_effect(
    eval_df: pd.DataFrame,
    config_name: str,
    feature_columns: list[str],
    e9_module,
    e12_module,
) -> dict:
    nested_consumed = capture_consumed_columns(
        eval_df,
        consumer="repeated_nested_cv",
        runner=lambda df: e9_module.repeated_nested_cv(df, "logistic", f"{config_name}_feature_audit"),
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
        if columns != feature_columns:
            raise RuntimeError(
                f"{config_name}: {consumer} consumed {columns}, expected {feature_columns}. "
                "Stopping instead of producing numbers from the wrong feature set."
            )

    return {
        "configuration": config_name,
        "verification_model": "logistic",
        "verification_n_genes": int(len(eval_df)),
        "verification_n_positives": int(eval_df["label"].sum()),
        "feature_columns_consumed": {
            "repeated_nested_cv": nested_consumed,
            "full_fit_ranking": ranking_consumed,
        },
    }


def evaluate_configuration(
    config_spec: dict,
    eval_df: pd.DataFrame,
    e9_module,
    e12_module,
    reuse_payloads: dict[str, dict],
    reuse_validations: dict[str, dict],
    prior_only_baseline_reference: dict,
    check_only: bool,
) -> dict:
    print(
        f"Preparing {config_spec['name']} | source={config_spec['source_label']} | "
        f"n_features={config_spec['n_features']} | features={config_spec['feature_columns']}",
        flush=True,
    )
    feature_rebinding_audit = apply_feature_rebinding(e9_module, e12_module, config_spec["feature_columns"])
    feature_rebinding_audit["actual_consumption_verification"] = verify_rebinding_takes_effect(
        eval_df=eval_df,
        config_name=config_spec["name"],
        feature_columns=config_spec["feature_columns"],
        e9_module=e9_module,
        e12_module=e12_module,
    )

    entry = {
        "feature_columns": list(config_spec["feature_columns"]),
        "n_features": int(config_spec["n_features"]),
        "dropped_blocks": list(config_spec["dropped_blocks"]),
        "retained_blocks": list(config_spec["retained_blocks"]),
        "dropped_features": list(config_spec["dropped_features"]),
        "source_label": config_spec["source_label"],
        "feature_rebinding_audit": feature_rebinding_audit,
    }

    if config_spec["name"] == "prior_only":
        entry["read_only_json_baseline_reference"] = copy.deepcopy(prior_only_baseline_reference)

    if check_only:
        entry["execution_mode"] = "check_only"
        print(f"Check-only audit passed for {config_spec['name']}", flush=True)
        return entry

    if config_spec["reuse_source"] is not None:
        benchmark = reuse_payloads[config_spec["reuse_source"]]["benchmarks"][STRICT_FOLD_C_KEY]
        entry["execution_mode"] = "reused"
        entry["reused_from"] = copy.deepcopy(reuse_validations[config_spec["reuse_source"]])
        for model_key in MODEL_KEYS:
            entry[model_key] = copy.deepcopy(benchmark[model_key])
        print(
            f"Reused {config_spec['name']} from {entry['reused_from']['reference_result_path']} "
            f"under identical strict Fold C protocol.",
            flush=True,
        )
        return entry

    entry["execution_mode"] = "recomputed"
    for model_key in MODEL_KEYS:
        model_entry = e9_module.repeated_nested_cv(eval_df, model_key, config_spec["name"])
        scores, cfg = e12_module.full_fit_ranking(eval_df, model_key)
        model_entry["full_fit_rank_metrics"] = {"config": cfg, **e12_module.rank_metrics(eval_df, scores)}
        entry[model_key] = model_entry
        print(
            f"Completed {config_spec['name']}/{model_key}: "
            f"AUROC={fmt_mean_sd(model_entry['auroc_repeat_mean'], model_entry['auroc_repeat_sd'])} "
            f"P@10={fmt_float(model_entry['full_fit_rank_metrics']['top10']['precision'])}",
            flush=True,
        )
    return entry


def build_table_rows(configuration_results: dict[str, dict]) -> list[dict]:
    full_metrics = {
        model_key: configuration_results["full_model"][model_key]["auroc_repeat_mean"]
        for model_key in MODEL_KEYS
    }

    rows = []
    for config_name, entry in configuration_results.items():
        row = {
            "configuration": config_name,
            "source": entry["source_label"],
            "n_features": entry["n_features"],
        }
        deltas = []
        for model_key in MODEL_KEYS:
            model_entry = entry[model_key]
            delta = float(model_entry["auroc_repeat_mean"] - full_metrics[model_key])
            p_at_10 = model_entry["full_fit_rank_metrics"]["top10"]["precision"]
            row[model_key] = {
                "auroc_repeat_mean": model_entry["auroc_repeat_mean"],
                "auroc_repeat_sd": model_entry["auroc_repeat_sd"],
                "delta_vs_full": delta,
                "p_at_10": p_at_10,
            }
            deltas.append(delta)
        row["mean_delta_vs_full_across_models"] = float(sum(deltas) / len(deltas))
        rows.append(row)

    full_row = next(row for row in rows if row["configuration"] == "full_model")
    other_rows = [row for row in rows if row["configuration"] != "full_model"]
    other_rows.sort(key=lambda row: (row["mean_delta_vs_full_across_models"], row["configuration"]))
    return [full_row] + other_rows


def fmt_float(value) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.3f}"


def fmt_signed(value) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:+.3f}"


def fmt_mean_sd(mean_value, sd_value) -> str:
    return f"{fmt_float(mean_value)} +/- {fmt_float(sd_value)}"


def print_summary_table(rows: list[dict], prior_only_baseline_reference: dict) -> None:
    print("\nStrict Fold C deployed-feature block ablation", flush=True)
    print(
        "full_model is pinned first; remaining rows are sorted by mean delta vs full across xgb and logistic.",
        flush=True,
    )
    print(
        f"{'configuration':<22} {'source':<12} {'nfeat':<5} "
        f"{'xgb repeat +/- sd':<18} {'xgb d':<8} {'xgb p10':<8} "
        f"{'logit repeat +/- sd':<18} {'logit d':<8} {'logit p10':<8}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['configuration']:<22} {row['source']:<12} {row['n_features']:<5} "
            f"{fmt_mean_sd(row['xgb']['auroc_repeat_mean'], row['xgb']['auroc_repeat_sd']):<18} "
            f"{fmt_signed(row['xgb']['delta_vs_full']):<8} "
            f"{fmt_float(row['xgb']['p_at_10']):<8} "
            f"{fmt_mean_sd(row['logistic']['auroc_repeat_mean'], row['logistic']['auroc_repeat_sd']):<18} "
            f"{fmt_signed(row['logistic']['delta_vs_full']):<8} "
            f"{fmt_float(row['logistic']['p_at_10']):<8}",
            flush=True,
        )

    print(
        "\nReuse status: full_model <- E12 strict Fold C; temporal_only <- E19 strict Fold C; "
        "all block removals plus prior_only were recomputed here.",
        flush=True,
    )
    print(
        f"Cached prior-only baseline read from E12/E19: "
        f"AUROC={fmt_float(prior_only_baseline_reference['strict_fold_c_baseline']['auroc'])} "
        f"AUPRC={fmt_float(prior_only_baseline_reference['strict_fold_c_baseline']['auprc'])}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    rc.ensure_results_dir()
    start = time.time()
    print(f"[{rc.utc_timestamp()}] E21 deployed block ablation starting", flush=True)

    import rev_exp_e9_nested_tuning as e9
    import rev_exp_e12_strict_windows as e12

    feature_block_audit = validate_feature_blocks()
    configuration_specs = build_configuration_specs()
    configuration_audit = validate_configuration_specs(configuration_specs)

    e12_payload = load_json(E12_PATH)
    e19_payload = load_json(E19_PATH)
    reuse_validations = {
        "e12": validate_e12_reference(e12_payload, e12),
        "e19": validate_e19_reference(e19_payload, e12),
    }
    prior_only_baseline_reference = load_prior_only_baseline_reference(e12_payload, e19_payload)

    eval_df, summary = e12.build_eval(e12.STRICT_SPECS[STRICT_FOLD_C_KEY])
    live_summary_audit = assert_strict_fold_c_summary(summary, "Live strict Fold C build")
    print(
        f"Strict Fold C built via E12: n={summary['n_genes']} pos={summary['n_positives']} "
        f"positives={summary['positives_in_eval']}",
        flush=True,
    )

    configuration_results = {}
    for config_spec in configuration_specs:
        configuration_results[config_spec["name"]] = evaluate_configuration(
            config_spec=config_spec,
            eval_df=eval_df,
            e9_module=e9,
            e12_module=e12,
            reuse_payloads={"e12": e12_payload, "e19": e19_payload},
            reuse_validations=reuse_validations,
            prior_only_baseline_reference=prior_only_baseline_reference,
            check_only=args.check_only,
        )

    if args.check_only:
        print("All E21 preflight checks passed; no nested CV was executed.", flush=True)
        return

    table_rows = build_table_rows(configuration_results)
    payload = rc.provenance(
        SCRIPT_NAME,
        parameters={
            "benchmark_key": STRICT_FOLD_C_KEY,
            "strict_fold_c_spec": copy.deepcopy(e12.STRICT_SPECS[STRICT_FOLD_C_KEY]),
            "expected_strict_fold_c_counts": copy.deepcopy(EXPECTED_STRICT_FOLD_C),
            "same_nested_cv_as_e12": True,
            "n_outer": e12.N_OUTER,
            "n_inner": e12.N_INNER,
            "outer_seeds": e12.OUTER_SEEDS,
            "inner_seed": e12.INNER_SEED,
            "xgb_grid": e12.XGB_GRID,
            "logistic_grid": e12.LOGISTIC_GRID,
            "deployed_full_features": list(FULL_FEATURES),
            "feature_blocks": copy.deepcopy(FEATURE_BLOCKS),
            "configuration_order": [spec["name"] for spec in configuration_specs],
            "reuse_policy": {
                "full_model": f"reused strict Fold C model rows from {E12_PATH.name}",
                "temporal_only": f"reused strict Fold C model rows from {E19_PATH.name}",
                "prior_only": "recomputed because the cached JSONs only contain a baseline, not an identical nested-CV model row",
            },
            "table_sort_policy": "full_model pinned first; remaining rows sorted by mean delta vs full across xgb/logistic ascending",
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_e9_nested_tuning.py",
            "rev_exp_e12_strict_windows.py",
            "rev_exp_e19_no_prior_ablation.py",
            "results/rev_exp_e12_strict_windows.json",
            "results/rev_exp_e19_no_prior_ablation.json",
            "paper/supplementary_table_s1.tex",
        ],
    )
    payload["python_executable"] = sys.executable
    payload["strict_fold_c_summary"] = summary
    payload["strict_fold_c_summary_audit"] = live_summary_audit
    payload["feature_block_partition_audit"] = feature_block_audit
    payload["configuration_audit"] = configuration_audit
    payload["reuse_source_validation"] = {
        "full_model": reuse_validations["e12"],
        "temporal_only": reuse_validations["e19"],
        "prior_only_json_baseline": prior_only_baseline_reference,
    }
    payload["configurations"] = configuration_results
    payload["reused_configurations"] = [
        name for name, entry in configuration_results.items() if entry["execution_mode"] == "reused"
    ]
    payload["recomputed_configurations"] = [
        name for name, entry in configuration_results.items() if entry["execution_mode"] == "recomputed"
    ]
    payload["table_rows"] = table_rows
    payload["runtime_seconds"] = float(time.time() - start)

    rc.write_json(OUT_PATH, payload)
    print_summary_table(table_rows, prior_only_baseline_reference)
    print(f"\nWrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
