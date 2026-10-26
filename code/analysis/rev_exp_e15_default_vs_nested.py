from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning, module="nm_exp5_llm_vs_ml_benchmark")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.model_selection._split",
    message=r"The least populated class in y has only 4 members, which is less than n_splits=5\.",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_exp_annotations import build_annotation_bundle
from rev_exp_common import BASE_DIR, FULL_FEATURES, make_fixed_xgb, provenance
from rev_exp_e9_nested_tuning import N_OUTER, OUTER_SEEDS, make_logistic_from_config
from rev_exp_e12_strict_windows import STRICT_SPECS, WITHIN_SECRETED_SPEC, build_eval

RESULTS_DIR = BASE_DIR / "results"
E12_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"
OUT_PATH = RESULTS_DIR / "rev_exp_e15_default_vs_nested.json"

BENCHMARK_KEYS = ["strict_fold_c", "strict_fold_d", "within_secreted_strict"]
MODEL_KINDS = ["xgb", "logistic"]
EXPECTED_SUMMARY = {
    "strict_fold_c": {"n_genes": 2403, "n_positives": 16},
    "strict_fold_d": {"n_genes": 2536, "n_positives": 8},
    "within_secreted_strict": {"n_genes": 246, "n_positives": 4},
}
DEFAULT_CONFIGS = {
    "xgb": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    "logistic": {"C": 1.0},
}


def load_e12_results() -> dict:
    return json.loads(E12_PATH.read_text())


def assert_summary(key: str, summary: dict, nested_summary: dict) -> None:
    expected = EXPECTED_SUMMARY[key]
    for field in ["n_genes", "n_positives"]:
        expected_value = expected[field]
        nested_value = int(nested_summary[field])
        actual_value = int(summary[field])
        assert nested_value == expected_value, (
            f"{key} E12 summary mismatch for {field}: expected {expected_value}, found {nested_value} "
            f"in {E12_PATH}"
        )
        assert actual_value == nested_value, (
            f"{key} rebuilt summary mismatch for {field}: expected {nested_value} from {E12_PATH}, got {actual_value}"
        )


def make_default_model(kind: str):
    if kind == "xgb":
        return make_fixed_xgb()
    if kind == "logistic":
        return make_logistic_from_config({"C": 1.0})
    raise ValueError(f"Unsupported model kind: {kind}")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def finite_sd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def evaluate_default_repeated_cv(eval_df, kind: str) -> dict:
    X = eval_df[FULL_FEATURES].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    repeats = []
    for seed in OUTER_SEEDS:
        splitter = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        oof = np.full(len(eval_df), np.nan)
        fold_aurocs = []
        for train_idx, test_idx in splitter.split(X, y):
            model = make_default_model(kind)
            model.fit(X[train_idx], y[train_idx])
            scores = model.predict_proba(X[test_idx])[:, 1]
            oof[test_idx] = scores
            y_test = y[test_idx]
            if np.unique(y_test).size < 2:
                fold_aurocs.append(float("nan"))
            else:
                fold_aurocs.append(float(roc_auc_score(y_test, scores)))
        repeats.append(
            {
                "outer_seed": seed,
                "fold_aurocs": fold_aurocs,
                "auroc_mean": finite_mean(fold_aurocs),
                "auroc_oof": float(roc_auc_score(y, oof)),
                "auprc_oof": float(average_precision_score(y, oof)),
            }
        )
    repeat_means = [repeat["auroc_mean"] for repeat in repeats]
    oof_aurocs = [repeat["auroc_oof"] for repeat in repeats]
    oof_auprcs = [repeat["auprc_oof"] for repeat in repeats]
    all_fold_aurocs = [auc for repeat in repeats for auc in repeat["fold_aurocs"]]
    return {
        "config": DEFAULT_CONFIGS[kind],
        "n_repeats": len(OUTER_SEEDS),
        "n_outer_folds": N_OUTER,
        "auroc_repeat_mean": finite_mean(repeat_means),
        "auroc_repeat_sd": finite_sd(repeat_means),
        "auroc_allfolds_mean": finite_mean(all_fold_aurocs),
        "auroc_allfolds_sd": finite_sd(all_fold_aurocs),
        "auroc_oof_mean": finite_mean(oof_aurocs),
        "auroc_oof_sd": finite_sd(oof_aurocs),
        "auprc_oof_mean": finite_mean(oof_auprcs),
        "auprc_oof_sd": finite_sd(oof_auprcs),
        "repeats": repeats,
    }


def copy_nested_aggregate(nested_entry: dict) -> dict:
    copied = {
        key: nested_entry[key]
        for key in [
            "n_repeats",
            "n_outer_folds",
            "n_inner_folds",
            "auroc_repeat_mean",
            "auroc_repeat_sd",
            "auroc_allfolds_mean",
            "auroc_allfolds_sd",
            "auroc_oof_mean",
            "auprc_oof_mean",
        ]
        if key in nested_entry
    }
    copied["auroc_oof_sd"] = finite_sd([repeat["auroc_oof"] for repeat in nested_entry["repeats"]])
    return copied


def delta(nested_value: float, default_value: float) -> float:
    if math.isnan(float(nested_value)) or math.isnan(float(default_value)):
        return float("nan")
    return float(nested_value - default_value)


def paired_fold_aurocs(default_entry: dict, nested_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    default_values = []
    nested_values = []
    assert len(default_entry["repeats"]) == len(nested_entry["repeats"]) == len(OUTER_SEEDS), "Repeat count mismatch"
    for default_repeat, nested_repeat in zip(default_entry["repeats"], nested_entry["repeats"], strict=True):
        assert default_repeat["outer_seed"] == nested_repeat["outer_seed"], (
            f"Outer seed mismatch: default {default_repeat['outer_seed']} vs nested {nested_repeat['outer_seed']}"
        )
        assert len(default_repeat["fold_aurocs"]) == len(nested_repeat["fold_aurocs"]) == N_OUTER, "Fold count mismatch"
        for default_auc, nested_auc in zip(default_repeat["fold_aurocs"], nested_repeat["fold_aurocs"], strict=True):
            default_value = float(default_auc)
            nested_value = float(nested_auc)
            if math.isnan(default_value) or math.isnan(nested_value):
                continue
            default_values.append(default_value)
            nested_values.append(nested_value)
    return np.asarray(default_values, dtype=float), np.asarray(nested_values, dtype=float)


def wilcoxon_summary(default_entry: dict, nested_entry: dict) -> dict:
    default_values, nested_values = paired_fold_aurocs(default_entry, nested_entry)
    n_pairs = int(default_values.size)
    nested_better = int(np.sum(nested_values > default_values))
    p_value = None
    if n_pairs >= 2:
        diffs = nested_values - default_values
        if not np.allclose(diffs, 0.0):
            p_value = float(wilcoxon(nested_values, default_values, alternative="two-sided").pvalue)
    return {
        "test": "wilcoxon_signed_rank_two_sided",
        "n_pairs": n_pairs,
        "p": p_value,
        "n_folds_nested_better": nested_better,
    }


def build_benchmark_inputs() -> dict[str, tuple]:
    bundle = build_annotation_bundle()
    kw_genes = set(bundle["uniprot"]["secreted_kw_genes"])
    return {
        "strict_fold_c": build_eval(STRICT_SPECS["strict_fold_c"]),
        "strict_fold_d": build_eval(STRICT_SPECS["strict_fold_d"]),
        "within_secreted_strict": build_eval(WITHIN_SECRETED_SPEC, kw_genes=kw_genes),
    }


def format_value(value) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.3f}"


def format_mean_sd(mean_value, sd_value) -> str:
    if mean_value is None or sd_value is None:
        return "NA"
    mean_float = float(mean_value)
    sd_float = float(sd_value)
    if math.isnan(mean_float) or math.isnan(sd_float):
        return "NA"
    return f"{mean_float:.3f}+/-{sd_float:.3f}"


def print_summary_table(rows: list[dict]) -> None:
    headers = [
        "benchmark",
        "model",
        "default_repeat",
        "nested_repeat",
        "default_oof",
        "nested_oof",
        "delta_repeat",
        "delta_oof",
        "wilcoxon_p",
        "n_pairs",
    ]
    rendered = [headers]
    for row in rows:
        rendered.append(
            [
                row["benchmark"],
                row["model"],
                row["default_repeat"],
                row["nested_repeat"],
                row["default_oof"],
                row["nested_oof"],
                row["delta_repeat"],
                row["delta_oof"],
                row["wilcoxon_p"],
                row["n_pairs"],
            ]
        )
    widths = [max(len(str(line[i])) for line in rendered) for i in range(len(headers))]
    for idx, line in enumerate(rendered):
        pieces = [str(value).ljust(widths[i]) for i, value in enumerate(line)]
        print("  ".join(pieces), flush=True)
        if idx == 0:
            print("  ".join("-" * width for width in widths), flush=True)


def main() -> None:
    t0 = time.time()
    nested_payload = load_e12_results()
    benchmark_inputs = build_benchmark_inputs()
    payload = provenance(
        "rev_exp_e15_default_vs_nested.py",
        parameters={
            "benchmarks": BENCHMARK_KEYS,
            "model_kinds": MODEL_KINDS,
            "outer_protocol": {
                "n_outer_folds": N_OUTER,
                "shuffle": True,
                "outer_seeds": OUTER_SEEDS,
            },
            "default_configs": DEFAULT_CONFIGS,
            "pairing": "paired by benchmark, model, outer seed order, and fold order",
            "wilcoxon": "scipy.stats.wilcoxon two-sided after dropping NaN fold pairs",
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_e9_nested_tuning.py",
            "rev_exp_e12_strict_windows.py",
            "results/rev_exp_e12_strict_windows.json",
        ],
    )
    payload["benchmarks"] = {}
    table_rows = []

    for key in BENCHMARK_KEYS:
        eval_df, summary = benchmark_inputs[key]
        nested_summary = nested_payload["benchmarks"][key]["summary"]
        assert_summary(key, summary, nested_summary)
        entry = {"summary": {"n_genes": int(summary["n_genes"]), "n_positives": int(summary["n_positives"])}}
        for kind in MODEL_KINDS:
            default_entry = evaluate_default_repeated_cv(eval_df, kind)
            nested_entry = nested_payload["benchmarks"][key][kind]
            nested_aggregate = copy_nested_aggregate(nested_entry)
            delta_repeat_mean = delta(nested_aggregate["auroc_repeat_mean"], default_entry["auroc_repeat_mean"])
            delta_oof_mean = delta(nested_aggregate["auroc_oof_mean"], default_entry["auroc_oof_mean"])
            wilcoxon_entry = wilcoxon_summary(default_entry, nested_entry)
            entry[kind] = {
                "default": default_entry,
                "nested": nested_aggregate,
                "delta_repeat_mean": delta_repeat_mean,
                "delta_oof_mean": delta_oof_mean,
                "wilcoxon": wilcoxon_entry,
            }
            table_rows.append(
                {
                    "benchmark": key,
                    "model": kind,
                    "default_repeat": format_mean_sd(
                        default_entry["auroc_repeat_mean"],
                        default_entry["auroc_repeat_sd"],
                    ),
                    "nested_repeat": format_mean_sd(
                        nested_aggregate["auroc_repeat_mean"],
                        nested_aggregate["auroc_repeat_sd"],
                    ),
                    "default_oof": format_value(default_entry["auroc_oof_mean"]),
                    "nested_oof": format_value(nested_aggregate["auroc_oof_mean"]),
                    "delta_repeat": format_value(delta_repeat_mean),
                    "delta_oof": format_value(delta_oof_mean),
                    "wilcoxon_p": format_value(wilcoxon_entry["p"]),
                    "n_pairs": str(wilcoxon_entry["n_pairs"]),
                }
            )
        payload["benchmarks"][key] = entry

    payload["runtime_seconds"] = time.time() - t0
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print_summary_table(table_rows)


if __name__ == "__main__":
    main()
