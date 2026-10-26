#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import math
import sys
import uuid
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "rev_exp_e16_prespecified_lr010"

P3_REPRO_JSON = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr005_repro.json"
P3_REPRO_PNG = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr005_repro_heatmap.png"
P3_REPRO_LOG = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr005_repro.log"

P3_LR010_JSON = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr010.json"
P3_LR010_PNG = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr010_heatmap.png"
P3_LR010_LOG = RESULTS_DIR / "nm_exp_p3_temporal_robustness_lr010.log"

P1_LR010_JSON = RESULTS_DIR / "nm_exp_p1_prior_control_lr010.json"
P1_LR010_PNG = RESULTS_DIR / "nm_exp_p1_prior_control_lr010.png"
P1_LR010_LOG = RESULTS_DIR / "nm_exp_p1_prior_control_lr010.log"

E14_LR010_JSON = RESULTS_DIR / "rev_exp_e14_strict_permutation_lr010.json"
E14_LR010_LOG = RESULTS_DIR / "rev_exp_e14_strict_permutation_lr010.log"

SUMMARY_JSON = RESULTS_DIR / "summary.json"

P3_CLOUD_JSON = ROOT / "results" / "rerun_realyear_cloud" / "nm_exp_p3_temporal_robustness.json"
E14_ORIG_JSON = ROOT / "results" / "rev_exp_e14_strict_permutation.json"

EXPECTED_REPRO = {
    "stratified_auroc_mean": 0.8262512885649068,
    "stratified_auroc_std": 0.08402519142037017,
}


class ExperimentError(RuntimeError):
    pass


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


@contextlib.contextmanager
def patched_attr(obj, name: str, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextlib.contextmanager
def captured_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            yield


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded4(value: float) -> float:
    return float(f"{value:.4f}")


def load_runtime_versions() -> dict[str, str]:
    import numpy
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def patch_lr_on_builder(builder, lr: float):
    def patched_builder(*args, **kwargs):
        model = builder(*args, **kwargs)
        model.set_params(learning_rate=lr)
        return model

    return patched_builder


def set_module_outputs(module: ModuleType, json_path: Path, plot_path: Path | None) -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(patched_attr(module, "RESULTS_DIR", RESULTS_DIR))
    stack.enter_context(patched_attr(module, "OUTPUT_JSON", json_path))
    if plot_path is not None and hasattr(module, "OUTPUT_PLOT"):
        stack.enter_context(patched_attr(module, "OUTPUT_PLOT", plot_path))
    return stack


def run_nm_module(module_name: str, lr: float, json_path: Path, plot_path: Path | None, log_path: Path) -> dict:
    module = importlib.import_module(module_name)
    builder_name = "make_xgb"
    original_builder = getattr(module, builder_name)
    patched_builder = patch_lr_on_builder(original_builder, lr)

    with set_module_outputs(module, json_path, plot_path):
        with patched_attr(module, builder_name, patched_builder):
            with captured_output(log_path):
                module.main()

    if not json_path.exists():
        raise ExperimentError(f"{module_name} did not produce {json_path}")
    return load_json(json_path)


def assert_reproducibility(result: dict, reference: dict) -> dict:
    actual = result["sub_experiment_1_study_blocked_split"]
    ref = reference["sub_experiment_1_study_blocked_split"]

    exact_fields = ["stratified_auroc_mean", "stratified_auroc_std"]
    for field in exact_fields:
        expected = EXPECTED_REPRO[field]
        if abs(actual[field] - expected) > 1e-3:
            raise ExperimentError(
                f"Reproducibility check failed for {field}: expected {expected:.15f}, got {actual[field]:.15f}"
            )
        if abs(actual[field] - ref[field]) > 1e-3:
            raise ExperimentError(
                f"Reproducibility check failed against cloud rerun for {field}: "
                f"expected {ref[field]:.15f}, got {actual[field]:.15f}"
            )

    if rounded4(actual["stratified_auroc_mean"]) != 0.8263:
        raise ExperimentError(
            f"Rounded stratified AUROC mismatch: expected 0.8263, got {actual['stratified_auroc_mean']:.4f}"
        )
    if rounded4(actual["stratified_auroc_std"]) != 0.0840:
        raise ExperimentError(
            f"Rounded stratified AUROC SD mismatch: expected 0.0840, got {actual['stratified_auroc_std']:.4f}"
        )
    return {
        "passed": True,
        "reference_cloud_json": str(P3_CLOUD_JSON),
        "local_json": str(P3_REPRO_JSON),
        "stratified_auroc_mean": actual["stratified_auroc_mean"],
        "stratified_auroc_std": actual["stratified_auroc_std"],
        "study_blocked_cloud_auroc_mean": ref["study_blocked_auroc_mean"],
        "study_blocked_cloud_auroc_std": ref["study_blocked_auroc_std"],
        "study_blocked_local_auroc_mean": actual["study_blocked_auroc_mean"],
        "study_blocked_local_auroc_std": actual["study_blocked_auroc_std"],
        "study_blocked_delta_local_minus_cloud": actual["study_blocked_auroc_mean"] - ref["study_blocked_auroc_mean"],
        "notes": (
            "GroupKFold with many equal-size groups is platform-sensitive because fold assignment "
            "depends on non-stable ordering among singleton groups; stratified metrics reproduced exactly."
        ),
    }


def run_e14_lr010(lr: float, output_json: Path, log_path: Path) -> dict:
    rev_exp_common = importlib.import_module("rev_exp_common")
    original_make_fixed_xgb = rev_exp_common.make_fixed_xgb

    def patched_make_fixed_xgb(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["learning_rate"] = lr
        return original_make_fixed_xgb(*args, **kwargs)

    original_write_text = Path.write_text

    def patched_write_text(self: Path, data: str, *args, **kwargs):
        if self == E14_ORIG_JSON:
            payload = json.loads(data)
            payload["script_name"] = "rev_exp_e16_prespecified_lr010.py"
            payload.setdefault("parameters", {})["model"] = "pre-specified XGBoost 200/4/0.1"
            return original_write_text(output_json, json.dumps(payload, indent=2), *args, **kwargs)
        return original_write_text(self, data, *args, **kwargs)

    module_name = f"_rev_exp_e14_strict_permutation_e16_{uuid.uuid4().hex}"
    module_path = ROOT / "rev_exp_e14_strict_permutation.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ExperimentError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)

    try:
        Path.write_text = patched_write_text
        with patched_attr(rev_exp_common, "make_fixed_xgb", patched_make_fixed_xgb):
            with captured_output(log_path):
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
    finally:
        Path.write_text = original_write_text
        sys.modules.pop(module_name, None)

    if not output_json.exists():
        raise ExperimentError(f"E14 lr=0.1 run did not produce {output_json}")
    return load_json(output_json)


def build_summary(p3_repro: dict, p3_lr010: dict, p1_lr010: dict, e14_lr010: dict) -> dict:
    p3_repro_block = p3_repro["sub_experiment_1_study_blocked_split"]
    p3_block = p3_lr010["sub_experiment_1_study_blocked_split"]
    p3_labels = p3_lr010["sub_experiment_3_alternative_label_definitions"]
    p3_fold_d = p3_lr010["sub_experiment_4_fold_d_validation"]
    p1_full = p1_lr010["full_model"]
    p1_perm = p1_lr010["permutation_test"]
    p1_drop = p1_lr010["drop_classic_markers"]["full"]
    e14_null = e14_lr010["null"]

    guards = {
        "internal_permutation_p_lt_0_001": p1_perm["empirical_p_value"] < 0.001,
        "internal_null_max_below_observed": p1_perm["null_distribution_stats"]["max"] < p1_perm["observed_auroc"],
        "study_blocked_within_0_05": abs(p3_block["study_blocked_auroc_mean"] - p3_block["stratified_auroc_mean"]) < 0.05,
        "drop_classic_above_baseline": p1_drop["mean_auroc"] > p1_full["mean_auroc"],
        "e14_observed_above_null_p99": e14_lr010["observed_cv_auroc"] > e14_null["p99"],
        "internal_auroc_within_0_1_of_external_fold_c_range": p1_full["mean_auroc"] >= 0.882 - 0.1,
    }
    failed = [name for name, passed in guards.items() if not passed]
    if failed:
        raise ExperimentError(f"Qualitative guard failure(s): {', '.join(failed)}")

    summary = {
        "python_executable": sys.executable,
        "versions": load_runtime_versions(),
        "results_dir": str(RESULTS_DIR),
        "reproducibility_check": assert_reproducibility(p3_repro, load_json(P3_CLOUD_JSON)),
        "notes": (
            "GroupKFold is platform-sensitive here because the implementation orders many equal-size singleton "
            "groups non-stably across numpy/scikit-learn builds. The stratified five-fold metrics reproduced "
            "the cloud rerun, so the data/year pipeline is treated as matched."
        ),
        "qualitative_guards": {"passed": True, "checks": guards},
        "internal_fold_2021": {
            "n_eval_genes": p1_lr010["n_eval_genes"],
            "n_positives": p1_lr010["n_positives"],
            "auroc_mean": p1_full["mean_auroc"],
            "auroc_std": p1_full["std_auroc"],
            "permutation_observed_auroc": p1_perm["observed_auroc"],
            "permutation_empirical_p": p1_perm["empirical_p_value"],
            "null_max": p1_perm["null_distribution_stats"]["max"],
            "null_p95": p1_perm["null_distribution_stats"]["p95"],
            "drop_classic_auroc_mean": p1_drop["mean_auroc"],
            "drop_classic_auroc_std": p1_drop["std_auroc"],
        },
        "study_blocked_and_thresholds_lr010": {
            "study_blocked_auroc_mean": p3_block["study_blocked_auroc_mean"],
            "study_blocked_auroc_std": p3_block["study_blocked_auroc_std"],
            "stratified_auroc_mean": p3_block["stratified_auroc_mean"],
            "stratified_auroc_std": p3_block["stratified_auroc_std"],
            "delta_vs_stratified": p3_block["delta"],
            "threshold_0_90_auroc_mean": p3_labels["definition_A"]["def_A_auroc"],
            "threshold_0_90_auroc_std": p3_labels["definition_A"]["def_A_auroc_std"],
            "threshold_0_88_auroc_mean": p3_labels["definition_B"]["def_B_auroc"],
            "threshold_0_88_auroc_std": p3_labels["definition_B"]["def_B_auroc_std"],
            "threshold_0_80_auroc_mean": p3_labels["definition_C"]["def_C_auroc"],
            "threshold_0_80_auroc_std": p3_labels["definition_C"]["def_C_auroc_std"],
            "fold_d_n_eval_genes": p3_fold_d["fold_d_n_eval_genes"],
            "fold_d_n_positives": p3_fold_d["fold_d_n_positives"],
            "fold_d_auroc_mean": p3_fold_d["fold_d_auroc_mean"],
            "fold_d_auroc_std": p3_fold_d["fold_d_auroc_std"],
        },
        "strict_fold_c_permutation_lr010": {
            "n_genes": e14_lr010["summary"]["n_genes"],
            "n_positives": e14_lr010["summary"]["n_positives"],
            "observed_cv_auroc": e14_lr010["observed_cv_auroc"],
            "null_mean": e14_null["mean"],
            "null_std": e14_null["std"],
            "null_p95": e14_null["p95"],
            "null_p99": e14_null["p99"],
            "null_max": e14_null["max"],
            "empirical_p": e14_lr010["empirical_p"],
        },
        "artifacts": {
            "p3_lr005_repro_json": str(P3_REPRO_JSON),
            "p3_lr005_repro_png": str(P3_REPRO_PNG),
            "p3_lr005_repro_log": str(P3_REPRO_LOG),
            "p3_lr010_json": str(P3_LR010_JSON),
            "p3_lr010_png": str(P3_LR010_PNG),
            "p3_lr010_log": str(P3_LR010_LOG),
            "p1_lr010_json": str(P1_LR010_JSON),
            "p1_lr010_png": str(P1_LR010_PNG),
            "p1_lr010_log": str(P1_LR010_LOG),
            "e14_lr010_json": str(E14_LR010_JSON),
            "e14_lr010_log": str(E14_LR010_LOG),
            "summary_json": str(SUMMARY_JSON),
        },
        "tex_ready": {
            "internal_2021_auroc_mean_3dp": f"{p1_full['mean_auroc']:.3f}",
            "internal_2021_auroc_std_3dp": f"{p1_full['std_auroc']:.3f}",
            "internal_2022_auroc_mean_3dp": f"{p3_fold_d['fold_d_auroc_mean']:.3f}",
            "internal_2022_auroc_std_3dp": f"{p3_fold_d['fold_d_auroc_std']:.3f}",
            "internal_null_max_3dp": f"{p1_perm['null_distribution_stats']['max']:.3f}",
            "internal_null_p95_3dp": f"{p1_perm['null_distribution_stats']['p95']:.3f}",
            "study_blocked_auroc_3dp": f"{p3_block['study_blocked_auroc_mean']:.3f}",
            "study_blocked_delta_3dp": f"{abs(p3_block['delta']):.3f}",
            "threshold_0_88_auroc_3dp": f"{p3_labels['definition_B']['def_B_auroc']:.3f}",
            "threshold_0_90_auroc_3dp": f"{p3_labels['definition_A']['def_A_auroc']:.3f}",
            "threshold_0_80_auroc_3dp": f"{p3_labels['definition_C']['def_C_auroc']:.3f}",
            "drop_classic_auroc_3dp": f"{p1_drop['mean_auroc']:.3f}",
            "e14_observed_3dp": f"{e14_lr010['observed_cv_auroc']:.3f}",
            "e14_null_mean_3dp": f"{e14_null['mean']:.3f}",
            "e14_null_std_3dp": f"{e14_null['std']:.3f}",
            "e14_null_p95_3dp": f"{e14_null['p95']:.3f}",
            "e14_null_p99_3dp": f"{e14_null['p99']:.3f}",
            "e14_null_max_3dp": f"{e14_null['max']:.3f}",
            "e14_empirical_p_3dp": f"{e14_lr010['empirical_p']:.3f}",
        },
    }
    return to_jsonable(summary)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))

    print(f"[E16] results dir: {RESULTS_DIR}", flush=True)
    print("[E16] running p3 reproducibility check at learning_rate=0.05", flush=True)
    p3_repro = run_nm_module(
        module_name="nm_exp_p3_temporal_robustness",
        lr=0.05,
        json_path=P3_REPRO_JSON,
        plot_path=P3_REPRO_PNG,
        log_path=P3_REPRO_LOG,
    )
    repro = assert_reproducibility(p3_repro, load_json(P3_CLOUD_JSON))
    print(json.dumps({"reproducibility_check": repro}, indent=2), flush=True)

    print("[E16] running p3 at learning_rate=0.1", flush=True)
    p3_lr010 = run_nm_module(
        module_name="nm_exp_p3_temporal_robustness",
        lr=0.1,
        json_path=P3_LR010_JSON,
        plot_path=P3_LR010_PNG,
        log_path=P3_LR010_LOG,
    )

    print("[E16] running p1 at learning_rate=0.1", flush=True)
    p1_lr010 = run_nm_module(
        module_name="nm_exp_p1_prior_control",
        lr=0.1,
        json_path=P1_LR010_JSON,
        plot_path=P1_LR010_PNG,
        log_path=P1_LR010_LOG,
    )

    print("[E16] running E14 strict Fold C permutation at learning_rate=0.1", flush=True)
    e14_lr010 = run_e14_lr010(
        lr=0.1,
        output_json=E14_LR010_JSON,
        log_path=E14_LR010_LOG,
    )

    summary = build_summary(p3_repro, p3_lr010, p1_lr010, e14_lr010)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[E16] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
