#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_common as rc

SCRIPT_NAME = Path(__file__).name
RESULTS_DIR = rc.BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e18_uniprot_prior_probe.json"
OLD_E12_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"

BENCHMARK_KEYS = ("strict_fold_c", "strict_fold_d", "within_secreted_strict")
MODEL_KEYS = ("xgb", "logistic")
BASELINE_KEYS = ("secretion_status", "publication_count", "max_signature_auc", "layer_count")
PROBE_CUTOFF_YEAR = 2020
EXPECTED_SCORED_UNIVERSE_SIZE = 2843
EXPECTED_OLD_SECRETED_IN_UNIVERSE = 35
EXPECTED_NEW_SECRETED_IN_UNIVERSE = 246


def load_cached_annotation_bundle() -> tuple[dict, set[str], Path]:
    import rev_exp_annotations as annotations

    bundle_path = annotations.OUTPUT_JSON
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing cached annotation bundle: {bundle_path}. "
            "This probe will not call build_annotation_bundle() because that would rewrite existing annotation caches."
        )
    payload = json.loads(bundle_path.read_text())
    provenance = payload.get("provenance", {})
    if provenance.get("script_name") != "rev_exp_annotations.py":
        raise RuntimeError(f"Unexpected annotation provenance in {bundle_path}: {provenance}")
    bundle = payload.get("annotation_bundle")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"annotation_bundle missing from {bundle_path}")
    kw_genes = set(bundle["uniprot"]["secreted_kw_genes"])
    return bundle, kw_genes, bundle_path


def load_old_e12_results() -> dict:
    if not OLD_E12_PATH.exists():
        raise FileNotFoundError(f"Missing reference E12 results: {OLD_E12_PATH}")
    payload = json.loads(OLD_E12_PATH.read_text())
    missing = [key for key in BENCHMARK_KEYS if key not in payload.get("benchmarks", {})]
    if missing:
        raise RuntimeError(f"Reference E12 results are missing benchmarks: {missing}")
    return payload


def get_secreted_gene_list(feature_df) -> list[str]:
    return sorted(feature_df.loc[feature_df["is_secreted"] == 1, "gene"].astype(str).tolist())


def apply_uniprot_secreted_prior() -> tuple[dict, set[str], Path, dict]:
    bundle, kw_genes, bundle_path = load_cached_annotation_bundle()
    scored_universe = bundle["scored_universe"]["genes"]
    scored_universe_set = set(scored_universe)
    kw_overlap = sorted(scored_universe_set & kw_genes)

    if len(scored_universe) != EXPECTED_SCORED_UNIVERSE_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_SCORED_UNIVERSE_SIZE} genes in the scored universe, found {len(scored_universe)}."
        )
    if len(kw_overlap) != EXPECTED_NEW_SECRETED_IN_UNIVERSE:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_SECRETED_IN_UNIVERSE} UniProt KW-0964 genes in the scored universe, found {len(kw_overlap)}."
        )

    old_secreted_source = sorted(set(rc.SECRETED_GENES))
    old_ctx = rc.build_feature_context(cutoff_year=PROBE_CUTOFF_YEAR)
    old_secreted_genes = get_secreted_gene_list(old_ctx["feature_df"])
    old_secreted_count = len(old_secreted_genes)

    if old_secreted_count != EXPECTED_OLD_SECRETED_IN_UNIVERSE:
        raise RuntimeError(
            f"Expected {EXPECTED_OLD_SECRETED_IN_UNIVERSE} old secreted genes in the scored universe, found {old_secreted_count}."
        )

    rc.SECRETED_GENES = kw_genes
    new_ctx = rc.build_feature_context(cutoff_year=PROBE_CUTOFF_YEAR)
    new_secreted_genes = get_secreted_gene_list(new_ctx["feature_df"])
    new_secreted_count = len(new_secreted_genes)

    if old_ctx["feature_df"]["is_secreted"].equals(new_ctx["feature_df"]["is_secreted"]):
        raise RuntimeError(
            "Rebinding rc.SECRETED_GENES did not change build_feature_context(). "
            "Aborting instead of producing misleading numbers."
        )
    if new_secreted_count != EXPECTED_NEW_SECRETED_IN_UNIVERSE:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_SECRETED_IN_UNIVERSE} cutoff-{PROBE_CUTOFF_YEAR} secreted genes after rebinding, "
            f"found {new_secreted_count}."
        )
    if set(new_secreted_genes) != set(kw_overlap):
        raise RuntimeError(
            "After rebinding, the genes flagged by is_secreted do not match the UniProt KW-0964 overlap with the scored universe."
        )

    print(f"[{rc.utc_timestamp()}] Loaded cached annotation bundle from {bundle_path}", flush=True)
    print(
        f"Old secreted prior: source_size={len(old_secreted_source)} overlap_with_2843={old_secreted_count}",
        flush=True,
    )
    print(
        f"UniProt KW-0964 prior: source_size={len(kw_genes)} overlap_with_2843={len(kw_overlap)}",
        flush=True,
    )
    print(
        f"cutoff={PROBE_CUTOFF_YEAR} is_secreted=1 count: old={old_secreted_count} new={new_secreted_count}",
        flush=True,
    )

    audit = {
        "annotation_bundle_path": str(bundle_path),
        "old_secreted_source_size": int(len(old_secreted_source)),
        "new_secreted_source_size": int(len(kw_genes)),
        "scored_universe_size": int(len(scored_universe)),
        "old_secreted_in_scored_universe": int(old_secreted_count),
        "new_secreted_in_scored_universe": int(new_secreted_count),
        "old_secreted_genes_in_scored_universe": old_secreted_genes,
        "new_secreted_genes_in_scored_universe": new_secreted_genes,
        "new_secreted_genes_match_uniprot_overlap": True,
        "new_minus_old_count": int(len(set(new_secreted_genes) - set(old_secreted_genes))),
        "old_minus_new_count": int(len(set(old_secreted_genes) - set(new_secreted_genes))),
    }
    return bundle, kw_genes, bundle_path, audit


def collect_comparison(old_payload: dict, new_payload: dict) -> dict:
    comparison = {
        "reference_result_path": str(OLD_E12_PATH),
        "benchmark_model_auroc": {},
        "baseline_reports": {},
        "full_fit_rankings": {},
    }

    for benchmark in BENCHMARK_KEYS:
        comparison["benchmark_model_auroc"][benchmark] = {}
        comparison["baseline_reports"][benchmark] = {}
        for model in MODEL_KEYS:
            old_model = old_payload["benchmarks"][benchmark][model]
            new_model = new_payload["benchmarks"][benchmark][model]
            comparison["benchmark_model_auroc"][benchmark][model] = {
                "old": {
                    "auroc_repeat_mean": old_model["auroc_repeat_mean"],
                    "auroc_repeat_sd": old_model["auroc_repeat_sd"],
                    "auroc_oof_mean": old_model["auroc_oof_mean"],
                },
                "new": {
                    "auroc_repeat_mean": new_model["auroc_repeat_mean"],
                    "auroc_repeat_sd": new_model["auroc_repeat_sd"],
                    "auroc_oof_mean": new_model["auroc_oof_mean"],
                },
            }
        for baseline in BASELINE_KEYS:
            comparison["baseline_reports"][benchmark][baseline] = {
                "old": old_payload["benchmarks"][benchmark]["baselines"][baseline],
                "new": new_payload["benchmarks"][benchmark]["baselines"][baseline],
            }

    for benchmark in ("strict_fold_c", "within_secreted_strict"):
        comparison["full_fit_rankings"][benchmark] = {}
        for model in MODEL_KEYS:
            old_rank = old_payload["benchmarks"][benchmark][model]["full_fit_rank_metrics"]
            new_rank = new_payload["benchmarks"][benchmark][model]["full_fit_rank_metrics"]
            comparison["full_fit_rankings"][benchmark][model] = {
                "old": {
                    "p_at_5": old_rank["top5"]["precision"],
                    "p_at_10": old_rank["top10"]["precision"],
                    "ef_at_10": old_rank["top10"]["enrichment_factor"],
                    "top_10_genes": old_rank["top10"]["top_genes"],
                },
                "new": {
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
    return f"{float(value):.3f}"


def fmt_mean_sd(mean_value, sd_value) -> str:
    return f"{fmt_float(mean_value)} +/- {fmt_float(sd_value)}"


def fmt_baseline(report: dict) -> str:
    parts = [f"AUROC={fmt_float(report.get('auroc'))}"]
    if "auprc" in report:
        parts.append(f"AUPRC={fmt_float(report['auprc'])}")
    if "note" in report:
        parts.append(str(report["note"]))
    return "; ".join(parts)


def print_comparison_table(comparison: dict) -> None:
    print("\nBenchmark x model AUROC comparison", flush=True)
    print(
        f"{'benchmark':<24} {'model':<10} {'OLD repeat +/- sd':<20} {'NEW repeat +/- sd':<20} {'OLD oof':<10} {'NEW oof':<10}",
        flush=True,
    )
    for benchmark in BENCHMARK_KEYS:
        for model in MODEL_KEYS:
            entry = comparison["benchmark_model_auroc"][benchmark][model]
            print(
                f"{benchmark:<24} {model:<10} "
                f"{fmt_mean_sd(entry['old']['auroc_repeat_mean'], entry['old']['auroc_repeat_sd']):<20} "
                f"{fmt_mean_sd(entry['new']['auroc_repeat_mean'], entry['new']['auroc_repeat_sd']):<20} "
                f"{fmt_float(entry['old']['auroc_oof_mean']):<10} {fmt_float(entry['new']['auroc_oof_mean']):<10}",
                flush=True,
            )

    print("\nBaseline comparison", flush=True)
    for benchmark in BENCHMARK_KEYS:
        for baseline in BASELINE_KEYS:
            entry = comparison["baseline_reports"][benchmark][baseline]
            print(
                f"{benchmark} | {baseline} | OLD {fmt_baseline(entry['old'])} | NEW {fmt_baseline(entry['new'])}",
                flush=True,
            )

    print("\nFull-fit ranking comparison", flush=True)
    for benchmark in ("strict_fold_c", "within_secreted_strict"):
        for model in MODEL_KEYS:
            entry = comparison["full_fit_rankings"][benchmark][model]
            print(
                f"{benchmark}/{model}: "
                f"OLD P@5={fmt_float(entry['old']['p_at_5'])} NEW P@5={fmt_float(entry['new']['p_at_5'])} | "
                f"OLD P@10={fmt_float(entry['old']['p_at_10'])} NEW P@10={fmt_float(entry['new']['p_at_10'])} | "
                f"OLD EF@10={fmt_float(entry['old']['ef_at_10'])} NEW EF@10={fmt_float(entry['new']['ef_at_10'])}",
                flush=True,
            )
            print(f"  OLD top10: {', '.join(entry['old']['top_10_genes'])}", flush=True)
            print(f"  NEW top10: {', '.join(entry['new']['top_10_genes'])}", flush=True)


def main() -> None:
    rc.ensure_results_dir()
    start = time.time()
    print(f"[{rc.utc_timestamp()}] E18 UniProt prior probe starting", flush=True)

    old_payload = load_old_e12_results()
    bundle, kw_genes, bundle_path, rebinding_audit = apply_uniprot_secreted_prior()

    import rev_exp_e12_strict_windows as e12

    payload = rc.provenance(
        SCRIPT_NAME,
        parameters={
            "reference_e12_result_path": str(OLD_E12_PATH),
            "annotation_bundle_path": str(bundle_path),
            "annotation_source_mode": (
                "read-only load from results/rev_exp_annotations.json because "
                "build_annotation_bundle() rewrites cached annotation artifacts"
            ),
            "secreted_prior_old_source": "nm_exp_p1_prior_control.SECRETED_GENES",
            "secreted_prior_new_source": "rev_exp_annotations.annotation_bundle.uniprot.secreted_kw_genes",
            "strict_specs": e12.STRICT_SPECS,
            "within_secreted_spec": e12.WITHIN_SECRETED_SPEC,
            "same_nested_cv_as_e12": True,
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_e12_strict_windows.py",
            "rev_exp_e9_nested_tuning.py",
            "results/rev_exp_e12_strict_windows.json",
            "results/rev_exp_annotations.json",
            "paper/supplementary_table_s1.tex",
        ],
    )
    payload["python_executable"] = sys.executable
    payload["secreted_prior_rebinding_audit"] = rebinding_audit
    payload["annotation_bundle_manifest"] = bundle["uniprot"]["manifest"]
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
    is_secreted_unique_values = sorted(int(value) for value in ws_df["is_secreted"].drop_duplicates().tolist())
    ws_entry["is_secreted_diagnostic"] = {
        "nunique": int(ws_df["is_secreted"].nunique()),
        "unique_values": is_secreted_unique_values,
        "secretion_status_baseline_report": ws_entry["baselines"]["secretion_status"],
    }
    print(
        "within_secreted_strict is_secreted diagnostic: "
        f"nunique={ws_entry['is_secreted_diagnostic']['nunique']} "
        f"values={ws_entry['is_secreted_diagnostic']['unique_values']} "
        f"baseline={ws_entry['is_secreted_diagnostic']['secretion_status_baseline_report']}",
        flush=True,
    )
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

    payload["comparison_to_rev_exp_e12_strict_windows"] = collect_comparison(old_payload, payload)
    payload["runtime_seconds"] = float(time.time() - start)

    rc.write_json(OUT_PATH, payload)
    print_comparison_table(payload["comparison_to_rev_exp_e12_strict_windows"])
    print(f"\nWrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
