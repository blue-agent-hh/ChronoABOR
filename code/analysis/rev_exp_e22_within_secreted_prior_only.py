#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_common as rc
import rev_exp_e12_strict_windows as e12

SCRIPT_NAME = Path(__file__).name
RESULTS_DIR = rc.BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e22_within_secreted_prior_only.json"

MODEL_KEYS = ("xgb", "logistic")
PRIOR_ONLY_FEATURES = ["is_secreted"]
EXPECTED_POSITIVES = ["APOE", "BDNF", "CLU", "MAPT"]
EXPECTED_WITHIN_SECRETED = {
    "cutoff_year": 2020,
    "label_year_start": 2021,
    "label_year_end": 2025,
    "n_genes": 246,
    "n_positives": 4,
    "positives_in_eval": EXPECTED_POSITIVES,
}
PUBLISHED_FULL_FIT_PRECISION = {
    "logistic": {"p_at_5": 0.600, "p_at_10": 0.400},
    "xgb": {"p_at_5": 0.200, "p_at_10": 0.100},
}
TIE_BREAK_POLICY = (
    "E12 ranks by descending predicted probability with np.argsort(-scores, kind='stable'); "
    "ties preserve the evaluation-frame row order, which is ascending gene symbol after the feature "
    "matrix is sorted by gene and then filtered to the UniProt KW-0964 subset."
)


def load_cached_uniprot_kw_genes() -> tuple[dict, set[str], Path]:
    import rev_exp_annotations as annotations

    bundle_path = annotations.OUTPUT_JSON
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing cached annotation bundle: {bundle_path}. "
            "E22 will not call build_annotation_bundle() because that rewrites annotation caches."
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


def assert_deployed_feature_state() -> dict:
    expected = list(rc.FEATURE_COLUMNS) + PRIOR_ONLY_FEATURES
    if list(rc.FULL_FEATURES) != expected:
        raise RuntimeError(f"rev_exp_common.FULL_FEATURES mismatch: {rc.FULL_FEATURES} vs {expected}")
    if list(e12.FULL_FEATURES) != expected:
        raise RuntimeError(f"rev_exp_e12_strict_windows.FULL_FEATURES mismatch: {e12.FULL_FEATURES} vs {expected}")
    return {
        "feature_columns": list(rc.FEATURE_COLUMNS),
        "full_features": expected,
        "n_temporal_features": len(rc.FEATURE_COLUMNS),
        "n_full_features": len(expected),
    }


def assert_within_secreted_frame(summary: dict) -> dict:
    observed = {key: summary.get(key) for key in EXPECTED_WITHIN_SECRETED}
    if observed != EXPECTED_WITHIN_SECRETED:
        raise RuntimeError(
            f"Within-secreted frame mismatch: observed {observed}, expected {EXPECTED_WITHIN_SECRETED}. "
            "Stopping instead of reporting numbers for a different stratum."
        )
    return observed


def ordered_genes(eval_df, scores: np.ndarray) -> list[str]:
    order = np.argsort(-scores, kind="stable")
    return eval_df["gene"].to_numpy()[order].tolist()


def positive_rank_map(eval_df, scores: np.ndarray, positives: list[str]) -> dict[str, int]:
    ranked_genes = ordered_genes(eval_df, scores)
    ranks = {gene: idx + 1 for idx, gene in enumerate(ranked_genes)}
    return {gene: int(ranks[gene]) for gene in positives}


def score_summary(scores: np.ndarray) -> dict:
    unique_scores = np.unique(scores)
    out = {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_unique_exact": int(len(unique_scores)),
        "all_scores_equal": bool(len(unique_scores) == 1),
    }
    if len(unique_scores) <= 20:
        out["unique_scores"] = [float(score) for score in unique_scores.tolist()]
    return out


def compact_topk_metrics(rank_metrics: dict) -> dict:
    return {
        "p_at_5": rank_metrics["top5"]["precision"],
        "p_at_10": rank_metrics["top10"]["precision"],
        "p_at_20": rank_metrics["top20"]["precision"],
        "ef_at_10": rank_metrics["top10"]["enrichment_factor"],
        "hypergeom_p_upper_at_10": rank_metrics["top10"]["hypergeom_p_upper"],
        "top10_genes": rank_metrics["top10"]["top_genes"],
    }


def score_groups_by_prior(eval_df, scores: np.ndarray) -> dict:
    groups = {}
    prior = eval_df["is_secreted"].to_numpy(dtype=int)
    labels = eval_df["label"].to_numpy(dtype=int)
    genes = eval_df["gene"].to_numpy()
    for value in sorted(set(prior.tolist())):
        mask = prior == value
        unique_scores = np.unique(scores[mask])
        groups[str(int(value))] = {
            "n_genes": int(mask.sum()),
            "n_positives": int(labels[mask].sum()),
            "positive_genes": sorted(genes[mask & (labels == 1)].tolist()),
            "n_unique_scores": int(len(unique_scores)),
            "unique_scores": [float(score) for score in unique_scores.tolist()],
        }
    return groups


def run_full_fit(eval_df, kind: str, feature_columns: list[str]) -> tuple[np.ndarray, dict]:
    previous_features = list(e12.FULL_FEATURES)
    e12.FULL_FEATURES = list(feature_columns)
    try:
        scores, cfg = e12.full_fit_ranking(eval_df, kind)
    finally:
        e12.FULL_FEATURES = previous_features
    return np.asarray(scores, dtype=float), cfg


def ranking_entry(eval_df, kind: str, feature_columns: list[str]) -> dict:
    scores, cfg = run_full_fit(eval_df, kind, feature_columns)
    ranked_genes = ordered_genes(eval_df, scores)
    entry = {
        "feature_columns": list(feature_columns),
        "n_features": int(len(feature_columns)),
        "config": cfg,
        "score_summary": score_summary(scores),
        "tie_break_policy": TIE_BREAK_POLICY,
        "positive_ranks": positive_rank_map(eval_df, scores, EXPECTED_POSITIVES),
        "top10_scores": [float(scores[idx]) for idx in np.argsort(-scores, kind="stable")[:10]],
        "top10_labels": [
            int(value)
            for value in eval_df["label"].to_numpy(dtype=int)[np.argsort(-scores, kind="stable")[:10]].tolist()
        ],
    }
    if entry["score_summary"]["all_scores_equal"]:
        entry["degenerate_ranking"] = True
        entry["rank_metrics"] = None
        entry["requested_metrics"] = None
        entry["degenerate_note"] = (
            "All predicted probabilities are equal; top-k precision is not reported because the model "
            "does not define a meaningful ranking beyond E12's stable tie-break order."
        )
        entry["top10_genes_by_stable_tie_break"] = ranked_genes[:10]
        if feature_columns == PRIOR_ONLY_FEATURES:
            entry["score_groups_by_is_secreted"] = score_groups_by_prior(eval_df, scores)
        return entry

    rank_metrics = e12.rank_metrics(eval_df, scores, ks=(5, 10, 20))
    entry["degenerate_ranking"] = False
    entry["rank_metrics"] = rank_metrics
    entry["requested_metrics"] = compact_topk_metrics(rank_metrics)
    if feature_columns == PRIOR_ONLY_FEATURES:
        entry["score_groups_by_is_secreted"] = score_groups_by_prior(eval_df, scores)
    return entry


def assert_published_full_fit(full_results: dict) -> dict:
    audit = {}
    for kind, expected in PUBLISHED_FULL_FIT_PRECISION.items():
        metrics = full_results[kind]["requested_metrics"]
        observed = {"p_at_5": metrics["p_at_5"], "p_at_10": metrics["p_at_10"]}
        for key, expected_value in expected.items():
            if not math.isclose(observed[key], expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"Recomputed deployed {kind} {key}={observed[key]:.12g}, expected {expected_value:.12g}. "
                    "Stopping because the E12 published-value cross-check failed."
                )
        audit[kind] = {"observed": observed, "expected": expected, "matches_published": True}
    return audit


def mean_rank(rank_map: dict[str, int]) -> float:
    return float(np.mean([rank_map[gene] for gene in EXPECTED_POSITIVES]))


def verdict(full_results: dict, prior_results: dict) -> str:
    clauses = []
    for kind in MODEL_KEYS:
        full_mean = mean_rank(full_results[kind]["positive_ranks"])
        prior_mean = mean_rank(prior_results[kind]["positive_ranks"])
        if prior_results[kind]["degenerate_ranking"]:
            clauses.append(
                f"{kind}: prior-only is an all-score tie, so it has no meaningful model ranking "
                f"(stable tie-order mean ranks {full_mean:.2f} vs {prior_mean:.2f})"
            )
            continue
        if full_mean < prior_mean:
            direction = "full earlier by mean positive rank"
        elif full_mean > prior_mean:
            direction = "prior-only earlier by mean positive rank"
        else:
            direction = "tied by mean positive rank"
        full_all_earlier = all(
            full_results[kind]["positive_ranks"][gene] < prior_results[kind]["positive_ranks"][gene]
            for gene in EXPECTED_POSITIVES
        )
        clauses.append(
            f"{kind}: {direction} ({full_mean:.2f} vs {prior_mean:.2f}); "
            f"all-four-individually-earlier={full_all_earlier}"
        )
    return "Verdict: " + "; ".join(clauses) + "."


def main() -> None:
    start = time.time()
    rc.ensure_results_dir()
    print(f"[{rc.utc_timestamp()}] E22 within-secreted prior-only ranking starting", flush=True)

    deployed_features = assert_deployed_feature_state()
    bundle, kw_genes, bundle_path = load_cached_uniprot_kw_genes()
    eval_df, summary = e12.build_eval(e12.WITHIN_SECRETED_SPEC, kw_genes=kw_genes)
    frame_audit = assert_within_secreted_frame(summary)

    positives_secreted = sorted(
        eval_df.loc[(eval_df["label"] == 1) & (eval_df["is_secreted"] == 1), "gene"].astype(str).tolist()
    )
    secreted_genes_in_frame = sorted(eval_df.loc[eval_df["is_secreted"] == 1, "gene"].astype(str).tolist())
    prior_audit = {
        "is_secreted_1_count_in_246_gene_frame": int(eval_df["is_secreted"].sum()),
        "is_secreted_1_genes_in_frame": secreted_genes_in_frame,
        "positive_genes_with_is_secreted_1": positives_secreted,
        "positive_genes_with_is_secreted_0": sorted(set(EXPECTED_POSITIVES) - set(positives_secreted)),
        "tie_break_policy": TIE_BREAK_POLICY,
    }

    full_results = {
        kind: ranking_entry(eval_df, kind, list(rc.FULL_FEATURES))
        for kind in MODEL_KEYS
    }
    full_cross_check = assert_published_full_fit(full_results)

    prior_results = {
        kind: ranking_entry(eval_df, kind, PRIOR_ONLY_FEATURES)
        for kind in MODEL_KEYS
    }

    payload = rc.provenance(
        SCRIPT_NAME,
        parameters={
            "within_secreted_spec": e12.WITHIN_SECRETED_SPEC,
            "model_kinds": MODEL_KEYS,
            "prior_only_features": PRIOR_ONLY_FEATURES,
            "same_full_fit_ranking_function_as_e12": True,
            "same_rank_metrics_function_as_e12": True,
            "rank_metrics_ks": [5, 10, 20],
            "published_full_fit_precision_cross_check": PUBLISHED_FULL_FIT_PRECISION,
            "annotation_source_mode": "read-only load from results/rev_exp_annotations.json",
            "n_outer": e12.N_OUTER,
            "n_inner": e12.N_INNER,
            "outer_seeds": e12.OUTER_SEEDS,
            "inner_seed": e12.INNER_SEED,
        },
        input_paths=[
            "rev_exp_common.py",
            "rev_exp_e12_strict_windows.py",
            "rev_exp_e9_nested_tuning.py",
            "results/rev_exp_annotations.json",
            "paper/supplementary_table_s1.tex",
        ],
    )
    payload["python_executable"] = sys.executable
    payload["annotation_bundle_path"] = str(bundle_path)
    payload["annotation_bundle_manifest"] = bundle["uniprot"]["manifest"]
    payload["deployed_feature_audit"] = deployed_features
    payload["within_secreted_summary"] = summary
    payload["within_secreted_frame_audit"] = frame_audit
    payload["prior_feature_audit"] = prior_audit
    payload["deployed_full_16_feature_recomputed"] = full_results
    payload["published_full_fit_cross_check"] = full_cross_check
    payload["prior_only"] = prior_results
    payload["verdict"] = verdict(full_results, prior_results)
    payload["runtime_seconds"] = float(time.time() - start)

    rc.write_json(OUT_PATH, payload)

    print(f"within-secreted strict: n={summary['n_genes']} pos={summary['n_positives']} ({summary['positives_in_eval']})", flush=True)
    print(
        f"is_secreted=1 in frame: {prior_audit['is_secreted_1_count_in_246_gene_frame']} genes; "
        f"positives among them: {prior_audit['positive_genes_with_is_secreted_1']}",
        flush=True,
    )
    for section, results in [("prior-only", prior_results), ("full-16 recomputed", full_results)]:
        for kind in MODEL_KEYS:
            if results[kind]["degenerate_ranking"]:
                print(f"{section}/{kind}: degenerate all-score tie", flush=True)
                continue
            metrics = results[kind]["requested_metrics"]
            print(
                f"{section}/{kind}: P@5={metrics['p_at_5']:.3f} P@10={metrics['p_at_10']:.3f} "
                f"P@20={metrics['p_at_20']:.3f} EF@10={metrics['ef_at_10']:.3f} "
                f"hypergeomP@10={metrics['hypergeom_p_upper_at_10']:.6g} "
                f"top10={metrics['top10_genes']}",
                flush=True,
            )
    print(payload["verdict"], flush=True)
    print(f"Wrote {OUT_PATH} in {payload['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
