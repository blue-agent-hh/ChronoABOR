#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nm_exp4_full_universe import RANDOM_SEED, to_jsonable
from nm_exp_p6_secreted_ablation import summarize_distribution
from nm_exp_p7b_matched_enrichment import (
    MATCHED_PERMUTATIONS,
    build_feasible_match_pools,
    build_scored_universe,
    compute_enrichment_fixed_ranking,
    compute_matched_null,
    compute_nes_and_pvalue,
)
from nm_exp_p11_noncanonical_enrichment import HARDCODED_CANONICAL_GENES

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p12_noncanonical_matched_null.json"
P11_JSON = RESULTS_DIR / "nm_exp_p11_noncanonical_enrichment.json"


def load_p11_noncanonical_hits() -> dict[str, list[str]]:
    data = json.loads(P11_JSON.read_text())
    result = {}
    for cohort_key, cohort in data["cohort_analyses"].items():
        result[cohort_key] = cohort["hit_genes_noncanonical_in_universe"]
    return result


def run_matched_null_for_cohort(
    cohort_name: str,
    hit_list: list[str],
    scored_df_noncanon,
    ranked_genes_noncanon: np.ndarray,
    ranked_scores_noncanon: np.ndarray,
) -> dict:
    hit_set = set(hit_list)
    n_hits_in_universe = len(hit_set & set(ranked_genes_noncanon))
    if n_hits_in_universe < 3:
        return {
            "cohort": cohort_name,
            "n_hits_in_noncanon_universe": n_hits_in_universe,
            "error": "Too few non-canonical hits overlap the non-canonical universe",
        }

    observed_curve = compute_enrichment_fixed_ranking(ranked_genes_noncanon, ranked_scores_noncanon, hit_set)
    observed_es = float(observed_curve["es"])

    try:
        match_pools, matching_diagnostics = build_feasible_match_pools(scored_df_noncanon, hit_set)
    except RuntimeError as e:
        return {
            "cohort": cohort_name,
            "n_hits_in_noncanon_universe": n_hits_in_universe,
            "observed_es": observed_es,
            "error": f"Match pool construction failed: {e}",
        }

    matched_es_arr, _ = compute_matched_null(ranked_genes_noncanon, ranked_scores_noncanon, match_pools)
    matched_nes, matched_p_value = compute_nes_and_pvalue(observed_es, matched_es_arr)

    return {
        "cohort": cohort_name,
        "n_hits_in_noncanon_universe": n_hits_in_universe,
        "hit_genes": sorted(hit_set & set(ranked_genes_noncanon)),
        "observed_enrichment": {
            "es": observed_es,
            "peak_index_1based": int(observed_curve["peak_index"] + 1),
            "leading_edge_genes": observed_curve["leading_edge_genes"],
        },
        "matched_null": {
            "matched_nes": float(matched_nes),
            "matched_p_value": float(matched_p_value),
            "null_distribution": {
                **summarize_distribution(matched_es_arr),
                "ci95_lower": float(np.quantile(matched_es_arr, 0.025)),
                "ci95_upper": float(np.quantile(matched_es_arr, 0.975)),
            },
        },
        "matching_criteria": {
            "canonical_genes_excluded_from_universe": True,
            "is_secreted": "exact match",
            "has_signature": "exact match (sig_count_all > 0)",
            "num_studies": "±50% range",
            "candidate_space": "non-canonical, non-hit genes only",
        },
        "matching_diagnostics": {
            "n_hits": matching_diagnostics["n_hits"],
            "n_non_hit_genes": matching_diagnostics["n_non_hit_genes"],
            "pool_size_summary": matching_diagnostics["pool_size_summary"],
        },
    }


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("P12: Non-canonical matched-null enrichment analysis")
    print(f"  Canonical genes excluded from universe: {len(HARDCODED_CANONICAL_GENES)} hardcoded + top-30 by evidence")

    noncanon_hits = load_p11_noncanonical_hits()
    print(f"  Loaded P11 non-canonical hit sets: {list(noncanon_hits.keys())}")

    scored_df, dataset_summary = build_scored_universe()

    p11_data = json.loads(P11_JSON.read_text())
    canonical_excluded = set()
    for cohort in p11_data["cohort_analyses"].values():
        canonical_excluded.update(cohort.get("canonical_hits_removed", []))
    top30_by_studies = (
        scored_df.sort_values("num_studies", ascending=False)
        .head(30)["gene"]
        .tolist()
    )
    canonical_excluded.update(top30_by_studies)
    canonical_excluded.update(HARDCODED_CANONICAL_GENES)
    print(f"  Total canonical genes excluded from ranking universe: {len(canonical_excluded)}")

    scored_df_noncanon = scored_df.loc[~scored_df["gene"].isin(canonical_excluded)].copy().reset_index(drop=True)
    ranked_df = scored_df_noncanon.sort_values(
        by=["chronoabor_score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked_genes_noncanon = ranked_df["gene"].to_numpy(dtype=str)
    ranked_scores_noncanon = ranked_df["chronoabor_score"].to_numpy(dtype=float)
    print(f"  Non-canonical universe size: {len(ranked_genes_noncanon)} genes")

    cohort_results = {}
    for cohort_key, hit_list in noncanon_hits.items():
        print(f"\n  Processing cohort: {cohort_key} ({len(hit_list)} non-canonical hits)")
        result = run_matched_null_for_cohort(
            cohort_name=cohort_key,
            hit_list=hit_list,
            scored_df_noncanon=scored_df_noncanon,
            ranked_genes_noncanon=ranked_genes_noncanon,
            ranked_scores_noncanon=ranked_scores_noncanon,
        )
        cohort_results[cohort_key] = result

        if "matched_null" in result:
            mn = result["matched_null"]
            print(f"    Matched NES = {mn['matched_nes']:.3f}, p = {mn['matched_p_value']:.4f}")
        elif "error" in result:
            print(f"    ERROR: {result['error']}")

    output = {
        "analysis_name": "P12: Non-canonical matched-null enrichment",
        "description": (
            "Runs matched-null GSEA (controlling for is_secreted, num_studies, detectability) "
            "on the non-canonical hit sets from P11, after also excluding canonical genes from "
            "the ChronoABOR ranking universe. Tests whether enrichment survives prior-matched "
            "controls even without canonical gene contribution."
        ),
        "canonical_genes_excluded": sorted(canonical_excluded),
        "n_canonical_genes_excluded": len(canonical_excluded),
        "noncanonical_universe_size": int(len(ranked_genes_noncanon)),
        "matched_permutations": MATCHED_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "cohort_analyses": cohort_results,
        "dataset_summary": dataset_summary,
        "runtime_seconds": float(time.time() - start_time),
    }

    OUTPUT_JSON.write_text(json.dumps(to_jsonable(output), indent=2))
    print(f"\nResults saved to: {OUTPUT_JSON}")

    print("\n=== P12 SUMMARY ===")
    for cohort_key, result in cohort_results.items():
        if "matched_null" in result:
            mn = result["matched_null"]
            print(f"  {cohort_key}: Matched NES={mn['matched_nes']:.3f}, p={mn['matched_p_value']:.4f} (n_hits={result['n_hits_in_noncanon_universe']})")
        else:
            print(f"  {cohort_key}: {result.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()
