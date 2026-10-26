#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nm_exp4_full_universe import RANDOM_SEED, to_jsonable
from nm_exp_p5_external_validation import compute_overlap
from nm_exp_p11_noncanonical_enrichment import (
    build_canonical_summary,
    load_set1_hits,
    train_full_fold_c_model,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p13_noncanon_thresholds.json"

THRESHOLDS = [10, 25, 50, 100]


def main() -> None:
    start = time.time()
    data_summary, full_df, ranked_df = train_full_fold_c_model()
    canonical_summary = build_canonical_summary(full_df)
    canonical_genes = set(canonical_summary["canonical_union_in_universe"])

    noncanon = ranked_df.loc[~ranked_df["gene"].isin(canonical_genes)].copy().reset_index(drop=True)

    set1_hits, set1_summary = load_set1_hits()
    noncanon_hits = sorted(set(set1_hits) - canonical_genes)
    n_universe = len(noncanon)

    rankings = {
        "chronoabor": noncanon.sort_values(
            by=["chronoabor_score", "num_studies", "gene"], ascending=[False, False, True]
        ).reset_index(drop=True),
        "sig_max_auc": noncanon.sort_values(
            by=["sig_max_auc", "num_studies", "gene"], ascending=[False, False, True]
        ).reset_index(drop=True),
        "num_studies": noncanon.sort_values(
            by=["num_studies", "gene"], ascending=[False, True]
        ).reset_index(drop=True),
    }

    results = {"thresholds": {}}
    for k in THRESHOLDS:
        row = {}
        for method, rdf in rankings.items():
            overlap = compute_overlap(
                label=f"p13_{method}_top{k}",
                chrono_ranked_genes=rdf.head(k)["gene"].tolist(),
                external_genes_in_universe=noncanon_hits,
                background_n=n_universe,
            )
            row[method] = overlap
        results["thresholds"][str(k)] = row

    results.update(
        {
            "analysis_name": "nm_exp_p13_noncanon_thresholds",
            "random_seed": RANDOM_SEED,
            "n_canonical_genes_removed": len(canonical_genes),
            "n_noncanon_universe": n_universe,
            "n_noncanon_hits": len(noncanon_hits),
            "noncanon_hits": noncanon_hits,
            "canonical_genes_removed": sorted(canonical_genes),
            "dataset_summary": data_summary,
            "runtime_seconds": time.time() - start,
        }
    )
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("P13: non-canonical threshold robustness (Indiana/Knight)")
    print(f"  canonical removed: {len(canonical_genes)} | universe: {n_universe} | hits: {len(noncanon_hits)}")
    for k in THRESHOLDS:
        row = results["thresholds"][str(k)]
        parts = []
        for method in ("chronoabor", "sig_max_auc", "num_studies"):
            o = row[method]
            parts.append(
                f"{method}: {o['overlap_count']} OR={o['odds_ratio']:.2f} p={o['fisher_p_value']:.3g}"
            )
        print(f"  top-{k}: " + " | ".join(parts))
    print(f"Results JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
