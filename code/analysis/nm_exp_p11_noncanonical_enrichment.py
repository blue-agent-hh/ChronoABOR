#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nm_exp4_full_universe import FEATURE_COLUMNS, RANDOM_SEED, make_xgb, to_jsonable
from nm_exp_gemini_baseline import build_data
from nm_exp_p1_prior_control import SECRETED_GENES
from nm_exp_p10_second_validation import ensure_frick_workbook, extract_frick_set2
from nm_exp_p5_external_validation import compute_overlap
from nm_exp_p7_ranked_enrichment import compute_enrichment_curve, compute_nes_and_pvalue

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p11_noncanonical_enrichment.json"
P5_JSON = RESULTS_DIR / "nm_exp_p5_overlap.json"
P10_JSON = RESULTS_DIR / "nm_exp_p10_second_validation.json"

PERMUTATION_COUNT = 1000
TOP_K = 50
FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]
HARDCODED_CANONICAL_GENES = {
    "APOE",
    "APP",
    "MAPT",
    "GFAP",
    "NEFL",
    "TREM2",
    "CLU",
    "BIN1",
    "PICALM",
    "CR1",
    "SNAP25",
    "NRGN",
}


def summarize_array(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.5)),
        "p975": float(np.quantile(values, 0.975)),
    }


def train_full_fold_c_model() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    data_summary, full_df = build_data()
    full_df = full_df.copy()
    full_df["is_secreted"] = full_df["gene"].isin(SECRETED_GENES).astype(int)

    X = full_df[FULL_FEATURES].to_numpy(dtype=float)
    y = full_df["label_external"].to_numpy(dtype=int)

    model = make_xgb()
    model.fit(X, y)

    full_df["chronoabor_score"] = model.predict_proba(X)[:, 1]
    ranked_df = full_df.sort_values(
        by=["chronoabor_score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return data_summary, full_df, ranked_df


def build_canonical_summary(full_df: pd.DataFrame) -> dict:
    top30_df = full_df.sort_values(by=["num_studies", "gene"], ascending=[False, True]).head(30).copy()
    top30_genes = top30_df["gene"].tolist()
    union_genes = sorted(set(top30_genes) | HARDCODED_CANONICAL_GENES)
    in_universe = sorted(set(union_genes) & set(full_df["gene"]))
    missing_from_universe = sorted(set(union_genes) - set(full_df["gene"]))

    top30_records = top30_df[["gene", "num_studies", "layer_count", "sig_count_all"]].to_dict(orient="records")
    hardcoded_only = sorted(HARDCODED_CANONICAL_GENES - set(top30_genes))

    return {
        "top30_by_num_studies": top30_genes,
        "top30_by_num_studies_detail": top30_records,
        "hardcoded_always_excluded": sorted(HARDCODED_CANONICAL_GENES),
        "hardcoded_only_not_already_in_top30": hardcoded_only,
        "canonical_union_all": union_genes,
        "canonical_union_in_universe": in_universe,
        "canonical_union_missing_from_universe": missing_from_universe,
        "n_top30": int(len(top30_genes)),
        "n_hardcoded": int(len(HARDCODED_CANONICAL_GENES)),
        "n_canonical_union_all": int(len(union_genes)),
        "n_canonical_union_in_universe": int(len(in_universe)),
    }


def load_set1_hits() -> tuple[list[str], dict]:
    if P10_JSON.exists():
        payload = json.loads(P10_JSON.read_text())
        genes = payload["cohort_analyses"]["set1"]["hit_genes_in_universe"]
        return genes, {"source_file": str(P10_JSON), "source_key": "cohort_analyses.set1.hit_genes_in_universe"}

    if P5_JSON.exists():
        payload = json.loads(P5_JSON.read_text())
        genes = payload["external_validation"]["external_upregulated"]["genes_in_universe"]
        return genes, {"source_file": str(P5_JSON), "source_key": "external_validation.external_upregulated.genes_in_universe"}

    raise FileNotFoundError("Missing both results/nm_exp_p10_second_validation.json and results/nm_exp_p5_overlap.json")


def load_set2_hits(universe_set: set[str]) -> tuple[list[str], dict]:
    if P10_JSON.exists():
        payload = json.loads(P10_JSON.read_text())
        genes = payload["cohort_analyses"]["set2"]["hit_genes_in_universe"]
        return genes, {"source_file": str(P10_JSON), "source_key": "cohort_analyses.set2.hit_genes_in_universe"}

    workbook_path = ensure_frick_workbook()
    summary = extract_frick_set2(workbook_path, universe_set)
    return summary["genes_in_universe"], {
        "source_file": str(workbook_path),
        "source_key": "Sup_table_2/model1/FDR<0.05",
        "article_url": summary["article_url"],
        "supplement_url": summary["supplement_url"],
    }


def analyze_noncanonical_hit_set(
    name: str,
    description: str,
    ranked_df: pd.DataFrame,
    canonical_genes: set[str],
    original_hit_genes: list[str],
) -> dict:
    noncanonical_ranked_df = ranked_df.loc[~ranked_df["gene"].isin(canonical_genes)].copy().reset_index(drop=True)
    noncanonical_hits = sorted(set(original_hit_genes) - canonical_genes)
    hit_set = set(noncanonical_hits)
    if not hit_set:
        raise RuntimeError(f"{name} has no non-canonical hits after filtering.")

    top50_fisher = compute_overlap(
        label=f"{name}_noncanonical_top50",
        chrono_ranked_genes=noncanonical_ranked_df.head(TOP_K)["gene"].tolist(),
        external_genes_in_universe=noncanonical_hits,
        background_n=len(noncanonical_ranked_df),
    )

    enrichment_curve = compute_enrichment_curve(noncanonical_ranked_df, "chronoabor_score", hit_set)
    nes, p_value, permutation_es = compute_nes_and_pvalue(
        noncanonical_ranked_df["chronoabor_score"].to_numpy(dtype=float),
        int(enrichment_curve["hit_mask"].sum()),
        float(enrichment_curve["es"]),
        n_permutations=PERMUTATION_COUNT,
        seed=RANDOM_SEED,
    )

    canonical_hits_removed = sorted(set(original_hit_genes) & canonical_genes)

    return {
        "name": name,
        "description": description,
        "n_hits_original_in_universe": int(len(set(original_hit_genes))),
        "n_canonical_hits_removed": int(len(canonical_hits_removed)),
        "canonical_hits_removed": canonical_hits_removed,
        "n_hits_noncanonical_in_universe": int(len(noncanonical_hits)),
        "hit_genes_noncanonical_in_universe": noncanonical_hits,
        "top50_fisher": top50_fisher,
        "gsea_random_null": {
            "es": float(enrichment_curve["es"]),
            "nes": float(nes),
            "p_value": float(p_value),
            "peak_index_1based": int(enrichment_curve["peak_index"] + 1),
            "leading_edge_genes": enrichment_curve["leading_edge_genes"],
            "leading_edge_size": int(len(enrichment_curve["leading_edge_genes"])),
            "null_summary": summarize_array(permutation_es),
        },
    }


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_summary, full_df, ranked_df = train_full_fold_c_model()
    universe_set = set(ranked_df["gene"])

    canonical_summary = build_canonical_summary(full_df)
    canonical_genes = set(canonical_summary["canonical_union_in_universe"])

    set1_hits, set1_source = load_set1_hits()
    set2_hits, set2_source = load_set2_hits(universe_set)

    noncanonical_ranked_df = ranked_df.loc[~ranked_df["gene"].isin(canonical_genes)].copy().reset_index(drop=True)

    set1_result = analyze_noncanonical_hit_set(
        name="set1_iadrc_knight_upregulated",
        description="Indiana ADRC + Knight ADRC/Stanford plasma proteomics upregulated hit set",
        ranked_df=ranked_df,
        canonical_genes=canonical_genes,
        original_hit_genes=set1_hits,
    )
    set2_result = analyze_noncanonical_hit_set(
        name="set2_frick2024_incident_load_all_associated",
        description="Frick et al. 2024 incident LOAD serum proteins, all FDR-significant model-1 associations",
        ranked_df=ranked_df,
        canonical_genes=canonical_genes,
        original_hit_genes=set2_hits,
    )

    results = {
        "analysis_name": "nm_exp_p11_noncanonical_enrichment",
        "random_seed": RANDOM_SEED,
        "permutation_count": PERMUTATION_COUNT,
        "runtime_seconds": float(time.time() - start_time),
        "score_source": {
            "feature_matrix_source": "nm_exp_gemini_baseline.build_data()",
            "ranking_model": "full-fit XGBoost on label_external using features <= 2021",
            "feature_columns": FULL_FEATURES,
        },
        "dataset_summary": data_summary,
        "canonical_gene_definition": canonical_summary,
        "ranking_summary": {
            "n_ranked_genes_original": int(len(ranked_df)),
            "n_canonical_genes_removed": int(len(canonical_genes)),
            "n_ranked_genes_noncanonical": int(len(noncanonical_ranked_df)),
            "top50_noncanonical_genes": noncanonical_ranked_df.head(TOP_K)["gene"].tolist(),
        },
        "external_hit_sources": {
            "set1": set1_source,
            "set2": set2_source,
        },
        "cohort_analyses": {
            "set1": set1_result,
            "set2": set2_result,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2, sort_keys=True))

    print("NM Experiment P11: non-canonical external enrichment")
    print(f"Canonical genes removed: {results['ranking_summary']['n_canonical_genes_removed']}")
    print(f"Non-canonical genes remaining in universe: {results['ranking_summary']['n_ranked_genes_noncanonical']}")
    print(f"Set 1 non-canonical hits in universe: {set1_result['n_hits_noncanonical_in_universe']}")
    print(
        f"Set 1 top50 Fisher OR={set1_result['top50_fisher']['odds_ratio']:.3f} "
        f"p={set1_result['top50_fisher']['fisher_p_value']:.6g}"
    )
    print(
        f"Set 1 GSEA NES={set1_result['gsea_random_null']['nes']:.3f} "
        f"p={set1_result['gsea_random_null']['p_value']:.6g}"
    )
    print(f"Set 2 non-canonical hits in universe: {set2_result['n_hits_noncanonical_in_universe']}")
    print(
        f"Set 2 top50 Fisher OR={set2_result['top50_fisher']['odds_ratio']:.3f} "
        f"p={set2_result['top50_fisher']['fisher_p_value']:.6g}"
    )
    print(
        f"Set 2 GSEA NES={set2_result['gsea_random_null']['nes']:.3f} "
        f"p={set2_result['gsea_random_null']['p_value']:.6g}"
    )
    print(f"Results JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
