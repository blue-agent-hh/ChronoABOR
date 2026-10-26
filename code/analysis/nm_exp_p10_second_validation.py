#!/usr/bin/env python3

from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from nm_exp4_full_universe import RANDOM_SEED, to_jsonable
from nm_exp_p5_external_validation import compute_overlap, normalize_external_gene_symbol
from nm_exp_p7_ranked_enrichment import (
    FEATURE_CUTOFF_YEAR,
    build_fold_c_frame,
    compute_enrichment_curve,
    compute_nes_and_pvalue,
    train_fold_c_model,
)
from nm_exp_p7b_matched_enrichment import (
    build_feasible_match_pools,
    compute_enrichment_fixed_ranking,
    compute_matched_null,
    compute_nes_and_pvalue as compute_matched_nes_and_pvalue,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
EXTERNAL_DATA_DIR = BASE_DIR / "external_data"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p10_second_validation.json"
INPUT_P7_JSON = RESULTS_DIR / "nm_exp_p7_ranked_enrichment.json"
INPUT_GEMINI_JSON = RESULTS_DIR / "nm_exp_gemini_31pro_core.json"
INPUT_WITHIN_SECRETED_JSON = RESULTS_DIR / "nm_exp_llm_within_secreted.json"

FRICK_2024_ARTICLE_URL = "https://www.nature.com/articles/s43587-024-00693-1"
FRICK_2024_SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs43587-024-00693-1/"
    "MediaObjects/43587_2024_693_MOESM3_ESM.xlsx"
)
FRICK_2024_SUPPLEMENT_XLSX = EXTERNAL_DATA_DIR / "Frick2024_SupplementaryTables.xlsx"


def summarize_array(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.5)),
        "p975": float(np.quantile(values, 0.975)),
    }


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as handle:
        handle.write(response.read())
    temp_path.replace(destination)


def ensure_frick_workbook() -> Path:
    if FRICK_2024_SUPPLEMENT_XLSX.exists() and zipfile.is_zipfile(FRICK_2024_SUPPLEMENT_XLSX):
        return FRICK_2024_SUPPLEMENT_XLSX
    download_file(FRICK_2024_SUPPLEMENT_URL, FRICK_2024_SUPPLEMENT_XLSX)
    if not zipfile.is_zipfile(FRICK_2024_SUPPLEMENT_XLSX):
        raise RuntimeError(f"Downloaded file is not a valid XLSX workbook: {FRICK_2024_SUPPLEMENT_XLSX}")
    return FRICK_2024_SUPPLEMENT_XLSX


def build_scored_universe() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_df, dataset_summary = build_fold_c_frame()
    scored_df = train_fold_c_model(feature_df)
    scored_df["has_signature"] = (scored_df["sig_count_all"] > 0).astype(int)
    ranked_df = scored_df.sort_values(
        by=["chronoabor_score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return scored_df, ranked_df, dataset_summary


def load_set1_hits() -> list[str]:
    if not INPUT_P7_JSON.exists():
        raise FileNotFoundError(f"Missing prerequisite file: {INPUT_P7_JSON}")
    payload = json.loads(INPUT_P7_JSON.read_text())
    return payload["proteomics_hits"]["hit_genes_in_universe"]


def extract_frick_set2(workbook_path: Path, universe_set: set[str]) -> dict:
    df = pd.read_excel(workbook_path, sheet_name="Sup_table_2", header=5)
    df = df.loc[df[df.columns[0]] != "Somamer"].copy()
    df = df.rename(
        columns={
            df.columns[0]: "Somamer",
            df.columns[1]: "Gene_symbol",
            df.columns[3]: "APOE_dependence",
            df.columns[4]: "HR_model1",
            df.columns[7]: "P_model1",
            df.columns[8]: "FDR_model1",
            df.columns[10]: "HR_model2",
            df.columns[13]: "P_model2",
            df.columns[14]: "FDR_model2",
            df.columns[16]: "HR_model3",
            df.columns[19]: "P_model3",
            df.columns[20]: "FDR_model3",
        }
    )

    for column in [
        "HR_model1",
        "P_model1",
        "FDR_model1",
        "HR_model2",
        "P_model2",
        "FDR_model2",
        "HR_model3",
        "P_model3",
        "FDR_model3",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Gene_norm"] = df["Gene_symbol"].map(normalize_external_gene_symbol)
    significant = df.loc[(df["FDR_model1"] < 0.05) & df["Gene_norm"].notna()].copy()
    significant = significant.sort_values(
        by=["FDR_model1", "P_model1", "Somamer"],
        ascending=[True, True, True],
    )
    representative_rows = significant.drop_duplicates("Gene_norm", keep="first").copy()

    all_genes = representative_rows["Gene_norm"].tolist()
    genes_hr_positive = representative_rows.loc[representative_rows["HR_model1"] > 1.0, "Gene_norm"].tolist()
    genes_hr_negative = representative_rows.loc[representative_rows["HR_model1"] < 1.0, "Gene_norm"].tolist()
    genes_in_universe = [gene for gene in all_genes if gene in universe_set]
    genes_not_in_universe = [gene for gene in all_genes if gene not in universe_set]

    return {
        "citation": "Frick et al., Nature Aging 2024",
        "article_title": (
            "Serum proteomics reveal APOE-e4-dependent and APOE-e4-independent "
            "protein signatures in Alzheimer's disease"
        ),
        "article_url": FRICK_2024_ARTICLE_URL,
        "supplement_url": FRICK_2024_SUPPLEMENT_URL,
        "supplement_local_path": str(workbook_path),
        "sheet_name": "Sup_table_2",
        "selection_rule": (
            "All model-1 incident-LOAD serum proteins with FDR < 0.05 were retained "
            "regardless of HR direction because ChronoABOR ranks disease relevance "
            "rather than effect-sign direction."
        ),
        "cohort_note": (
            "AGES cohort incident LOAD associations (Cox PH, age+sex adjusted) from "
            "Supplementary Table 2."
        ),
        "n_significant_aptamers_model1": int(len(significant)),
        "n_significant_genes_model1_unique": int(len(all_genes)),
        "n_hr_positive_model1": int(len(genes_hr_positive)),
        "n_hr_negative_model1": int(len(genes_hr_negative)),
        "genes_hr_positive_model1": genes_hr_positive,
        "genes_hr_negative_model1": genes_hr_negative,
        "n_genes_in_universe": int(len(genes_in_universe)),
        "n_genes_not_in_universe": int(len(genes_not_in_universe)),
        "genes_in_universe": genes_in_universe,
        "genes_not_in_universe": genes_not_in_universe,
        "representative_rows_preview": representative_rows.head(20)[
            ["Somamer", "Gene_symbol", "Gene_norm", "APOE_dependence", "HR_model1", "P_model1", "FDR_model1"]
        ].to_dict(orient="records"),
    }


def analyze_hit_set(
    name: str,
    description: str,
    hit_genes: list[str],
    scored_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> dict:
    hit_set = set(hit_genes)
    ranked_genes = ranked_df["gene"].to_numpy(dtype=str)
    ranked_scores = ranked_df["chronoabor_score"].to_numpy(dtype=float)

    random_curve = compute_enrichment_curve(ranked_df, "chronoabor_score", hit_set)
    random_nes, random_p_value, random_null = compute_nes_and_pvalue(
        ranked_scores,
        int(random_curve["hit_mask"].sum()),
        float(random_curve["es"]),
    )

    match_pools, matching_diagnostics = build_feasible_match_pools(scored_df, hit_set)
    observed_curve = compute_enrichment_fixed_ranking(ranked_genes, ranked_scores, hit_set)
    matched_null_es, matched_preview = compute_matched_null(ranked_genes, ranked_scores, match_pools)
    matched_nes, matched_p_value = compute_matched_nes_and_pvalue(
        float(observed_curve["es"]),
        matched_null_es,
    )

    top10 = compute_overlap(f"{name}_top10", ranked_df.head(10)["gene"].tolist(), hit_genes, len(ranked_df))
    top50 = compute_overlap(f"{name}_top50", ranked_df.head(50)["gene"].tolist(), hit_genes, len(ranked_df))

    return {
        "name": name,
        "description": description,
        "n_hits_in_universe": int(len(hit_set)),
        "hit_genes_in_universe": sorted(hit_set),
        "random_null_enrichment": {
            "es": float(random_curve["es"]),
            "nes": float(random_nes),
            "p_value": float(random_p_value),
            "peak_index_1based": int(random_curve["peak_index"] + 1),
            "leading_edge_genes": random_curve["leading_edge_genes"],
            "leading_edge_size": int(len(random_curve["leading_edge_genes"])),
            "null_summary": summarize_array(random_null),
        },
        "matched_null_enrichment": {
            "es": float(observed_curve["es"]),
            "nes": float(matched_nes),
            "p_value": float(matched_p_value),
            "peak_index_1based": int(observed_curve["peak_index"] + 1),
            "leading_edge_genes": observed_curve["leading_edge_genes"],
            "leading_edge_size": int(len(observed_curve["leading_edge_genes"])),
            "null_summary": summarize_array(matched_null_es),
            "matching_pool_summary": {
                "tolerance_level": matching_diagnostics["tolerance_level"],
                "n_hits": int(matching_diagnostics["n_hits"]),
                "n_non_hit_genes": int(matching_diagnostics["n_non_hit_genes"]),
                "n_unique_candidate_genes_across_all_pools": int(
                    matching_diagnostics["n_unique_candidate_genes_across_all_pools"]
                ),
                "pool_size_summary": matching_diagnostics["pool_size_summary"],
            },
            "sampled_gene_sets_preview": matched_preview,
        },
        "top10_fisher": top10,
        "top50_fisher": top50,
    }


def load_gemini_core_metrics() -> dict:
    if not INPUT_GEMINI_JSON.exists():
        return {"available": False}

    payload = json.loads(INPUT_GEMINI_JSON.read_text())
    result = {
        "available": True,
        "source_file": str(INPUT_GEMINI_JSON),
        "model_name": payload.get("eval_summary", {}).get("model_name"),
        "eval_summary": payload.get("eval_summary", {}),
        "auroc": payload.get("auroc"),
        "p_at_10": payload.get("p_at_10"),
        "p_at_20": payload.get("p_at_20"),
        "ef_at_10": payload.get("ef_at_10"),
        "top_15_genes": payload.get("top_15_genes"),
    }

    if INPUT_WITHIN_SECRETED_JSON.exists():
        within_secreted = json.loads(INPUT_WITHIN_SECRETED_JSON.read_text())
        secreted_genes = within_secreted.get("secreted_subset", {}).get("genes", [])
        score_genes = set(payload.get("scores", {}).keys())
        missing_genes = [gene for gene in secreted_genes if gene not in score_genes]
        result["within_secreted_coverage"] = {
            "n_secreted_genes_total": int(len(secreted_genes)),
            "n_secreted_genes_scored": int(len(secreted_genes) - len(missing_genes)),
            "n_secreted_genes_missing": int(len(missing_genes)),
            "missing_genes": missing_genes,
            "can_compute_full_within_secreted_row": bool(not missing_genes),
        }
    return result


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    workbook_path = ensure_frick_workbook()
    scored_df, ranked_df, dataset_summary = build_scored_universe()
    universe_set = set(ranked_df["gene"])

    set1_hits = load_set1_hits()
    set2_summary = extract_frick_set2(workbook_path, universe_set)

    set1_result = analyze_hit_set(
        name="set1_iadrc_knight_upregulated",
        description="Indiana ADRC + Knight ADRC/Stanford plasma proteomics upregulated hit set",
        hit_genes=set1_hits,
        scored_df=scored_df,
        ranked_df=ranked_df,
    )
    set2_result = analyze_hit_set(
        name="set2_frick2024_incident_load_all_associated",
        description="Frick et al. 2024 incident LOAD serum proteins, all FDR-significant model-1 associations",
        hit_genes=set2_summary["genes_in_universe"],
        scored_df=scored_df,
        ranked_df=ranked_df,
    )

    set1_hit_set = set(set1_result["hit_genes_in_universe"])
    set2_hit_set = set(set2_result["hit_genes_in_universe"])
    leading_edge_1 = set(set1_result["random_null_enrichment"]["leading_edge_genes"])
    leading_edge_2 = set(set2_result["random_null_enrichment"]["leading_edge_genes"])

    results = {
        "analysis_name": "nm_exp_p10_second_validation",
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "runtime_seconds": float(time.time() - start_time),
        "dataset_summary": dataset_summary,
        "ranking_summary": {
            "n_ranked_genes": int(len(ranked_df)),
            "top50_genes": ranked_df.head(50)["gene"].tolist(),
        },
        "set2_source": set2_summary,
        "cohort_analyses": {
            "set1": set1_result,
            "set2": set2_result,
        },
        "cohort_concordance": {
            "set1_n_hits_in_universe": int(len(set1_hit_set)),
            "set2_n_hits_in_universe": int(len(set2_hit_set)),
            "hit_intersection_size": int(len(set1_hit_set & set2_hit_set)),
            "hit_intersection_genes": sorted(set1_hit_set & set2_hit_set),
            "hit_jaccard_index": float(len(set1_hit_set & set2_hit_set) / len(set1_hit_set | set2_hit_set)),
            "leading_edge_overlap_size": int(len(leading_edge_1 & leading_edge_2)),
            "leading_edge_overlap_genes": sorted(leading_edge_1 & leading_edge_2),
            "top50_set1_hits": set1_result["top50_fisher"]["overlap_genes"],
            "top50_set2_hits": set2_result["top50_fisher"]["overlap_genes"],
            "top50_shared_hits_between_sets": sorted(
                set(set1_result["top50_fisher"]["overlap_genes"]) & set(set2_result["top50_fisher"]["overlap_genes"])
            ),
        },
        "gemini_31pro_core_check": load_gemini_core_metrics(),
    }

    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("NM Experiment P10: second external proteomics validation")
    print(f"Set 1 random NES={set1_result['random_null_enrichment']['nes']:.3f} p={set1_result['random_null_enrichment']['p_value']:.3f}")
    print(f"Set 1 matched NES={set1_result['matched_null_enrichment']['nes']:.3f} p={set1_result['matched_null_enrichment']['p_value']:.3f}")
    print(
        f"Set 1 top50 overlap={set1_result['top50_fisher']['overlap_count']} "
        f"({', '.join(set1_result['top50_fisher']['overlap_genes'])})"
    )
    print(f"Set 2 random NES={set2_result['random_null_enrichment']['nes']:.3f} p={set2_result['random_null_enrichment']['p_value']:.3f}")
    print(f"Set 2 matched NES={set2_result['matched_null_enrichment']['nes']:.3f} p={set2_result['matched_null_enrichment']['p_value']:.3f}")
    print(
        f"Set 2 top50 overlap={set2_result['top50_fisher']['overlap_count']} "
        f"({', '.join(set2_result['top50_fisher']['overlap_genes'])})"
    )
    print(f"Shared leading-edge genes: {', '.join(results['cohort_concordance']['leading_edge_overlap_genes'])}")
    print(f"Results JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
