#!/usr/bin/env python3

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from rev_exp_common import (
    BASE_DIR,
    FRICK_WORKBOOK_PATH,
    NM_BENCHMARK_TEX_PATH,
    RESULTS_DIR,
    SPECIAL_NM_ALIASES,
    build_scored_universe,
    build_sheet_with_detected_header,
    compute_rank_metrics_from_scores,
    ensure_results_dir,
    find_header_row,
    get_forward_filled_superheader,
    load_all_tables,
    normalize_external_gene_symbol,
    paired_stratified_bootstrap_auc_delta,
    provenance,
    stratified_bootstrap_auc,
    to_jsonable,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
OUTPUT_ANALYTE_CSV = RESULTS_DIR / "rev_exp_e6_analyte_map.csv"
GENE_SPLIT_RE = re.compile(r"[,;/]")
PTM_RE = re.compile(r"%?p-?tau|phospho|phosphorylated|acetylated|glycosylated", flags=re.IGNORECASE)
NON_ANALYTE_TOKEN_RE = re.compile(
    (
        r"age|sex|apoe genotype|allele count|cdr|clinical|demographic|risk factors|tabcat|pet|mri|"
        r"signature|protein panel|biomarker panel|indicators|composite-score|gene expression|proteomics|"
        r"model including|multiple metabolites|^metabolites$|^metabolite$|retinal|gliosis|rnfl|thickness"
    ),
    flags=re.IGNORECASE,
)
ANALYTE_ALIAS_PATTERNS = {
    "APP": [r"a[\s-]?[βb]|abeta|amyloid|aβ|beta-amyloid"],
    "MAPT": [r"tau|t-tau|p-?tau|ptau"],
    "NEFL": [r"\bnf-?l\b|nefl"],
    "NEFM": [r"\bnf-?m\b|nefm"],
    "NEFH": [r"\bnf-?h\b|nefh"],
    "GFAP": [r"\bgfap\b"],
    "APOE": [r"\bapoe\b"],
    "CLU": [r"clusterin|\bclu\b"],
    "VSNL1": [r"vilip-?1|vsnl1"],
    "ENO2": [r"\bnse\b|\beno2\b"],
    "ACHE": [r"acetylcholinesterase|\bache\b"],
    "BDNF": [r"\bbdnf\b"],
    "NRGN": [r"neurogranin|\bnrgn\b"],
    "SNAP25": [r"snap-?25|\bsnap25\b"],
    "SYT1": [r"synaptotagmin-?1|\bsyt1\b"],
    "VGF": [r"\bvgf\b"],
}


def ordered_signature_gene_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    genes = []
    for part in GENE_SPLIT_RE.split(str(raw_value).upper()):
        gene = part.strip()
        if not gene:
            continue
        gene = SPECIAL_NM_ALIASES.get(gene, gene)
        canonical = normalize_external_gene_symbol(gene)
        if canonical is None:
            continue
        genes.append(canonical)
    return genes


def split_analyte_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    tokens = []
    current = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if depth == 0 and char in {"+", ";", ",", "，", "；"}:
            token = "".join(current).strip()
            if token:
                tokens.append(re.sub(r"\s+", " ", token))
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        tokens.append(re.sub(r"\s+", " ", token))
    return tokens


def analyte_matches_gene(gene: str, analyte: str) -> bool:
    patterns = ANALYTE_ALIAS_PATTERNS.get(gene, [rf"\b{re.escape(gene)}\b"])
    return any(re.search(pattern, analyte, flags=re.IGNORECASE) for pattern in patterns)


def is_non_analyte_token(token: str) -> bool:
    return NON_ANALYTE_TOKEN_RE.search(token) is not None


def normalize_analyte_name(token: str) -> str:
    text = token.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:and|or|with)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" ,;:")
    text = text.replace("Abeta", "Aβ").replace("ABETA", "Aβ").replace("A-beta", "Aβ")
    text = text.replace("ptau", "p-tau").replace("P-tau", "p-tau")
    text = text.replace("Nf-L", "NfL").replace("Nf-H", "NfH").replace("Nf-M", "NfM")
    text = text.replace("，", ",")
    return text


def extract_gene_analyte_pairs(row: pd.Series, top_gene_set: set[str]) -> tuple[list[dict], bool]:
    all_genes = ordered_signature_gene_tokens(row["Gene_symbol"])
    genes = [gene for gene in all_genes if gene in top_gene_set]
    analyte_tokens = [normalize_analyte_name(token) for token in split_analyte_tokens(row["Biomarker_name"])]
    if not genes or not analyte_tokens:
        return [], False

    pairs: list[dict] = []
    if len(genes) == len(analyte_tokens):
        for gene, analyte in zip(genes, analyte_tokens, strict=True):
            if is_non_analyte_token(analyte):
                continue
            pairs.append({"gene": gene, "analyte": analyte, "mapping_method": "positional"})
        if pairs:
            return pairs, False

    matched_by_alias = False
    for gene in dict.fromkeys(genes):
        gene_matches = [
            analyte
            for analyte in analyte_tokens
            if not is_non_analyte_token(analyte) and analyte_matches_gene(gene, analyte)
        ]
        if gene_matches:
            matched_by_alias = True
            for analyte in dict.fromkeys(gene_matches):
                pairs.append({"gene": gene, "analyte": analyte, "mapping_method": "alias"})
    if matched_by_alias:
        return pairs, False

    unique_genes = list(dict.fromkeys(genes))
    if len(unique_genes) == 1 and len(dict.fromkeys(all_genes)) == 1:
        usable_tokens = [token for token in analyte_tokens if not is_non_analyte_token(token)]
        for analyte in dict.fromkeys(usable_tokens):
            pairs.append({"gene": unique_genes[0], "analyte": analyte, "mapping_method": "single_gene_row"})
        if pairs:
            return pairs, False

    return [], True


def build_analyte_map(top_genes: list[str]) -> tuple[pd.DataFrame, dict]:
    signature_df = load_all_tables()["signature"].copy()
    signature_df["Year_int"] = pd.to_numeric(signature_df["Year"], errors="coerce")
    signature_df["AUC_num"] = pd.to_numeric(signature_df["AUC"], errors="coerce")
    blood_df = signature_df.loc[
        signature_df["Tissue"].fillna("").astype(str).str.contains("blood|plasma|serum", case=False, na=False)
    ].copy()

    top_gene_set = set(top_genes)
    top_gene_rank = {gene: idx + 1 for idx, gene in enumerate(top_genes)}
    mapped_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    mapping_method_counter = defaultdict(int)

    for row_idx, row in blood_df.iterrows():
        pairs, ambiguous = extract_gene_analyte_pairs(row, top_gene_set)
        if ambiguous:
            ambiguous_rows.append(
                {
                    "row_index": int(row_idx),
                    "gene_symbol": row["Gene_symbol"],
                    "biomarker_name": row["Biomarker_name"],
                    "type": row["Type"],
                    "tissue": row["Tissue"],
                }
            )
        for pair in pairs:
            mapping_method_counter[pair["mapping_method"]] += 1
            mapped_rows.append(
                {
                    "chronoabor_rank": int(top_gene_rank[pair["gene"]]),
                    "gene": pair["gene"],
                    "analyte": pair["analyte"],
                    "mapping_method": pair["mapping_method"],
                    "auc": float(row["AUC_num"]) if pd.notna(row["AUC_num"]) else np.nan,
                    "sample_type": row["Tissue"],
                    "sample_description": row["Sample"],
                    "signature_type": row["Type"],
                    "pmid": row["PMID"],
                    "year": int(row["Year_int"]) if pd.notna(row["Year_int"]) else None,
                    "biomarker_name": row["Biomarker_name"],
                }
            )

    mapped_df = pd.DataFrame(mapped_rows)
    if mapped_df.empty:
        raise RuntimeError("No top-50 blood analyte mappings could be extracted from signature.csv.")

    aggregated_rows = []
    for (gene, analyte), group_df in mapped_df.groupby(["gene", "analyte"], dropna=False):
        best_row = (
            group_df.sort_values(
                by=["auc", "year", "pmid"],
                ascending=[False, True, True],
                na_position="last",
            )
            .iloc[0]
        )
        aggregated_rows.append(
            {
                "chronoabor_rank": int(best_row["chronoabor_rank"]),
                "gene": gene,
                "analyte": analyte,
                "best_auc": None if pd.isna(best_row["auc"]) else float(best_row["auc"]),
                "sample_type": best_row["sample_type"],
                "sample_description": best_row["sample_description"],
                "signature_type": best_row["signature_type"],
                "best_pmid": best_row["pmid"],
                "best_year": best_row["year"],
                "n_supporting_rows": int(len(group_df)),
                "mapping_methods": sorted(group_df["mapping_method"].unique().tolist()),
                "supporting_biomarker_names": sorted(group_df["biomarker_name"].astype(str).unique().tolist()),
            }
        )

    analyte_map_df = pd.DataFrame(aggregated_rows).sort_values(
        by=["chronoabor_rank", "gene", "best_auc", "analyte"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    analyte_map_df.to_csv(OUTPUT_ANALYTE_CSV, index=False)

    analyte_counts = analyte_map_df.groupby("gene")["analyte"].nunique()
    ptm_analytes = sorted(
        analyte_map_df.loc[analyte_map_df["analyte"].str.contains(PTM_RE, na=False), "analyte"].unique().tolist()
    )
    summary = {
        "n_top50_genes": int(len(top_genes)),
        "n_top50_genes_with_any_blood_analyte_mapping": int(analyte_counts.size),
        "n_top50_genes_with_ge_2_distinct_analytes": int((analyte_counts >= 2).sum()),
        "n_distinct_analytes": int(analyte_map_df["analyte"].nunique()),
        "n_ptm_modified_analytes": int(len(ptm_analytes)),
        "ptm_modified_analytes": ptm_analytes,
        "mapping_method_counts": dict(sorted(mapping_method_counter.items())),
        "n_ambiguous_rows_skipped": int(len(ambiguous_rows)),
        "ambiguous_rows_preview": ambiguous_rows[:30],
        "genes_without_any_mapping": [gene for gene in top_genes if gene not in analyte_counts.index],
    }
    return analyte_map_df, summary


def load_somascan_measured_universe() -> tuple[pd.DataFrame, dict]:
    raw_df = pd.read_excel(FRICK_WORKBOOK_PATH, sheet_name="Sup_table_23", header=None)
    header_row_idx = find_header_row(raw_df, ["Somamer", "Entrez gene symbol"])
    df = build_sheet_with_detected_header(raw_df, header_row_idx)
    if "Entrez gene symbol" not in df.columns:
        raise KeyError("Sup_table_23 is missing the 'Entrez gene symbol' column after header detection.")
    df = df.loc[df["Somamer"].notna()].copy().reset_index(drop=True)
    df["gene_norm"] = df["Entrez gene symbol"].map(normalize_external_gene_symbol)
    measured_df = df.loc[df["gene_norm"].notna()].copy().reset_index(drop=True)
    if measured_df.empty:
        raise RuntimeError("Sup_table_23 did not yield any canonical gene symbols.")
    summary = {
        "sheet_name": "Sup_table_23",
        "header_row_index_0based": int(header_row_idx),
        "n_rows_with_somamer": int(len(df)),
        "n_rows_with_canonical_gene": int(len(measured_df)),
        "n_unique_measured_genes": int(measured_df["gene_norm"].nunique()),
    }
    return measured_df, summary


def resolve_incident_load_model1_columns(raw_df: pd.DataFrame, header_row_idx: int) -> dict[str, int]:
    if header_row_idx <= 0:
        raise ValueError("Sup_table_2 header row has no super-header row above it.")
    header_values = ["" if pd.isna(value) else str(value).strip() for value in raw_df.iloc[header_row_idx].tolist()]
    superheader_values = [
        "" if pd.isna(value) else str(value).strip()
        for value in get_forward_filled_superheader(raw_df, header_row_idx - 1).tolist()
    ]

    def find_unique_column(header_text: str, superheader_substring: str | None = None) -> int:
        matches = []
        for idx, (header_value, superheader_value) in enumerate(zip(header_values, superheader_values, strict=True)):
            if header_value != header_text:
                continue
            if superheader_substring is not None and superheader_substring.lower() not in superheader_value.lower():
                continue
            matches.append(idx)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one column for header='{header_text}', superheader contains '{superheader_substring}', got {matches}."
            )
        return matches[0]

    return {
        "somamer": find_unique_column("Somamer"),
        "gene_symbol": find_unique_column("Gene symbol"),
        "hr_model1": find_unique_column("HR", "Model 1"),
        "p_model1": find_unique_column("P", "Model 1"),
        "fdr_model1": find_unique_column("FDR", "Model 1"),
    }


def load_incident_load_model1() -> tuple[pd.DataFrame, dict]:
    raw_df = pd.read_excel(FRICK_WORKBOOK_PATH, sheet_name="Sup_table_2", header=None)
    header_row_idx = find_header_row(raw_df, ["Somamer", "Gene symbol", "HR", "FDR"])
    column_idx = resolve_incident_load_model1_columns(raw_df, header_row_idx)
    body = raw_df.iloc[header_row_idx + 1 :].copy().reset_index(drop=True)
    selected_df = body.iloc[
        :,
        [
            column_idx["somamer"],
            column_idx["gene_symbol"],
            column_idx["hr_model1"],
            column_idx["p_model1"],
            column_idx["fdr_model1"],
        ],
    ].copy()
    selected_df.columns = ["Somamer", "Gene_symbol", "HR_model1", "P_model1", "FDR_model1"]
    selected_df = selected_df.loc[selected_df["Somamer"].notna()].copy().reset_index(drop=True)
    for column in ["HR_model1", "P_model1", "FDR_model1"]:
        selected_df[column] = pd.to_numeric(selected_df[column], errors="coerce")
    selected_df["gene_norm"] = selected_df["Gene_symbol"].map(normalize_external_gene_symbol)
    usable_df = selected_df.loc[selected_df["gene_norm"].notna()].copy().reset_index(drop=True)
    if usable_df.empty:
        raise RuntimeError("Sup_table_2 did not yield any canonical gene symbols.")
    summary = {
        "sheet_name": "Sup_table_2",
        "header_row_index_0based": int(header_row_idx),
        "column_indices": column_idx,
        "n_rows_with_somamer": int(len(selected_df)),
        "n_rows_with_canonical_gene": int(len(usable_df)),
    }
    return usable_df, summary


def coverage_stats(top_genes: list[str], measurable_set: set[str], universe_set: set[str]) -> dict:
    top_gene_set = set(top_genes)
    rest_set = universe_set - top_gene_set
    top_measurable = int(len(top_gene_set & measurable_set))
    top_not_measurable = int(len(top_gene_set - measurable_set))
    rest_measurable = int(len(rest_set & measurable_set))
    rest_not_measurable = int(len(rest_set - measurable_set))
    odds_ratio, p_value = fisher_exact(
        [[top_measurable, top_not_measurable], [rest_measurable, rest_not_measurable]],
        alternative="two-sided",
    )
    return {
        "n_top_genes": int(len(top_genes)),
        "n_top_genes_measurable": top_measurable,
        "fraction_top_genes_measurable": float(top_measurable / len(top_genes)) if top_genes else 0.0,
        "fraction_scored_universe_measurable": float(len(universe_set & measurable_set) / len(universe_set)),
        "fisher_exact_vs_rest_of_scored_universe": {
            "odds_ratio": None if math.isnan(odds_ratio) else float(odds_ratio),
            "p_value_two_sided": float(p_value),
            "contingency_table": {
                "top_measurable": top_measurable,
                "top_not_measurable": top_not_measurable,
                "rest_measurable": rest_measurable,
                "rest_not_measurable": rest_not_measurable,
            },
        },
    }


def evaluate_score_column(eval_df: pd.DataFrame, score_column: str) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    scores = eval_df[score_column].to_numpy(dtype=float)
    auroc = float(roc_auc_score(labels, scores))
    bootstrap = stratified_bootstrap_auc(labels, scores, n_bootstrap=1000, seed=42)
    rank_metrics = compute_rank_metrics_from_scores(eval_df, scores, ks=(10, 50))
    return {
        "score_column": score_column,
        "auroc": auroc,
        "auroc_bootstrap_ci95": bootstrap["ci95"],
        "auprc": float(average_precision_score(labels, scores)),
        "ranking_metrics": rank_metrics,
    }


def main() -> None:
    start_time = time.time()
    ensure_results_dir()
    if not FRICK_WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Missing Frick workbook: {FRICK_WORKBOOK_PATH}")

    scored_df, ranked_df, ranked_summary = build_scored_universe()
    scored_universe_genes = scored_df["gene"].tolist()
    scored_universe_set = set(scored_universe_genes)
    global_rank_lookup = {gene: idx + 1 for idx, gene in enumerate(ranked_df["gene"].tolist())}
    top10 = ranked_df.head(10)["gene"].tolist()
    top50 = ranked_df.head(50)["gene"].tolist()
    top100 = ranked_df.head(100)["gene"].tolist()

    analyte_map_df, analyte_summary = build_analyte_map(top50)

    somascan_df, somascan_summary = load_somascan_measured_universe()
    measurable_gene_set = set(somascan_df["gene_norm"].unique().tolist())
    coverage = {
        "top10": coverage_stats(top10, measurable_gene_set, scored_universe_set),
        "top50": coverage_stats(top50, measurable_gene_set, scored_universe_set),
        "top100": coverage_stats(top100, measurable_gene_set, scored_universe_set),
        "scored_universe": {
            "n_genes": int(len(scored_universe_genes)),
            "n_measurable": int(len(scored_universe_set & measurable_gene_set)),
            "fraction_measurable": float(len(scored_universe_set & measurable_gene_set) / len(scored_universe_set)),
        },
    }

    incident_df, incident_summary = load_incident_load_model1()
    significant_df = incident_df.loc[(incident_df["FDR_model1"] < 0.05) & incident_df["gene_norm"].isin(measurable_gene_set)].copy()
    significant_df = significant_df.sort_values(
        by=["FDR_model1", "P_model1", "Somamer"],
        ascending=[True, True, True],
        na_position="last",
    )
    representative_significant_df = significant_df.drop_duplicates("gene_norm", keep="first").copy()
    significant_gene_set = set(representative_significant_df["gene_norm"].tolist())

    evaluation_genes = sorted(scored_universe_set & measurable_gene_set)
    eval_df = scored_df.loc[scored_df["gene"].isin(evaluation_genes)].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["gene"].isin(significant_gene_set).astype(int)
    if eval_df["label"].sum() == 0:
        raise RuntimeError("No incident-LOAD significant genes overlap the measured ChronoABOR universe.")
    if eval_df["label"].sum() == len(eval_df):
        raise RuntimeError("Every measurable ChronoABOR gene is incident-LOAD significant; AUROC is undefined.")
    eval_df["global_rank_1based"] = eval_df["gene"].map(global_rank_lookup).astype(int)

    chrono_metrics = evaluate_score_column(eval_df, "chronoabor_score")
    num_studies_metrics = evaluate_score_column(eval_df, "num_studies")
    is_secreted_metrics = evaluate_score_column(eval_df, "is_secreted")

    labels = eval_df["label"].to_numpy(dtype=int)
    chrono_scores = eval_df["chronoabor_score"].to_numpy(dtype=float)
    num_studies_scores = eval_df["num_studies"].to_numpy(dtype=float)
    is_secreted_scores = eval_df["is_secreted"].to_numpy(dtype=float)
    comparisons = {
        "chronoabor_vs_num_studies": paired_stratified_bootstrap_auc_delta(
            labels,
            chrono_scores,
            num_studies_scores,
            n_bootstrap=1000,
            seed=142,
        ),
        "chronoabor_vs_is_secreted": paired_stratified_bootstrap_auc_delta(
            labels,
            chrono_scores,
            is_secreted_scores,
            n_bootstrap=1000,
            seed=242,
        ),
    }

    effect_lookup = representative_significant_df.set_index("gene_norm")["HR_model1"]
    positive_overlap_df = eval_df.loc[eval_df["label"] == 1, ["gene", "global_rank_1based"]].copy()
    positive_overlap_df["HR_model1"] = positive_overlap_df["gene"].map(effect_lookup)
    positive_overlap_df["abs_log_hr_model1"] = positive_overlap_df["HR_model1"].astype(float).map(lambda value: abs(math.log(value)))
    spearman_rho, spearman_p = spearmanr(
        positive_overlap_df["global_rank_1based"].to_numpy(dtype=float),
        positive_overlap_df["abs_log_hr_model1"].to_numpy(dtype=float),
    )

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "blood_tissue_filter": "blood|plasma|serum",
                "significant_incident_load_rule": "Sup_table_2 Model 1 FDR < 0.05, one representative aptamer per gene chosen by lowest FDR then lowest P-value.",
                "effect_size_for_correlation": "absolute log(HR_model1)",
                "scored_universe_note": (
                    "The executable ChronoABOR ranking universe built by the checked-in p7/p10 path contains "
                    f"{len(scored_universe_genes)} genes, not the manuscript's stated 2,894."
                ),
            },
            input_paths=[
                BASE_DIR / "nm_exp_p10_second_validation.py",
                BASE_DIR / "nm_exp_p7_ranked_enrichment.py",
                BASE_DIR / "signature.csv",
                FRICK_WORKBOOK_PATH,
                NM_BENCHMARK_TEX_PATH,
            ],
        ),
        "outputs": {"json": str(OUTPUT_JSON), "analyte_csv": str(OUTPUT_ANALYTE_CSV)},
        "chronoabor_ranking_source": {
            "n_scored_genes": int(len(scored_universe_genes)),
            "top10_genes": top10,
            "top50_genes": top50,
            "top100_genes": top100,
            "dataset_summary": ranked_summary,
        },
        "part_a_analyte_translation": {
            "summary": analyte_summary,
            "top_rows_preview": analyte_map_df.head(50).to_dict(orient="records"),
        },
        "part_b_somascan_coverage": {
            "somascan_summary": somascan_summary,
            "coverage": coverage,
        },
        "part_c_incident_load_prediction": {
            "incident_load_summary": {
                **incident_summary,
                "n_significant_rows_model1_measured": int(len(significant_df)),
                "n_significant_genes_model1_measured": int(len(significant_gene_set)),
                "significant_genes_model1_measured": sorted(significant_gene_set),
                "n_eval_genes_measured_and_scored": int(len(eval_df)),
                "n_eval_positives": int(eval_df["label"].sum()),
                "n_eval_negatives": int(len(eval_df) - eval_df["label"].sum()),
            },
            "models": {
                "chronoabor_score": chrono_metrics,
                "num_studies": num_studies_metrics,
                "is_secreted": is_secreted_metrics,
            },
            "paired_bootstrap_comparisons": {
                key: {
                    "observed_delta": value["observed_delta"],
                    "ci95": value["ci95"],
                    "p_value_two_sided": value["p_value_two_sided"],
                    "statistically_distinguishable_by_bootstrap_95ci": not (
                        value["ci95"]["lower"] <= 0.0 <= value["ci95"]["upper"]
                    ),
                }
                for key, value in comparisons.items()
            },
            "spearman_rank_vs_effect_size_among_positives": {
                "rho": float(spearman_rho),
                "p_value_two_sided": float(spearman_p),
                "note": "Rank is 1-based, so a negative rho means better ChronoABOR ranks align with larger absolute hazard ratios.",
                "rows": positive_overlap_df.sort_values("global_rank_1based").to_dict(orient="records"),
            },
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, to_jsonable(results))

    print(
        f"E6 analyte translation: mapped {analyte_summary['n_top50_genes_with_any_blood_analyte_mapping']}/50 top genes "
        f"to {analyte_summary['n_distinct_analytes']} distinct blood analytes; "
        f"{analyte_summary['n_top50_genes_with_ge_2_distinct_analytes']} genes have >=2 analytes."
    )
    print(
        f"SomaScan coverage: top10={coverage['top10']['n_top_genes_measurable']}/10, "
        f"top50={coverage['top50']['n_top_genes_measurable']}/50, "
        f"top100={coverage['top100']['n_top_genes_measurable']}/100, "
        f"scored universe={coverage['scored_universe']['n_measurable']}/{coverage['scored_universe']['n_genes']}."
    )
    print(
        f"AGES incident-LOAD prediction: ChronoABOR AUROC={chrono_metrics['auroc']:.3f} "
        f"[{chrono_metrics['auroc_bootstrap_ci95']['lower']:.3f}, {chrono_metrics['auroc_bootstrap_ci95']['upper']:.3f}], "
        f"AUPRC={chrono_metrics['auprc']:.3f}, P@10={chrono_metrics['ranking_metrics']['p_at_10']:.3f}, "
        f"P@50={chrono_metrics['ranking_metrics']['p_at_50']:.3f}."
    )
    print(
        f"ChronoABOR vs num_studies ΔAUROC={comparisons['chronoabor_vs_num_studies']['observed_delta']:+.3f}; "
        f"vs is_secreted ΔAUROC={comparisons['chronoabor_vs_is_secreted']['observed_delta']:+.3f}."
    )
    print(f"Outputs: {OUTPUT_JSON} and {OUTPUT_ANALYTE_CSV}")


if __name__ == "__main__":
    main()
