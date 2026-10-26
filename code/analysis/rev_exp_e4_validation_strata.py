#!/usr/bin/env python3

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rev_exp_common import (
    ALL_LAYER_SPECS,
    BASE_DIR,
    FULL_FEATURES,
    RESULTS_DIR,
    annotate_tables_with_study_metadata,
    build_external_fold_dataset,
    build_scored_universe,
    build_within_secreted_fold_c_frame,
    compute_rank_metrics_from_scores,
    cross_validated_scores,
    ensure_results_dir,
    full_fit_scores,
    load_all_tables,
    make_fixed_xgb,
    normalize_pmid,
    normalize_gene_tokens,
    paired_stratified_bootstrap_auc_delta,
    provenance,
    shap_mean_abs_for_features,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
EXPERIMENTAL_FEATURES = ["n_experimental_evidence", "frac_experimental_evidence"]
CPG_STUDY_METHOD_ANNOTATIONS_PATH = (
    BASE_DIR / "revision-1" / "source_evidence" / "cpg_validation_20260917" / "cpg_study_method_annotations.csv"
)
RECORD_VALIDATION_REVIEW_PATH = (
    BASE_DIR / "revision-1" / "source_evidence" / "validation_review_20260918" / "record_validation_review.csv"
)
NONREVIEW_RECORD_CORRECTIONS_PATH = (
    BASE_DIR / "revision-1" / "source_evidence" / "validation_review_20260918" / "nonreview_record_validation_corrections.csv"
)
REVIEW_BASELINE_UNCLASSIFIED_PATH = (
    BASE_DIR / "revision-1" / "ABOR_record_validation_tiers.unclassified97.pre_review97_20260918.csv"
)
REVIEW_BASELINE_RECORD_EXPORT_PATH = (
    BASE_DIR / "revision-1" / "ABOR_record_validation_tiers.pre_review97_20260918.csv"
)
CPG_ANNOTATION_FIELDS = [
    "PMID",
    "DOI",
    "Source_title",
    "ABOR_record_count",
    "Method",
    "Validation_Strategy",
    "Validation_Tier",
    "Validation_Tier_Rationale",
    "Source_URL",
    "Source_sections",
    "Reviewed_date",
]
CPG_METHOD_FIELDS = ["Method", "Validation_Strategy"]
RECORD_VALIDATION_REVIEW_FIELDS = [
    "row_id",
    "layer",
    "abor_id",
    "original_study_id",
    "original_method_text",
    "normalized_method",
    "evidence_subtype",
    "assigned_tier",
    "source_url",
    "source_title",
    "source_sections",
    "review_rationale",
    "review_status",
    "unresolved_reason",
    "reviewed_date",
    "provenance_flag",
    "validation_evidence_year",
]
REVIEW_PAYLOAD_FIELDS = [
    "normalized_method",
    "evidence_subtype",
    "source_url",
    "source_title",
    "source_sections",
    "review_rationale",
    "review_status",
    "unresolved_reason",
    "reviewed_date",
    "provenance_flag",
    "validation_evidence_year",
]
METHOD_FIELDS = {
    "deg": ["Measuring_omics", "platform_info_id"],
    "snp": ["Method"],
    "cpg": CPG_METHOD_FIELDS,
    "hub": ["Method"],
    "mirna": ["Measuring_omics"],
    "signature": ["Method", "Validation_Strategy"],
    "target": ["ExperimentType", "ExperimentModel", "RegulatoryApproval"],
}
METHOD_NORMALIZATION_RULES = [
    {
        "rule_id": "unicode_dash_to_ascii_hyphen",
        "description": "Map Unicode hyphen/dash/minus characters to ASCII hyphen before regex matching.",
    },
    {
        "rule_id": "collapse_whitespace_and_hyphen_spacing",
        "description": "Collapse repeated whitespace and remove spaces around hyphens.",
    },
    {
        "rule_id": "rna_sequencing_synonym",
        "description": "Normalize RNA sequencing wording to RNA-seq.",
    },
    {
        "rule_id": "single_molecule_array_synonym",
        "description": "Normalize single molecule/molecular array wording to SIMOA.",
    },
    {
        "rule_id": "meso_scale_discovery_synonym",
        "description": "Normalize Meso Scale Discovery hyphenation to the existing targeted-assay phrase.",
    },
]
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)
_METHOD_SYNONYMS = [
    (re.compile(r"\bRNA\s+sequencing\b", flags=re.IGNORECASE), "RNA-seq"),
    (re.compile(r"\bsingle\s+molecul(?:e|ar)\s+array(?:\s+method)?\b", flags=re.IGNORECASE), "SIMOA"),
    (re.compile(r"\bMeso\s*-\s*Scale\s+Discovery\b", flags=re.IGNORECASE), "Meso Scale Discovery"),
]
CLINICAL_NEGATION_PATTERNS = [
    r"\b(?:no|not|without|lacks?|lack of)\b.{0,40}\b(?:clinical trial|regulatory approval|approved|clinical validation|prospective cohort)\b",
    r"\bnot\s+approved\b",
    r"\bunapproved\b",
]
VALIDATION_RULES = [
    {
        "rule_id": "tier3_clinical_trial_or_approval",
        "tier": "tier3_clinical",
        "pattern": r"\bclinical trial\b|\bphase\s*[1234]\w*\b|\bregulatory approval\b|\bapproved\b|\bclinical validation\b",
        "note": "Clinical trial, clinical validation, or explicit regulatory approval.",
    },
    {
        "rule_id": "tier3_prospective_cohort",
        "tier": "tier3_clinical",
        "pattern": r"\bprospective cohort\b",
        "note": "Prospective cohort wording in the validation metadata.",
    },
    {
        "rule_id": "tier2_targeted_assay",
        "tier": "tier2_targeted_assay",
        "pattern": (
            r"elisa|western blot|qpcr|qrt-pcr|rt-q?pcr|immunoassay|simoa|meso scale discovery|"
            r"\bmsd\b|mass spectrometry|lc-ms|lc/ms|ms/ms|maldi|immunohistochemistry|"
            r"flow cytometry|immunofluorescence|immunoprecipitation|nulisa"
        ),
        "note": "Targeted biochemical/molecular assay evidence. Functional cell, animal, or organoid validation is assigned only through curated review.",
    },
    {
        "rule_id": "tier1_omics_or_in_silico",
        "tier": "tier1_computational",
        "pattern": (
            r"microarray|rna-?seq|snrna-?seq|scrna-?seq|single cell transcriptome|mirna-?seq|mirna array|"
            r"gwas|wgcna|network analysis|bioinformatics|in silico|machine learning|deep learning|"
            r"xgboost|lasso|svm|roc analysis|cross-?validation|meta-analysis|ppi network|gsea|"
            r"cibersort|genie3|transcriptomics|proteomics|ai-assisted|multi-omics|ngs"
        ),
        "note": "Discovery-omics or computational-only evidence without an explicit targeted/clinical assay.",
    },
]
COMPILED_RULES = [
    {**rule, "regex": re.compile(rule["pattern"], flags=re.IGNORECASE)} for rule in VALIDATION_RULES
]
TIER_ORDER = [
    "tier1_computational",
    "tier2_targeted_assay",
    "tier3_clinical",
    "unclassified",
]
RULE_EVIDENCE_SUBTYPES = {
    "tier3_clinical_trial_or_approval": "clinical_or_regulatory_validation",
    "tier3_prospective_cohort": "prospective_clinical_cohort",
    "tier2_targeted_assay": "targeted_biochemical_or_molecular_assay",
    "tier1_omics_or_in_silico": "discovery_omics_or_computational",
}


def join_method_text(row: pd.Series, fields: list[str]) -> str:
    values = []
    for field in fields:
        value = row.get(field)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return " | ".join(values)


def normalize_method_text(method_text: object) -> str:
    if method_text is None or pd.isna(method_text):
        return ""
    text = str(method_text).translate(_DASH_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    for pattern, replacement in _METHOD_SYNONYMS:
        text = pattern.sub(replacement, text)
    return text


def is_negated_clinical_claim(normalized_method_text: str) -> bool:
    lower = normalized_method_text.lower()
    return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in CLINICAL_NEGATION_PATTERNS)


def evidence_subtype_for_rule(matched_rule: str | None) -> str:
    if matched_rule is None:
        return ""
    return RULE_EVIDENCE_SUBTYPES.get(matched_rule, "")


def classify_validation_tier(layer: str, method_text: str) -> tuple[str, str | None]:
    normalized_method_text = normalize_method_text(method_text)
    for rule in COMPILED_RULES:
        if not rule["regex"].search(normalized_method_text):
            continue
        if rule["tier"] == "tier3_clinical" and is_negated_clinical_claim(normalized_method_text):
            continue
        return rule["tier"], rule["rule_id"]
    return "unclassified", None


def _pmid_key(value) -> str:
    pmid = normalize_pmid(value)
    if pmid:
        return pmid
    return str(value).strip()


def load_cpg_study_method_annotations(cpg_df: pd.DataFrame) -> pd.DataFrame:
    if not CPG_STUDY_METHOD_ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing CpG method annotation manifest: {CPG_STUDY_METHOD_ANNOTATIONS_PATH}")

    annotations = pd.read_csv(CPG_STUDY_METHOD_ANNOTATIONS_PATH, dtype=str).fillna("")
    missing = [field for field in CPG_ANNOTATION_FIELDS if field not in annotations.columns]
    if missing:
        raise ValueError(f"CpG method annotation manifest is missing fields: {missing}")
    annotations = annotations[CPG_ANNOTATION_FIELDS].copy()
    annotations["_pmid_key"] = annotations["PMID"].map(_pmid_key)

    duplicate_pmids = annotations.loc[annotations["_pmid_key"].duplicated(), "PMID"].tolist()
    if duplicate_pmids:
        raise ValueError(f"CpG method annotation manifest contains duplicate PMIDs: {duplicate_pmids}")

    observed_counts = cpg_df["PMID"].map(_pmid_key).value_counts().sort_index()
    annotated_counts = annotations.set_index("_pmid_key")["ABOR_record_count"].astype(int).sort_index()
    if set(observed_counts.index) != set(annotated_counts.index):
        raise ValueError(
            "CpG method annotation PMIDs do not match cpg.csv PMIDs: "
            f"observed={sorted(observed_counts.index)}, annotated={sorted(annotated_counts.index)}"
        )
    mismatched_counts = {
        pmid: {"cpg_csv": int(observed_counts[pmid]), "annotation": int(annotated_counts[pmid])}
        for pmid in observed_counts.index
        if int(observed_counts[pmid]) != int(annotated_counts[pmid])
    }
    if mismatched_counts:
        raise ValueError(f"CpG method annotation record counts do not match cpg.csv: {mismatched_counts}")

    for field in CPG_METHOD_FIELDS + ["Validation_Tier", "Validation_Tier_Rationale"]:
        empty_pmids = annotations.loc[annotations[field].str.strip().eq(""), "PMID"].tolist()
        if empty_pmids:
            raise ValueError(f"CpG method annotation field {field} is empty for PMIDs: {empty_pmids}")

    invalid_tiers = annotations.loc[annotations["Validation_Tier"] != "tier1_computational", ["PMID", "Validation_Tier"]]
    if not invalid_tiers.empty:
        raise ValueError(f"CpG method annotation tiers must be tier1_computational: {invalid_tiers.to_dict(orient='records')}")

    evidence_dir = CPG_STUDY_METHOD_ANNOTATIONS_PATH.parent
    missing_sources = []
    for pmid in annotations["PMID"]:
        for suffix in [".pdf", ".txt"]:
            path = evidence_dir / f"PMID{pmid}{suffix}"
            if not path.exists():
                missing_sources.append(str(path))
    if missing_sources:
        raise FileNotFoundError(f"Missing CpG source evidence files: {missing_sources}")

    return annotations


def apply_cpg_study_method_annotations(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if "cpg" not in tables:
        raise KeyError("Expected cpg table in loaded ABOR tables.")

    annotated_tables = {layer: df.copy() for layer, df in tables.items()}
    cpg_df = annotated_tables["cpg"].copy()
    original_columns = list(cpg_df.columns)
    original_abor_ids = cpg_df["ABOR_ID"].astype(str).tolist() if "ABOR_ID" in cpg_df.columns else None
    annotations = load_cpg_study_method_annotations(cpg_df)

    cpg_df["_pmid_key"] = cpg_df["PMID"].map(_pmid_key)
    annotation_payload = annotations.drop(columns=["PMID"]).copy()
    merged = cpg_df.merge(annotation_payload, on="_pmid_key", how="left", validate="many_to_one")
    merged = merged.drop(columns=["_pmid_key"])

    if len(merged) != len(cpg_df):
        raise RuntimeError("CpG method annotation merge changed the row count.")
    if original_abor_ids is not None and merged["ABOR_ID"].astype(str).tolist() != original_abor_ids:
        raise RuntimeError("CpG method annotation merge changed ABOR_ID order.")
    if list(merged.columns[: len(original_columns)]) != original_columns:
        raise RuntimeError("CpG method annotation merge changed the original cpg.csv column order.")
    if merged[CPG_METHOD_FIELDS].isna().any().any() or merged[CPG_METHOD_FIELDS].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise RuntimeError("CpG method annotation merge left empty method fields.")

    annotated_tables["cpg"] = merged
    return annotated_tables, annotations


def load_all_tables_with_cpg_annotations() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    return apply_cpg_study_method_annotations(load_all_tables())


def _clean_review_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def blank_review_payload() -> dict[str, str]:
    return {field: "" for field in REVIEW_PAYLOAD_FIELDS + ["review_assigned_tier"]}


def load_review_target_records(
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
) -> pd.DataFrame:
    if not review_target_path.exists():
        return pd.DataFrame(columns=RECORD_VALIDATION_REVIEW_FIELDS)
    target = pd.read_csv(review_target_path, dtype=str).fillna("")
    required = ["row_id", "layer", "abor_id", "study_id", "study_id_raw", "method_text"]
    missing = [field for field in required if field not in target.columns]
    if missing:
        raise ValueError(f"Review baseline unclassified export is missing fields: {missing}")
    duplicates = target.loc[target["row_id"].duplicated(), "row_id"].tolist()
    if duplicates:
        raise ValueError(f"Review baseline unclassified export contains duplicate row_id values: {duplicates}")
    return target


def _canonical_study_id_values(*values: object) -> set[str]:
    cleaned = {_clean_review_value(value) for value in values if _clean_review_value(value)}
    pmids = {normalize_pmid(value) for value in cleaned}
    cleaned.update({pmid for pmid in pmids if pmid})
    return cleaned


def _parse_review_year(value: object, row_id: str) -> int | None:
    text = _clean_review_value(value)
    if not text:
        return None
    if not re.fullmatch(r"\d{4}", text):
        raise ValueError(f"{row_id}: validation_evidence_year must be a four-digit year")
    year = int(text)
    if year < 1900 or year > 2100:
        raise ValueError(f"{row_id}: validation_evidence_year is outside the accepted range")
    return year


def load_record_validation_review(
    current_row_df: pd.DataFrame,
    review_path: Path = RECORD_VALIDATION_REVIEW_PATH,
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
) -> tuple[pd.DataFrame, dict]:
    target = load_review_target_records(review_target_path)
    target_count = int(len(target))
    if not review_path.exists():
        return pd.DataFrame(columns=RECORD_VALIDATION_REVIEW_FIELDS), {
            "path": str(review_path),
            "target_path": str(review_target_path),
            "status": "manifest_not_present",
            "target_record_count": target_count,
            "reviewed_record_count": 0,
            "assigned_tier_counts": {},
            "evidence_subtype_counts": {},
            "provenance_flag_counts": {},
            "validation_evidence_year_counts": {},
            "unresolved_reason_counts": {},
        }

    if target.empty:
        raise FileNotFoundError(f"Review manifest exists but target baseline is missing or empty: {review_target_path}")

    review = pd.read_csv(review_path, dtype=str).fillna("")
    missing = [field for field in RECORD_VALIDATION_REVIEW_FIELDS if field not in review.columns]
    if missing:
        raise ValueError(f"Record validation review manifest is missing fields: {missing}")
    review = review[RECORD_VALIDATION_REVIEW_FIELDS].copy()
    for field in RECORD_VALIDATION_REVIEW_FIELDS:
        review[field] = review[field].map(_clean_review_value)

    duplicates = review.loc[review["row_id"].duplicated(), "row_id"].tolist()
    if duplicates:
        raise ValueError(f"Record validation review manifest contains duplicate row_id values: {duplicates}")

    target_ids = set(target["row_id"])
    review_ids = set(review["row_id"])
    if review_ids != target_ids:
        missing_ids = sorted(target_ids - review_ids)
        extra_ids = sorted(review_ids - target_ids)
        raise ValueError(f"Record validation review manifest must cover the 97 baseline unclassified rows exactly: missing={missing_ids}, extra={extra_ids}")

    current_by_id = current_row_df.set_index("row_id", drop=False)
    target_by_id = target.set_index("row_id", drop=False)
    errors: list[str] = []
    for _, row in review.iterrows():
        row_id = row["row_id"]
        if row_id not in current_by_id.index:
            errors.append(f"{row_id}: row_id is absent from current records")
            continue
        current = current_by_id.loc[row_id]
        baseline = target_by_id.loc[row_id]
        if row["layer"] != _clean_review_value(baseline["layer"]) or row["layer"] != _clean_review_value(current["layer"]):
            errors.append(f"{row_id}: layer mismatch")
        if row["abor_id"] != _clean_review_value(baseline["abor_id"]) or row["abor_id"] != _clean_review_value(current["abor_id"]):
            errors.append(f"{row_id}: ABOR_ID mismatch")
        if row["original_method_text"] != _clean_review_value(baseline["method_text"]):
            errors.append(f"{row_id}: original_method_text mismatch")
        allowed_study_ids = _canonical_study_id_values(baseline["study_id"], baseline["study_id_raw"], current["study_id"])
        if row["original_study_id"] not in allowed_study_ids:
            errors.append(f"{row_id}: original_study_id mismatch")
        if row["assigned_tier"] not in TIER_ORDER:
            errors.append(f"{row_id}: assigned_tier must be one of {TIER_ORDER}")
        if not row["normalized_method"]:
            errors.append(f"{row_id}: normalized_method is empty")
        if not row["review_status"]:
            errors.append(f"{row_id}: review_status is empty")
        if not row["provenance_flag"]:
            errors.append(f"{row_id}: provenance_flag is empty")
        if row["assigned_tier"] == "unclassified":
            if not row["unresolved_reason"]:
                errors.append(f"{row_id}: unresolved unclassified rows require unresolved_reason")
        else:
            for required_field in ["evidence_subtype", "source_url", "source_title", "source_sections", "review_rationale", "reviewed_date"]:
                if not row[required_field]:
                    errors.append(f"{row_id}: {required_field} is required for assigned tiers")
            try:
                if _parse_review_year(row["validation_evidence_year"], row_id) is None:
                    errors.append(f"{row_id}: validation_evidence_year is required for assigned tiers")
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError("Record validation review manifest failed validation: " + "; ".join(errors[:20]))

    audit = {
        "path": str(review_path),
        "target_path": str(review_target_path),
        "status": "loaded",
        "target_record_count": target_count,
        "reviewed_record_count": int(len(review)),
        "assigned_tier_counts": {tier: int(count) for tier, count in review["assigned_tier"].value_counts().sort_index().items()},
        "evidence_subtype_counts": {subtype: int(count) for subtype, count in review["evidence_subtype"].loc[review["evidence_subtype"] != ""].value_counts().sort_index().items()},
        "provenance_flag_counts": {flag: int(count) for flag, count in review["provenance_flag"].loc[review["provenance_flag"] != ""].value_counts().sort_index().items()},
        "validation_evidence_year_counts": {year: int(count) for year, count in review["validation_evidence_year"].loc[review["validation_evidence_year"] != ""].value_counts().sort_index().items()},
        "unresolved_reason_counts": {reason: int(count) for reason, count in review["unresolved_reason"].loc[review["unresolved_reason"] != ""].value_counts().sort_index().items()},
    }
    return review, audit


def load_nonreview_record_corrections(
    current_row_df: pd.DataFrame,
    correction_path: Path = NONREVIEW_RECORD_CORRECTIONS_PATH,
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
    baseline_record_path: Path = REVIEW_BASELINE_RECORD_EXPORT_PATH,
) -> tuple[pd.DataFrame, dict]:
    if not correction_path.exists():
        return pd.DataFrame(columns=RECORD_VALIDATION_REVIEW_FIELDS), {
            "path": str(correction_path),
            "status": "manifest_not_present",
            "corrected_record_count": 0,
            "assigned_tier_counts": {},
        }

    corrections = pd.read_csv(correction_path, dtype=str).fillna("")
    missing = [field for field in RECORD_VALIDATION_REVIEW_FIELDS if field not in corrections.columns]
    if missing:
        raise ValueError(f"Non-review record correction manifest is missing fields: {missing}")
    corrections = corrections[RECORD_VALIDATION_REVIEW_FIELDS].copy()
    for field in RECORD_VALIDATION_REVIEW_FIELDS:
        corrections[field] = corrections[field].map(_clean_review_value)

    duplicates = corrections.loc[corrections["row_id"].duplicated(), "row_id"].tolist()
    if duplicates:
        raise ValueError(f"Non-review record correction manifest contains duplicate row_id values: {duplicates}")

    target = load_review_target_records(review_target_path)
    review_target_ids = set(target["row_id"]) if not target.empty else set()
    overlapping_ids = sorted(set(corrections["row_id"]) & review_target_ids)
    if overlapping_ids:
        raise ValueError(f"Non-review corrections must not overlap the 97-row review manifest: {overlapping_ids}")
    if not baseline_record_path.exists():
        raise FileNotFoundError(f"Non-review corrections require baseline record export: {baseline_record_path}")

    baseline = pd.read_csv(baseline_record_path, dtype=str).fillna("").set_index("row_id", drop=False)
    current = current_row_df.set_index("row_id", drop=False)
    errors: list[str] = []
    for _, row in corrections.iterrows():
        row_id = row["row_id"]
        if row_id not in baseline.index or row_id not in current.index:
            errors.append(f"{row_id}: row_id is absent from baseline or current records")
            continue
        baseline_row = baseline.loc[row_id]
        current_row = current.loc[row_id]
        if row["layer"] != _clean_review_value(baseline_row["layer"]) or row["layer"] != _clean_review_value(current_row["layer"]):
            errors.append(f"{row_id}: layer mismatch")
        if row["abor_id"] != _clean_review_value(baseline_row["abor_id"]) or row["abor_id"] != _clean_review_value(current_row["abor_id"]):
            errors.append(f"{row_id}: ABOR_ID mismatch")
        if row["original_method_text"] != _clean_review_value(baseline_row["method_text"]):
            errors.append(f"{row_id}: original_method_text mismatch")
        allowed_study_ids = _canonical_study_id_values(baseline_row["study_id"], baseline_row["study_id_raw"], current_row["study_id"])
        if row["original_study_id"] not in allowed_study_ids:
            errors.append(f"{row_id}: original_study_id mismatch")
        if row["assigned_tier"] not in TIER_ORDER:
            errors.append(f"{row_id}: assigned_tier must be one of {TIER_ORDER}")
        if not row["normalized_method"]:
            errors.append(f"{row_id}: normalized_method is empty")
        if not row["review_status"]:
            errors.append(f"{row_id}: review_status is empty")
        if not row["provenance_flag"]:
            errors.append(f"{row_id}: provenance_flag is empty")
        if row["assigned_tier"] == "unclassified":
            if not row["unresolved_reason"]:
                errors.append(f"{row_id}: unresolved unclassified rows require unresolved_reason")
        else:
            for required_field in ["evidence_subtype", "source_url", "source_title", "source_sections", "review_rationale", "reviewed_date"]:
                if not row[required_field]:
                    errors.append(f"{row_id}: {required_field} is required for assigned tiers")
            try:
                if _parse_review_year(row["validation_evidence_year"], row_id) is None:
                    errors.append(f"{row_id}: validation_evidence_year is required for assigned tiers")
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError("Non-review record correction manifest failed validation: " + "; ".join(errors[:20]))

    audit = {
        "path": str(correction_path),
        "status": "loaded",
        "corrected_record_count": int(len(corrections)),
        "assigned_tier_counts": {tier: int(count) for tier, count in corrections["assigned_tier"].value_counts().sort_index().items()},
        "evidence_subtype_counts": {subtype: int(count) for subtype, count in corrections["evidence_subtype"].loc[corrections["evidence_subtype"] != ""].value_counts().sort_index().items()},
    }
    return corrections, audit


def apply_curated_assignments(
    reviewed_rows: pd.DataFrame,
    reviewed_genes: pd.DataFrame,
    assignments: pd.DataFrame,
    matched_rule_id: str,
    unresolved_rule_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty:
        return reviewed_rows, reviewed_genes
    assignment_by_id = assignments.set_index("row_id", drop=False)
    for row_id, assignment in assignment_by_id.iterrows():
        mask = reviewed_rows["row_id"] == row_id
        validation_year = _parse_review_year(assignment["validation_evidence_year"], row_id)
        reviewed_rows.loc[mask, "tier"] = assignment["assigned_tier"]
        reviewed_rows.loc[mask, "matched_rule_id"] = (
            matched_rule_id if assignment["assigned_tier"] != "unclassified" else unresolved_rule_id
        )
        reviewed_rows.loc[mask, "review_assigned_tier"] = assignment["assigned_tier"]
        if validation_year is not None:
            original_year = int(reviewed_rows.loc[mask, "resolved_year"].iloc[0])
            reviewed_rows.loc[mask, "evidence_year"] = max(original_year, validation_year)
            reviewed_rows.loc[mask, "evidence_year_source"] = "max_association_and_validation_evidence_year"
        for field in REVIEW_PAYLOAD_FIELDS:
            reviewed_rows.loc[mask, field] = assignment[field]

    row_lookup = reviewed_rows.set_index("row_id")
    propagated_fields = [
        "tier",
        "matched_rule_id",
        "normalized_method",
        "evidence_subtype",
        "review_assigned_tier",
        "evidence_year",
        "evidence_year_source",
    ] + REVIEW_PAYLOAD_FIELDS
    for field in propagated_fields:
        reviewed_genes[field] = reviewed_genes["row_id"].map(row_lookup[field]).fillna(reviewed_genes[field])
    return reviewed_rows, reviewed_genes


def apply_record_validation_review(
    row_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    review_path: Path = RECORD_VALIDATION_REVIEW_PATH,
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
    correction_path: Path = NONREVIEW_RECORD_CORRECTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    reviewed_rows = row_df.copy()
    reviewed_genes = gene_df.copy()
    for field, value in blank_review_payload().items():
        if field not in reviewed_rows.columns:
            reviewed_rows[field] = value
        if field not in reviewed_genes.columns:
            reviewed_genes[field] = value

    review, audit = load_record_validation_review(reviewed_rows, review_path, review_target_path)
    reviewed_rows, reviewed_genes = apply_curated_assignments(
        reviewed_rows,
        reviewed_genes,
        review,
        matched_rule_id="curated_review_20260918",
        unresolved_rule_id="curated_review_unresolved_20260918",
    )
    corrections, correction_audit = load_nonreview_record_corrections(reviewed_rows, correction_path=correction_path)
    reviewed_rows, reviewed_genes = apply_curated_assignments(
        reviewed_rows,
        reviewed_genes,
        corrections,
        matched_rule_id="curated_nonreview_correction_20260918",
        unresolved_rule_id="curated_nonreview_unresolved_20260918",
    )
    audit["nonreview_record_correction_audit"] = correction_audit
    return reviewed_rows, reviewed_genes, audit


def build_record_validation_review_template(
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
) -> pd.DataFrame:
    target = load_review_target_records(review_target_path)
    rows = []
    for _, row in target.iterrows():
        original_method_text = _clean_review_value(row["method_text"])
        rows.append(
            {
                "row_id": _clean_review_value(row["row_id"]),
                "layer": _clean_review_value(row["layer"]),
                "abor_id": _clean_review_value(row["abor_id"]),
                "original_study_id": _clean_review_value(row["study_id"]),
                "original_method_text": original_method_text,
                "normalized_method": normalize_method_text(original_method_text),
                "evidence_subtype": "",
                "assigned_tier": "",
                "source_url": "",
                "source_title": "",
                "source_sections": "",
                "review_rationale": "",
                "review_status": "pending_review",
                "unresolved_reason": "",
                "reviewed_date": "",
                "provenance_flag": "",
                "validation_evidence_year": "",
            }
        )
    return pd.DataFrame(rows, columns=RECORD_VALIDATION_REVIEW_FIELDS)


def build_nonreview_tier_change_audit(
    current_row_df: pd.DataFrame,
    baseline_record_path: Path = REVIEW_BASELINE_RECORD_EXPORT_PATH,
    review_target_path: Path = REVIEW_BASELINE_UNCLASSIFIED_PATH,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "row_id",
        "layer",
        "abor_id",
        "study_id",
        "method_text_before",
        "method_text_current",
        "normalized_method_current",
        "tier_before",
        "tier_current",
        "matched_rule_before",
        "matched_rule_current",
        "evidence_subtype_current",
        "review_status",
        "provenance_flag",
        "change_reason",
    ]
    if not baseline_record_path.exists():
        return pd.DataFrame(columns=columns), {
            "status": "baseline_record_export_not_present",
            "baseline_record_path": str(baseline_record_path),
            "changed_record_count_outside_review97": 0,
        }

    baseline = pd.read_csv(baseline_record_path, dtype=str).fillna("")
    target = load_review_target_records(review_target_path)
    review_target_ids = set(target["row_id"]) if not target.empty else set()
    current = current_row_df.copy()
    current["matched_rule"] = current["matched_rule_id"].fillna("")
    old_cols = ["row_id", "layer", "abor_id", "study_id", "method_text", "tier", "matched_rule"]
    new_cols = ["row_id", "method_text", "normalized_method", "tier", "matched_rule", "evidence_subtype", "review_status", "provenance_flag"]
    merged = baseline[old_cols].merge(current[new_cols], on="row_id", suffixes=("_before", "_current"), how="inner")
    merged = merged.loc[~merged["row_id"].isin(review_target_ids)].copy()
    changed = merged.loc[merged["tier_before"] != merged["tier_current"]].copy()
    if changed.empty:
        return pd.DataFrame(columns=columns), {
            "status": "computed",
            "baseline_record_path": str(baseline_record_path),
            "changed_record_count_outside_review97": 0,
            "tier_transitions": {},
        }

    changed["change_reason"] = np.select(
        [
            changed["matched_rule_current"].eq("curated_nonreview_correction_20260918"),
            changed["tier_before"].eq("tier3_clinical") & changed["tier_current"].ne("tier3_clinical"),
        ],
        [
            "curated_nonreview_correction_20260918",
            "generic_rule_guard_removed_preclinical_or_negated_clinical_match",
        ],
        default="generic_rule_or_manifest_changed_non_review97_record",
    )
    out = changed.rename(
        columns={
            "method_text_before": "method_text_before",
            "method_text_current": "method_text_current",
            "normalized_method": "normalized_method_current",
            "tier_before": "tier_before",
            "tier_current": "tier_current",
            "matched_rule_before": "matched_rule_before",
            "matched_rule_current": "matched_rule_current",
            "evidence_subtype": "evidence_subtype_current",
        }
    )
    out = out[columns].sort_values(["layer", "row_id"]).reset_index(drop=True)
    transitions = {
        f"{old} -> {new}": int(count)
        for (old, new), count in out.groupby(["tier_before", "tier_current"]).size().sort_index().items()
    }
    summary = {
        "status": "computed",
        "baseline_record_path": str(baseline_record_path),
        "review_target_path": str(review_target_path),
        "changed_record_count_outside_review97": int(len(out)),
        "tier_transitions": transitions,
    }
    return out, summary


def summarize_tier_counts(df: pd.DataFrame, group_key: str | None = None) -> dict:
    if group_key is None:
        total = int(len(df))
        counts = df["tier"].value_counts().to_dict()
        return {
            tier: {
                "count": int(counts.get(tier, 0)),
                "fraction": float(counts.get(tier, 0) / total) if total else 0.0,
            }
            for tier in TIER_ORDER
        }

    result = {}
    for group_value, group_df in df.groupby(group_key):
        result[str(group_value)] = summarize_tier_counts(group_df)
        result[str(group_value)]["n_records"] = int(len(group_df))
    return result


def collect_validation_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tables, cpg_annotations = load_all_tables_with_cpg_annotations()
    annotated_tables, pmid_year_map, pmid_conflicts = annotate_tables_with_study_metadata(tables)

    row_records: list[dict] = []
    gene_records: list[dict] = []
    top_raw_values = {}
    for layer, spec in ALL_LAYER_SPECS.items():
        df = annotated_tables[layer]
        top_raw_values[layer] = {}
        for field in METHOD_FIELDS[layer]:
            values = df[field].fillna("").astype(str).str.strip()
            value_counts = values.loc[values != ""].value_counts().head(30)
            top_raw_values[layer][field] = [
                {"raw_value": raw_value, "count": int(count)}
                for raw_value, count in value_counts.items()
            ]
        if not METHOD_FIELDS[layer]:
            top_raw_values[layer]["_note"] = "No method-like validation field exists in this layer."

        for row_idx, row in df.iterrows():
            method_text = join_method_text(row, METHOD_FIELDS[layer]) if METHOD_FIELDS[layer] else "[no method column]"
            if not method_text:
                method_text = "[empty method fields]"
            normalized_method = normalize_method_text(method_text)
            tier, matched_rule = classify_validation_tier(layer, method_text)
            genes = normalize_gene_tokens(row[spec["gene_column"]])
            row_records.append(
                {
                    "layer": layer,
                    "record_index": int(row_idx),
                    "row_id": f"{layer}:{row_idx}",
                    "abor_id": row.get("ABOR_ID", row.get("TargetID", "")),
                    "gene": ";".join(genes),
                    "gene_field_raw": row[spec["gene_column"]],
                    "study_id_type": row["_study_id_type"],
                    "study_id_field": spec["study_column"],
                    "study_id_raw": row[spec["study_column"]],
                    "tier": tier,
                    "matched_rule_id": matched_rule,
                    "method_text": method_text,
                    "normalized_method": normalized_method,
                    "method_fields": "|".join(METHOD_FIELDS[layer]),
                    "evidence_subtype": evidence_subtype_for_rule(matched_rule),
                    "resolved_year": int(row["_resolved_year"]),
                    "evidence_year": int(row["_resolved_year"]),
                    "evidence_year_source": "association_publication_year",
                    "year_resolution_source": row["_year_resolution_source"],
                    "study_id": row["_study_id"],
                    "source_doi": row.get("DOI", ""),
                    "source_title": row.get("Source_title", ""),
                    "manifest_validation_tier": row.get("Validation_Tier", ""),
                    "validation_tier_rationale": row.get("Validation_Tier_Rationale", ""),
                    "source_url": row.get("Source_URL", ""),
                    "source_sections": row.get("Source_sections", ""),
                    "reviewed_date": row.get("Reviewed_date", ""),
                    "review_rationale": "",
                    "review_status": "",
                    "unresolved_reason": "",
                    "review_assigned_tier": "",
                    "provenance_flag": "",
                    "validation_evidence_year": "",
                }
            )

            for gene in genes:
                gene_records.append(
                    {
                        "layer": layer,
                        "row_id": f"{layer}:{row_idx}",
                        "gene": gene,
                        "tier": tier,
                        "matched_rule_id": matched_rule,
                        "method_text": method_text,
                        "normalized_method": normalized_method,
                        "evidence_subtype": evidence_subtype_for_rule(matched_rule),
                        "resolved_year": int(row["_resolved_year"]),
                        "evidence_year": int(row["_resolved_year"]),
                        "evidence_year_source": "association_publication_year",
                        "year_resolution_source": row["_year_resolution_source"],
                        "study_id": row["_study_id"],
                        "review_rationale": "",
                        "review_status": "",
                        "unresolved_reason": "",
                        "review_assigned_tier": "",
                        "provenance_flag": "",
                        "validation_evidence_year": "",
                    }
                )

    row_level_df = pd.DataFrame(row_records)
    gene_record_df = pd.DataFrame(gene_records)
    row_level_df, gene_record_df, review_audit = apply_record_validation_review(row_level_df, gene_record_df)
    nonreview_tier_change_df, nonreview_tier_change_summary = build_nonreview_tier_change_audit(row_level_df)
    metadata = {
        "pmid_year_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "method_fields_by_layer": METHOD_FIELDS,
        "method_normalization_rules": METHOD_NORMALIZATION_RULES,
        "clinical_negation_patterns": CLINICAL_NEGATION_PATTERNS,
        "record_validation_review_path": str(RECORD_VALIDATION_REVIEW_PATH),
        "nonreview_record_corrections_path": str(NONREVIEW_RECORD_CORRECTIONS_PATH),
        "record_validation_review_target_path": str(REVIEW_BASELINE_UNCLASSIFIED_PATH),
        "record_validation_review_audit": review_audit,
        "nonreview_tier_change_audit": nonreview_tier_change_summary,
        "nonreview_tier_change_records": nonreview_tier_change_df.to_dict(orient="records"),
        "cpg_study_method_annotations_path": str(CPG_STUDY_METHOD_ANNOTATIONS_PATH),
        "cpg_annotation_fields": CPG_ANNOTATION_FIELDS,
        "cpg_study_method_annotations": cpg_annotations[CPG_ANNOTATION_FIELDS].to_dict(orient="records"),
    }
    return row_level_df, gene_record_df, {**metadata, "top_raw_values": top_raw_values}


def build_experimental_feature_frame(gene_record_df: pd.DataFrame, genes: list[str], cutoff_year: int) -> pd.DataFrame:
    evidence_year_column = "evidence_year" if "evidence_year" in gene_record_df.columns else "resolved_year"
    denominator_cutoff_df = gene_record_df.loc[gene_record_df["resolved_year"] <= cutoff_year].copy()
    experimental_cutoff_df = gene_record_df.loc[
        gene_record_df["tier"].isin(["tier2_targeted_assay", "tier3_clinical"])
        & gene_record_df[evidence_year_column].le(cutoff_year)
    ].copy()
    total_counts = denominator_cutoff_df.groupby("gene").size()
    experimental_counts = experimental_cutoff_df.groupby("gene").size()

    feature_df = pd.DataFrame({"gene": genes})
    feature_df["n_total_records_all7"] = feature_df["gene"].map(total_counts).fillna(0).astype(int)
    feature_df["n_experimental_evidence"] = feature_df["gene"].map(experimental_counts).fillna(0).astype(int)
    feature_df["frac_experimental_evidence"] = np.where(
        feature_df["n_total_records_all7"] > 0,
        feature_df["n_experimental_evidence"] / feature_df["n_total_records_all7"],
        0.0,
    )
    return feature_df


def evaluate_ablation(
    endpoint_name: str,
    eval_df: pd.DataFrame,
    ranking_score_source: str,
    bootstrap_seed: int,
) -> dict:
    baseline_features = FULL_FEATURES
    augmented_features = FULL_FEATURES + EXPERIMENTAL_FEATURES
    labels = eval_df["label"].to_numpy(dtype=int)

    baseline_cv = cross_validated_scores(eval_df, baseline_features, make_fixed_xgb)
    augmented_cv = cross_validated_scores(eval_df, augmented_features, make_fixed_xgb)
    delta = paired_stratified_bootstrap_auc_delta(
        labels,
        augmented_cv["oof_scores"],
        baseline_cv["oof_scores"],
        n_bootstrap=1000,
        seed=bootstrap_seed,
    )

    if ranking_score_source == "full_fit":
        baseline_rank_scores = full_fit_scores(eval_df, baseline_features, make_fixed_xgb)
        augmented_rank_scores = full_fit_scores(eval_df, augmented_features, make_fixed_xgb)
    elif ranking_score_source == "oof":
        baseline_rank_scores = baseline_cv["oof_scores"]
        augmented_rank_scores = augmented_cv["oof_scores"]
    else:
        raise ValueError(f"Unsupported ranking_score_source: {ranking_score_source}")

    shap_values = shap_mean_abs_for_features(
        eval_df,
        augmented_features,
        make_fixed_xgb,
        EXPERIMENTAL_FEATURES,
    )
    baseline_rank = compute_rank_metrics_from_scores(eval_df, baseline_rank_scores, ks=(5, 10, 50))
    augmented_rank = compute_rank_metrics_from_scores(eval_df, augmented_rank_scores, ks=(5, 10, 50))

    return {
        "endpoint_name": endpoint_name,
        "feature_columns_baseline": baseline_features,
        "feature_columns_augmented": augmented_features,
        "ranking_score_source": ranking_score_source,
        "baseline": {
            "auroc_mean": baseline_cv["auroc_mean"],
            "auroc_std": baseline_cv["auroc_std"],
            "auroc_oof": baseline_cv["auroc_oof"],
            "auprc_oof": baseline_cv["auprc_oof"],
            "ranking_metrics": baseline_rank,
        },
        "augmented": {
            "auroc_mean": augmented_cv["auroc_mean"],
            "auroc_std": augmented_cv["auroc_std"],
            "auroc_oof": augmented_cv["auroc_oof"],
            "auprc_oof": augmented_cv["auprc_oof"],
            "ranking_metrics": augmented_rank,
            "shap_mean_abs_new_features": shap_values,
        },
        "delta_augmented_minus_baseline": {
            "observed_delta_auroc": delta["observed_delta"],
            "ci95": delta["ci95"],
            "p_value_two_sided": delta["p_value_two_sided"],
            "statistically_distinguishable_by_bootstrap_95ci": not (
                delta["ci95"]["lower"] <= 0.0 <= delta["ci95"]["upper"]
            ),
            "delta_p_at_5": augmented_rank.get("p_at_5", float("nan")) - baseline_rank.get("p_at_5", float("nan")),
            "delta_p_at_10": augmented_rank.get("p_at_10", float("nan")) - baseline_rank.get("p_at_10", float("nan")),
        },
    }


def print_top_raw_method_values(top_raw_values: dict) -> None:
    print("Top raw method/validation strings by layer")
    for layer in ALL_LAYER_SPECS:
        print(f"[{layer}]")
        layer_payload = top_raw_values[layer]
        note = layer_payload.get("_note")
        if note is not None:
            print(f"  NOTE: {note}")
        for field, rows in layer_payload.items():
            if field == "_note":
                continue
            print(f"  {field}")
            for row in rows:
                print(f"    {row['count']:>4}  {row['raw_value']}")


def main() -> None:
    start_time = time.time()
    ensure_results_dir()

    row_level_df, gene_record_df, metadata = collect_validation_frames()
    secreted_df, secreted_summary = build_within_secreted_fold_c_frame()
    _, ranked_df, ranked_summary = build_scored_universe()
    external_df, external_summary, _ = build_external_fold_dataset("C")

    secreted_gene_set = set(secreted_df["gene"])
    top50_genes = ranked_df.head(50)["gene"].tolist()
    top50_gene_set = set(top50_genes)

    experimental_features = build_experimental_feature_frame(
        gene_record_df,
        genes=sorted(set(gene_record_df["gene"])),
        cutoff_year=2021,
    )
    experimental_lookup = experimental_features.set_index("gene")

    secreted_eval = secreted_df.copy()
    external_eval = external_df.copy()
    for eval_df in [secreted_eval, external_eval]:
        eval_df["n_experimental_evidence"] = (
            eval_df["gene"].map(experimental_lookup["n_experimental_evidence"]).fillna(0).astype(int)
        )
        eval_df["frac_experimental_evidence"] = (
            eval_df["gene"].map(experimental_lookup["frac_experimental_evidence"]).fillna(0.0).astype(float)
        )

    within_ablation = evaluate_ablation(
        endpoint_name="within_secreted_fold_c",
        eval_df=secreted_eval,
        ranking_score_source="full_fit",
        bootstrap_seed=142,
    )
    external_ablation = evaluate_ablation(
        endpoint_name="external_fold_c",
        eval_df=external_eval,
        ranking_score_source="oof",
        bootstrap_seed=242,
    )

    row_level_distribution = {
        "overall": summarize_tier_counts(row_level_df),
        "by_layer": summarize_tier_counts(row_level_df, "layer"),
    }
    gene_record_distribution = {
        "within_secreted_35": summarize_tier_counts(gene_record_df.loc[gene_record_df["gene"].isin(secreted_gene_set)]),
        "chronoabor_top50": summarize_tier_counts(gene_record_df.loc[gene_record_df["gene"].isin(top50_gene_set)]),
    }
    experimental_feature_summary = {
        "n_genes_with_any_cutoff_record": int((experimental_features["n_total_records_all7"] > 0).sum()),
        "n_genes_with_any_experimental_record": int((experimental_features["n_experimental_evidence"] > 0).sum()),
        "distribution_n_experimental_evidence": {
            "mean": float(experimental_features["n_experimental_evidence"].mean()),
            "median": float(experimental_features["n_experimental_evidence"].median()),
            "max": float(experimental_features["n_experimental_evidence"].max()),
        },
        "distribution_frac_experimental_evidence": {
            "mean": float(experimental_features["frac_experimental_evidence"].mean()),
            "median": float(experimental_features["frac_experimental_evidence"].median()),
            "max": float(experimental_features["frac_experimental_evidence"].max()),
        },
    }

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "cutoff_year": 2021,
                "feature_columns_baseline": FULL_FEATURES,
                "feature_columns_augmented": FULL_FEATURES + EXPERIMENTAL_FEATURES,
                "feature_set_note": (
                    "The checked-in executable pipeline uses 15 dynamic nm_exp4 FEATURE_COLUMNS plus is_secreted. "
                    "The manuscript's separate 10-feature / is_ad_driver formulation is not implemented in the listed source files."
                ),
                "record_validation_review_path": str(RECORD_VALIDATION_REVIEW_PATH),
                "nonreview_record_corrections_path": str(NONREVIEW_RECORD_CORRECTIONS_PATH),
                "validation_rules": VALIDATION_RULES,
                "method_normalization_rules": METHOD_NORMALIZATION_RULES,
                "distribution_note": (
                    "Overall/per-layer counts are row-level ABOR records. The within-secreted-35 and ChronoABOR-top50 summaries are "
                    "gene-record instances after expanding multi-gene rows to the genes they reference."
                ),
                "experimental_feature_year_note": (
                    "The total-record denominator uses the original association publication year. The Tier 2/3 numerator uses "
                    "max(association publication year, validation_evidence_year) for curated review evidence, so later support evidence cannot enter an earlier cutoff."
                ),
            },
            input_paths=[
                BASE_DIR / "nm_exp4_full_universe.py",
                BASE_DIR / "nm_exp_p6_secreted_ablation.py",
                BASE_DIR / "nm_exp_p10_second_validation.py",
                BASE_DIR / "paper" / "supplementary_table_s1.tex",
                BASE_DIR / "degs.csv",
                BASE_DIR / "snp.csv",
                BASE_DIR / "cpg.csv",
                CPG_STUDY_METHOD_ANNOTATIONS_PATH,
                RECORD_VALIDATION_REVIEW_PATH,
                NONREVIEW_RECORD_CORRECTIONS_PATH,
                REVIEW_BASELINE_UNCLASSIFIED_PATH,
                BASE_DIR / "hubgene.csv",
                BASE_DIR / "miRNA.csv",
                BASE_DIR / "signature.csv",
                BASE_DIR / "target.csv",
            ],
        ),
        "outputs": {"json": str(OUTPUT_JSON)},
        "dataset_summaries": {
            "within_secreted": secreted_summary,
            "external_fold_c": external_summary,
            "chronoabor_ranking_source": {
                "n_ranked_genes": int(len(ranked_df)),
                "top50_genes": top50_genes,
                "p7_dataset_summary": ranked_summary,
            },
        },
        "classification_audit": {
            "method_fields_by_layer": metadata["method_fields_by_layer"],
            "method_normalization_rules": metadata["method_normalization_rules"],
            "clinical_negation_patterns": metadata["clinical_negation_patterns"],
            "record_validation_review_path": metadata["record_validation_review_path"],
            "nonreview_record_corrections_path": metadata["nonreview_record_corrections_path"],
            "record_validation_review_target_path": metadata["record_validation_review_target_path"],
            "record_validation_review_audit": metadata["record_validation_review_audit"],
            "nonreview_tier_change_audit": metadata["nonreview_tier_change_audit"],
            "nonreview_tier_change_records": metadata["nonreview_tier_change_records"],
            "cpg_study_method_annotations_path": metadata["cpg_study_method_annotations_path"],
            "cpg_annotation_fields": metadata["cpg_annotation_fields"],
            "cpg_study_method_annotations": metadata["cpg_study_method_annotations"],
            "regex_to_tier_mapping": VALIDATION_RULES,
            "top_raw_values_by_layer_and_field": metadata["top_raw_values"],
        },
        "tier_distribution": {
            "row_level": row_level_distribution,
            "gene_record_restricted_sets": gene_record_distribution,
        },
        "experimental_feature_summary": experimental_feature_summary,
        "feature_preview_top50": experimental_features.loc[
            experimental_features["gene"].isin(top50_gene_set)
        ].sort_values(
            by=["n_experimental_evidence", "frac_experimental_evidence", "gene"],
            ascending=[False, False, True],
        ).to_dict(orient="records"),
        "ablation": {
            "within_secreted_fold_c": within_ablation,
            "external_fold_c": external_ablation,
        },
        "pmid_year_lookup_size": metadata["pmid_year_lookup_size"],
        "signature_pmid_year_conflicts": metadata["signature_pmid_year_conflicts"],
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, results)

    print(
        f"E4 validation strata: overall tier2+tier3 row fraction="
        f"{row_level_distribution['overall']['tier2_targeted_assay']['fraction'] + row_level_distribution['overall']['tier3_clinical']['fraction']:.3f}; "
        f"within-secreted ΔAUROC={within_ablation['delta_augmented_minus_baseline']['observed_delta_auroc']:+.3f} "
        f"[{within_ablation['delta_augmented_minus_baseline']['ci95']['lower']:+.3f}, "
        f"{within_ablation['delta_augmented_minus_baseline']['ci95']['upper']:+.3f}]."
    )
    print(
        f"External Fold C ΔAUROC={external_ablation['delta_augmented_minus_baseline']['observed_delta_auroc']:+.3f} "
        f"[{external_ablation['delta_augmented_minus_baseline']['ci95']['lower']:+.3f}, "
        f"{external_ablation['delta_augmented_minus_baseline']['ci95']['upper']:+.3f}]."
    )
    print(
        f"SHAP new features (external): n_experimental_evidence="
        f"{external_ablation['augmented']['shap_mean_abs_new_features']['n_experimental_evidence']:.3f}, "
        f"frac_experimental_evidence="
        f"{external_ablation['augmented']['shap_mean_abs_new_features']['frac_experimental_evidence']:.3f}."
    )
    print_top_raw_method_values(metadata["top_raw_values"])
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
