#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_year_external import external_year_lookup

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p3_temporal_robustness.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp_p3_robustness_heatmap.png"

RANDOM_SEED = 42
FEATURE_CUTOFF_YEAR = 2021
LABEL_START_YEAR = 2022
LABEL_END_YEAR = 2025
BASE_AUC_THRESHOLD = 0.88
FOLD_D_FEATURE_CUTOFF_YEAR = 2022
FOLD_D_LABEL_START_YEAR = 2023
FOLD_D_LABEL_END_YEAR = 2025
N_SPLITS = 5
MAX_RESOLVABLE_YEAR = 2025

VALID_GENE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
YEAR_IN_TEXT_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
GENE_SPLIT_RE = re.compile(r"[,;/]")
GENERIC_INVALID_GENE_TOKENS = {"", "NA", "N/A", "NAN", "NONE", "NULL"}
EXCLUDED_NON_GENE_TOKENS = {
    "AMYLOID",
    "CSF",
    "PET",
    "MRI",
    "APOE4",
    "P-TAU181",
    "P-TAU217",
    "ABETA",
    "A-BETA",
}
INVALID_GENE_TOKENS = GENERIC_INVALID_GENE_TOKENS | EXCLUDED_NON_GENE_TOKENS

LAYER_FILES = {
    "snp": ("snp.csv", "Gene_symbol"),
    "cpg": ("cpg.csv", "Gene_symbol"),
    "deg": ("degs.csv", "Gene_symbol"),
    "hub": ("hubgene.csv", "Gene_symbl"),
    "mirna": ("miRNA.csv", "Gene_symbol"),
    "signature": ("signature.csv", "Gene_symbol"),
}
FEATURE_COLUMNS = [
    "snp_count",
    "cpg_count",
    "deg_count",
    "deg_up_count",
    "deg_down_count",
    "deg_tissue_breadth",
    "hub_count",
    "mirna_count",
    "sig_count_all",
    "sig_count_peripheral",
    "sig_max_auc",
    "sig_count_high_auc",
    "layer_count",
    "num_studies",
    "num_deg_studies",
]


def normalize_gene_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    tokens: list[str] = []
    for part in GENE_SPLIT_RE.split(str(raw_value).upper()):
        gene = part.strip()
        if gene in INVALID_GENE_TOKENS:
            continue
        if not VALID_GENE_RE.fullmatch(gene):
            continue
        tokens.append(gene)
    return sorted(set(tokens))


def normalize_pmid(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return None


def normalize_study_id(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    pmid = normalize_pmid(raw_value)
    if pmid is not None:
        return pmid
    text = str(raw_value).strip().upper()
    return text if text else None


def normalize_abor_id(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip().upper()
    return text if text else None


def extract_year_from_doi_like(raw_value: object) -> int | None:
    if pd.isna(raw_value):
        return None
    if normalize_pmid(raw_value) is not None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    match = YEAR_IN_TEXT_RE.search(text)
    if match is None:
        return None
    year = int(match.group(1))
    if 1900 <= year <= MAX_RESOLVABLE_YEAR:
        return year
    return None


def is_peripheral_tissue(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(re.search(r"plasma|blood|serum", str(value), flags=re.IGNORECASE))


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for layer, (filename, _) in LAYER_FILES.items():
        tables[layer] = pd.read_csv(BASE_DIR / filename)
    return tables


def build_gene_universe(tables: dict[str, pd.DataFrame]) -> list[str]:
    universe: set[str] = set()
    for layer, (_, gene_column) in LAYER_FILES.items():
        for genes in tables[layer][gene_column].apply(normalize_gene_tokens):
            universe.update(genes)
    return sorted(universe)


def build_signature_pmid_year_map(signature_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    pmid_to_years: dict[str, list[int]] = {}
    for _, row in signature_df.loc[signature_df["Year_int"].notna(), ["PMID", "Year_int"]].iterrows():
        pmid = normalize_pmid(row["PMID"])
        if pmid is None:
            continue
        pmid_to_years.setdefault(pmid, []).append(int(row["Year_int"]))

    pmid_year_map = {pmid: min(years) for pmid, years in pmid_to_years.items()}
    pmid_conflicts = {pmid: len(set(years)) for pmid, years in pmid_to_years.items() if len(set(years)) > 1}
    return pmid_year_map, pmid_conflicts


def resolve_year(raw_value: object, pmid_year_map: dict[str, int], direct_year: int | None = None) -> tuple[int, str]:
    if direct_year is not None:
        return int(direct_year), "direct_year"

    pmid = normalize_pmid(raw_value)
    if pmid is not None:
        if pmid in pmid_year_map:
            return int(pmid_year_map[pmid]), "pmid_lookup"
        external = external_year_lookup(raw_value)
        if external is not None:
            return external, "external_verified"
        return 2018, "default_2018"

    external = external_year_lookup(raw_value)
    if external is not None:
        return external, "external_verified"

    extracted = extract_year_from_doi_like(raw_value)
    if extracted is not None:
        return int(extracted), "doi_extract"
    return 2018, "default_2018"


def build_first_peripheral_year(signature_df: pd.DataFrame, auc_threshold: float) -> dict[str, int]:
    first_year: dict[str, int] = {}
    peripheral_mask = signature_df["Tissue"].apply(is_peripheral_tissue)
    high_auc_mask = signature_df["AUC_num"].ge(auc_threshold)
    valid_year_mask = signature_df["Year_int"].notna()
    subset = signature_df.loc[peripheral_mask & high_auc_mask & valid_year_mask, ["Year_int", "Gene_symbol"]]
    for _, row in subset.iterrows():
        year = int(row["Year_int"])
        for gene in normalize_gene_tokens(row["Gene_symbol"]):
            first_year[gene] = min(first_year.get(gene, year), year)
    return first_year


def initialize_feature_state(gene_universe: list[str]) -> dict[str, dict]:
    feature_state: dict[str, dict] = {}
    for gene in gene_universe:
        feature_state[gene] = {
            "snp_count": 0,
            "cpg_count": 0,
            "deg_count": 0,
            "deg_up_count": 0,
            "deg_down_count": 0,
            "deg_tissue_set": set(),
            "hub_count": 0,
            "mirna_count": 0,
            "sig_count_all": 0,
            "sig_count_peripheral": 0,
            "sig_max_auc": float("nan"),
            "sig_count_high_auc": 0,
            "layer_set": set(),
            "study_ids": set(),
            "deg_study_ids": set(),
        }
    return feature_state


def add_study_id(record: dict, raw_study_id: object) -> None:
    study_id = normalize_study_id(raw_study_id)
    if study_id is not None:
        record["study_ids"].add(study_id)


def build_features_and_evidence(
    tables: dict[str, pd.DataFrame],
    gene_universe: list[str],
    pmid_year_map: dict[str, int],
    cutoff_year: int,
) -> tuple[pd.DataFrame, set[str], dict[str, Counter]]:
    feature_state = initialize_feature_state(gene_universe)
    evidence_by_cutoff: set[str] = set()
    year_source_counts: dict[str, Counter] = {layer: Counter() for layer in LAYER_FILES}

    for _, row in tables["snp"].iterrows():
        year, source = resolve_year(row.get("PMID"), pmid_year_map)
        year_source_counts["snp"][source] += 1
        if year > cutoff_year:
            continue
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            record = feature_state[gene]
            record["snp_count"] += 1
            record["layer_set"].add("snp")
            add_study_id(record, row.get("PMID"))
            evidence_by_cutoff.add(gene)

    for _, row in tables["cpg"].iterrows():
        year, source = resolve_year(row.get("PMID"), pmid_year_map)
        year_source_counts["cpg"][source] += 1
        if year > cutoff_year:
            continue
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            record = feature_state[gene]
            record["cpg_count"] += 1
            record["layer_set"].add("cpg")
            add_study_id(record, row.get("PMID"))
            evidence_by_cutoff.add(gene)

    for _, row in tables["deg"].iterrows():
        year, source = resolve_year(row.get("PMID_DOI"), pmid_year_map)
        year_source_counts["deg"][source] += 1
        if year > cutoff_year:
            continue
        alteration = str(row.get("Alteration", "")).lower()
        tissue = str(row.get("Tissue", "")).strip()
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            record = feature_state[gene]
            record["deg_count"] += 1
            if "up" in alteration:
                record["deg_up_count"] += 1
            if "down" in alteration:
                record["deg_down_count"] += 1
            if tissue:
                record["deg_tissue_set"].add(tissue)
            record["layer_set"].add("deg")
            study_id = normalize_study_id(row.get("PMID_DOI"))
            if study_id is not None:
                record["study_ids"].add(study_id)
                record["deg_study_ids"].add(study_id)
            evidence_by_cutoff.add(gene)

    for _, row in tables["hub"].iterrows():
        year, source = resolve_year(row.get("PMID"), pmid_year_map)
        year_source_counts["hub"][source] += 1
        if year > cutoff_year:
            continue
        for gene in normalize_gene_tokens(row.get("Gene_symbl")):
            record = feature_state[gene]
            record["hub_count"] += 1
            record["layer_set"].add("hub")
            add_study_id(record, row.get("PMID"))
            evidence_by_cutoff.add(gene)

    for _, row in tables["mirna"].iterrows():
        year, source = resolve_year(row.get("PMID_DOI"), pmid_year_map)
        year_source_counts["mirna"][source] += 1
        if year > cutoff_year:
            continue
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            record = feature_state[gene]
            record["mirna_count"] += 1
            record["layer_set"].add("mirna")
            add_study_id(record, row.get("PMID_DOI"))
            evidence_by_cutoff.add(gene)

    for _, row in tables["signature"].iterrows():
        direct_year = None
        if pd.notna(row.get("Year_int")):
            direct_year = int(row["Year_int"])
        year, source = resolve_year(row.get("PMID"), pmid_year_map, direct_year=direct_year)
        year_source_counts["signature"][source] += 1
        if year > cutoff_year:
            continue
        auc = row.get("AUC_num")
        peripheral = is_peripheral_tissue(row.get("Tissue"))
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            record = feature_state[gene]
            record["sig_count_all"] += 1
            if peripheral:
                record["sig_count_peripheral"] += 1
            if peripheral and pd.notna(auc) and float(auc) >= 0.80:
                record["sig_count_high_auc"] += 1
            if pd.notna(auc):
                if pd.isna(record["sig_max_auc"]):
                    record["sig_max_auc"] = float(auc)
                else:
                    record["sig_max_auc"] = max(float(record["sig_max_auc"]), float(auc))
            record["layer_set"].add("signature")
            add_study_id(record, row.get("PMID"))
            evidence_by_cutoff.add(gene)

    rows = []
    for gene in gene_universe:
        record = feature_state[gene]
        rows.append(
            {
                "gene": gene,
                "snp_count": int(record["snp_count"]),
                "cpg_count": int(record["cpg_count"]),
                "deg_count": int(record["deg_count"]),
                "deg_up_count": int(record["deg_up_count"]),
                "deg_down_count": int(record["deg_down_count"]),
                "deg_tissue_breadth": int(len(record["deg_tissue_set"])),
                "hub_count": int(record["hub_count"]),
                "mirna_count": int(record["mirna_count"]),
                "sig_count_all": int(record["sig_count_all"]),
                "sig_count_peripheral": int(record["sig_count_peripheral"]),
                "sig_max_auc": float(0.0 if pd.isna(record["sig_max_auc"]) else record["sig_max_auc"]),
                "sig_count_high_auc": int(record["sig_count_high_auc"]),
                "layer_count": int(len(record["layer_set"])),
                "num_studies": int(len(record["study_ids"])),
                "num_deg_studies": int(len(record["deg_study_ids"])),
                "has_evidence_by_cutoff": bool(gene in evidence_by_cutoff),
            }
        )

    feature_df = pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)
    return feature_df, evidence_by_cutoff, year_source_counts


def assign_labels(
    gene_universe: list[str],
    evidence_by_cutoff: set[str],
    first_peripheral_year: dict[str, int],
    label_start_year: int,
    label_end_year: int,
) -> pd.DataFrame:
    matured_by_label_end = {gene for gene, year in first_peripheral_year.items() if year <= label_end_year}
    positive_genes = {
        gene for gene, year in first_peripheral_year.items() if label_start_year <= year <= label_end_year
    }
    negative_genes = {gene for gene in evidence_by_cutoff if gene not in matured_by_label_end}

    rows = []
    for gene in gene_universe:
        if gene in positive_genes:
            label = 1
            label_status = "positive"
        elif gene in negative_genes:
            label = 0
            label_status = "negative"
        elif gene in matured_by_label_end:
            label = np.nan
            label_status = "historical_positive_excluded"
        else:
            label = np.nan
            label_status = "unlabeled"
        rows.append(
            {
                "gene": gene,
                "label": label,
                "label_status": label_status,
                "first_peripheral_high_auc_year": first_peripheral_year.get(gene),
            }
        )

    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def build_eval_dataframe(
    tables: dict[str, pd.DataFrame],
    gene_universe: list[str],
    pmid_year_map: dict[str, int],
    feature_cutoff_year: int,
    label_start_year: int,
    label_end_year: int,
    auc_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_peripheral_year = build_first_peripheral_year(tables["signature"], auc_threshold=auc_threshold)
    feature_df, evidence_by_cutoff, _ = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=feature_cutoff_year,
    )
    label_df = assign_labels(
        gene_universe=gene_universe,
        evidence_by_cutoff=evidence_by_cutoff,
        first_peripheral_year=first_peripheral_year,
        label_start_year=label_start_year,
        label_end_year=label_end_year,
    )
    full_df = feature_df.merge(label_df, on="gene", how="left")
    eval_df = full_df.loc[full_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    if eval_df.empty:
        raise RuntimeError("No labeled genes remain after applying the temporal split.")
    eval_df["label"] = eval_df["label"].astype(int)
    return full_df, eval_df


def evaluate_cv(
    eval_df: pd.DataFrame,
    splitter,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    X = eval_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    min_class_count = int(class_counts.min())
    requested_splits = getattr(splitter, "n_splits", None)

    if isinstance(splitter, StratifiedKFold) and requested_splits is not None and min_class_count < requested_splits:
        note = (
            f"Requested StratifiedKFold({requested_splits}) is infeasible because the minority class has "
            f"only {min_class_count} sample(s)."
        )
        return {
            "cv_feasible": False,
            "note": note,
            "fold_aurocs": [],
            "auroc_mean": None,
            "auroc_std": None,
        }

    if isinstance(splitter, GroupKFold):
        if groups is None:
            raise ValueError("GroupKFold requires a groups array.")
        unique_groups = int(pd.Series(groups).nunique())
        if requested_splits is not None and unique_groups < requested_splits:
            note = f"Requested GroupKFold({requested_splits}) is infeasible because only {unique_groups} groups exist."
            return {
                "cv_feasible": False,
                "note": note,
                "fold_aurocs": [],
                "auroc_mean": None,
                "auroc_std": None,
            }

    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        if len(np.unique(y[train_idx])) < 2:
            note = f"Fold {fold_idx} training set has a single class; AUROC training is invalid."
            return {
                "cv_feasible": False,
                "note": note,
                "fold_aurocs": fold_aurocs,
                "auroc_mean": None,
                "auroc_std": None,
            }
        if len(np.unique(y[test_idx])) < 2:
            note = f"Fold {fold_idx} test set has a single class; AUROC is undefined."
            return {
                "cv_feasible": False,
                "note": note,
                "fold_aurocs": fold_aurocs,
                "auroc_mean": None,
                "auroc_std": None,
            }
        model = make_xgb()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        fold_aurocs.append(float(roc_auc_score(y[test_idx], probs)))

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
    return {
        "cv_feasible": True,
        "note": None,
        "fold_aurocs": fold_aurocs,
        "auroc_mean": mean_auroc,
        "auroc_std": std_auroc,
    }


def build_gene_study_groups(signature_df: pd.DataFrame, gene_universe: list[str]) -> dict[str, str]:
    gene_to_abors: dict[str, Counter] = {}
    for _, row in signature_df.iterrows():
        abor_id = normalize_abor_id(row.get("ABOR_ID"))
        for gene in normalize_gene_tokens(row.get("Gene_symbol")):
            if abor_id is None:
                continue
            gene_to_abors.setdefault(gene, Counter())[abor_id] += 1

    gene_group_map: dict[str, str] = {}
    for gene in gene_universe:
        abor_counts = gene_to_abors.get(gene)
        if not abor_counts:
            gene_group_map[gene] = f"singleton_{gene}"
            continue
        gene_group_map[gene] = sorted(abor_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return gene_group_map


def build_heatmap(rows: list[tuple[str, float, int]]) -> None:
    conditions = [row[0] for row in rows]
    raw_values = np.array([[row[1], float(row[2])] for row in rows], dtype=float)
    normalized = raw_values.copy()

    for col_idx in range(normalized.shape[1]):
        column = normalized[:, col_idx]
        valid = column[~np.isnan(column)]
        if len(valid) == 0:
            normalized[:, col_idx] = 0.5
            continue
        col_min = float(np.min(valid))
        col_max = float(np.max(valid))
        if math.isclose(col_min, col_max):
            normalized[:, col_idx] = np.where(np.isnan(column), 0.5, 0.5)
        else:
            normalized[:, col_idx] = np.where(
                np.isnan(column),
                0.5,
                (column - col_min) / (col_max - col_min),
            )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(normalized, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=["AUROC", "n_positives"])
    ax.set_yticks(np.arange(len(conditions)), labels=conditions)
    ax.set_title("Temporal Robustness Heatmap")

    for row_idx in range(raw_values.shape[0]):
        for col_idx in range(raw_values.shape[1]):
            raw_value = raw_values[row_idx, col_idx]
            if np.isnan(raw_value):
                text = "NA"
            elif col_idx == 0:
                text = f"{raw_value:.3f}"
            else:
                text = f"{int(raw_value)}"
            text_color = "white" if normalized[row_idx, col_idx] >= 0.55 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=10)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Column-normalized intensity")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    gene_study_group_map = build_gene_study_groups(tables["signature"], gene_universe)

    _, baseline_eval_df = build_eval_dataframe(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        feature_cutoff_year=FEATURE_CUTOFF_YEAR,
        label_start_year=LABEL_START_YEAR,
        label_end_year=LABEL_END_YEAR,
        auc_threshold=BASE_AUC_THRESHOLD,
    )
    baseline_n_positives = int(baseline_eval_df["label"].sum())
    baseline_n_negatives = int((1 - baseline_eval_df["label"]).sum())

    baseline_groups = baseline_eval_df["gene"].map(gene_study_group_map).to_numpy()
    study_blocked_cv = evaluate_cv(
        eval_df=baseline_eval_df,
        splitter=GroupKFold(n_splits=N_SPLITS),
        groups=baseline_groups,
    )
    study_blocked_mean = study_blocked_cv["auroc_mean"]
    study_blocked_std = study_blocked_cv["auroc_std"]

    stratified_cv = evaluate_cv(
        eval_df=baseline_eval_df,
        splitter=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    )
    stratified_mean = stratified_cv["auroc_mean"]
    stratified_std = stratified_cv["auroc_std"]

    _, exclude_2024_eval_df = build_eval_dataframe(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        feature_cutoff_year=FEATURE_CUTOFF_YEAR,
        label_start_year=LABEL_START_YEAR,
        label_end_year=2023,
        auc_threshold=BASE_AUC_THRESHOLD,
    )
    exclude_2024_cv = evaluate_cv(
        eval_df=exclude_2024_eval_df,
        splitter=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    )
    auroc_2023 = exclude_2024_cv["auroc_mean"]
    auroc_2023_std = exclude_2024_cv["auroc_std"]
    n_positives_2023 = int(exclude_2024_eval_df["label"].sum())

    _, strict_eval_df = build_eval_dataframe(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        feature_cutoff_year=FEATURE_CUTOFF_YEAR,
        label_start_year=LABEL_START_YEAR,
        label_end_year=LABEL_END_YEAR,
        auc_threshold=0.90,
    )
    strict_cv = evaluate_cv(
        eval_df=strict_eval_df,
        splitter=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    )
    strict_mean = strict_cv["auroc_mean"]
    strict_std = strict_cv["auroc_std"]

    _, lenient_eval_df = build_eval_dataframe(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        feature_cutoff_year=FEATURE_CUTOFF_YEAR,
        label_start_year=LABEL_START_YEAR,
        label_end_year=LABEL_END_YEAR,
        auc_threshold=0.80,
    )
    lenient_cv = evaluate_cv(
        eval_df=lenient_eval_df,
        splitter=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    )
    lenient_mean = lenient_cv["auroc_mean"]
    lenient_std = lenient_cv["auroc_std"]

    _, fold_d_eval_df = build_eval_dataframe(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        feature_cutoff_year=FOLD_D_FEATURE_CUTOFF_YEAR,
        label_start_year=FOLD_D_LABEL_START_YEAR,
        label_end_year=FOLD_D_LABEL_END_YEAR,
        auc_threshold=BASE_AUC_THRESHOLD,
    )
    fold_d_cv = evaluate_cv(
        eval_df=fold_d_eval_df,
        splitter=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    )
    fold_d_mean = fold_d_cv["auroc_mean"]
    fold_d_std = fold_d_cv["auroc_std"]

    heatmap_rows = [
        ("study_blocked", study_blocked_mean, baseline_n_positives),
        ("stratified_cv", stratified_mean, baseline_n_positives),
        ("exclude_2024", auroc_2023, n_positives_2023),
        ("label_def_A", strict_mean, int(strict_eval_df["label"].sum())),
        ("label_def_B", stratified_mean, baseline_n_positives),
        ("label_def_C", lenient_mean, int(lenient_eval_df["label"].sum())),
        ("fold_d", fold_d_mean, int(fold_d_eval_df["label"].sum())),
    ]
    build_heatmap(heatmap_rows)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_columns": FEATURE_COLUMNS,
        "feature_cutoff_year_default": FEATURE_CUTOFF_YEAR,
        "label_window_default": f"{LABEL_START_YEAR}-{LABEL_END_YEAR}",
        "n_genes_universe": int(len(gene_universe)),
        "signature_pmid_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "sub_experiment_1_study_blocked_split": {
            "n_eval_genes": int(len(baseline_eval_df)),
            "n_positives": baseline_n_positives,
            "n_negatives": baseline_n_negatives,
            "n_unique_gene_groups": int(pd.Series(baseline_groups).nunique()),
            "study_group_rule": "most frequent signature ABOR_ID per gene; singleton if missing",
            "study_blocked_cv_feasible": study_blocked_cv["cv_feasible"],
            "study_blocked_note": study_blocked_cv["note"],
            "study_blocked_fold_aurocs": study_blocked_cv["fold_aurocs"],
            "study_blocked_auroc_mean": study_blocked_mean,
            "study_blocked_auroc_std": study_blocked_std,
            "stratified_cv_feasible": stratified_cv["cv_feasible"],
            "stratified_note": stratified_cv["note"],
            "stratified_fold_aurocs": stratified_cv["fold_aurocs"],
            "stratified_auroc_mean": stratified_mean,
            "stratified_auroc_std": stratified_std,
            "delta": None if study_blocked_mean is None or stratified_mean is None else float(study_blocked_mean - stratified_mean),
        },
        "sub_experiment_2_exclude_2024_sensitivity": {
            "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
            "label_start_year": LABEL_START_YEAR,
            "label_end_year_2023": 2023,
            "label_end_year_2025": LABEL_END_YEAR,
            "cv_feasible_2023": exclude_2024_cv["cv_feasible"],
            "note_2023": exclude_2024_cv["note"],
            "auroc_2023_fold_values": exclude_2024_cv["fold_aurocs"],
            "auroc_2023": auroc_2023,
            "auroc_2023_std": auroc_2023_std,
            "n_positives_2023": n_positives_2023,
            "n_eval_genes_2023": int(len(exclude_2024_eval_df)),
            "auroc_2025": stratified_mean,
            "auroc_2025_std": stratified_std,
            "n_positives_2025": baseline_n_positives,
            "n_eval_genes_2025": int(len(baseline_eval_df)),
        },
        "sub_experiment_3_alternative_label_definitions": {
            "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
            "label_window": f"{LABEL_START_YEAR}-{LABEL_END_YEAR}",
            "definition_A": {
                "auc_threshold": 0.90,
                "cv_feasible": strict_cv["cv_feasible"],
                "note": strict_cv["note"],
                "fold_aurocs": strict_cv["fold_aurocs"],
                "def_A_auroc": strict_mean,
                "def_A_auroc_std": strict_std,
                "def_A_n_pos": int(strict_eval_df["label"].sum()),
                "n_eval_genes": int(len(strict_eval_df)),
            },
            "definition_B": {
                "auc_threshold": 0.88,
                "cv_feasible": stratified_cv["cv_feasible"],
                "note": stratified_cv["note"],
                "fold_aurocs": stratified_cv["fold_aurocs"],
                "def_B_auroc": stratified_mean,
                "def_B_auroc_std": stratified_std,
                "def_B_n_pos": baseline_n_positives,
                "n_eval_genes": int(len(baseline_eval_df)),
            },
            "definition_C": {
                "auc_threshold": 0.80,
                "cv_feasible": lenient_cv["cv_feasible"],
                "note": lenient_cv["note"],
                "fold_aurocs": lenient_cv["fold_aurocs"],
                "def_C_auroc": lenient_mean,
                "def_C_auroc_std": lenient_std,
                "def_C_n_pos": int(lenient_eval_df["label"].sum()),
                "n_eval_genes": int(len(lenient_eval_df)),
            },
            "def_A_auroc": strict_mean,
            "def_A_n_pos": int(strict_eval_df["label"].sum()),
            "def_B_auroc": stratified_mean,
            "def_B_n_pos": baseline_n_positives,
            "def_C_auroc": lenient_mean,
            "def_C_n_pos": int(lenient_eval_df["label"].sum()),
        },
        "sub_experiment_4_fold_d_validation": {
            "feature_cutoff_year": FOLD_D_FEATURE_CUTOFF_YEAR,
            "label_start_year": FOLD_D_LABEL_START_YEAR,
            "label_end_year": FOLD_D_LABEL_END_YEAR,
            "cv_feasible": fold_d_cv["cv_feasible"],
            "note": fold_d_cv["note"],
            "fold_d_fold_aurocs": fold_d_cv["fold_aurocs"],
            "fold_d_auroc_mean": fold_d_mean,
            "fold_d_auroc_std": fold_d_std,
            "fold_d_n_positives": int(fold_d_eval_df["label"].sum()),
            "fold_d_n_eval_genes": int(len(fold_d_eval_df)),
        },
        "outputs": {
            "json": str(OUTPUT_JSON),
            "heatmap_png": str(OUTPUT_PLOT),
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("NM Experiment P3: Temporal robustness evaluation complete")
    print(
        f"Baseline setup (features <= {FEATURE_CUTOFF_YEAR}, labels {LABEL_START_YEAR}-{LABEL_END_YEAR}): "
        f"{len(baseline_eval_df)} eval genes, {baseline_n_positives} positives, {baseline_n_negatives} negatives"
    )
    delta_text = (
        "NA" if study_blocked_mean is None or stratified_mean is None else f"{study_blocked_mean - stratified_mean:+.4f}"
    )
    print(
        f"1. Study-blocked GroupKFold AUROC: {format_metric(study_blocked_mean)} +/- {format_metric(study_blocked_std)} "
        f"(delta vs stratified = {delta_text})"
    )
    print(f"   StratifiedKFold AUROC: {format_metric(stratified_mean)} +/- {format_metric(stratified_std)}")
    print(
        f"2. Exclude-2024 labels (2022-2023) AUROC: {format_metric(auroc_2023)} +/- {format_metric(auroc_2023_std)} "
        f"(n_positives={n_positives_2023})"
    )
    if exclude_2024_cv["note"] is not None:
        print(f"   Note: {exclude_2024_cv['note']}")
    print(
        f"   Baseline labels (2022-2025) AUROC: {format_metric(stratified_mean)} +/- {format_metric(stratified_std)} "
        f"(n_positives={baseline_n_positives})"
    )
    print(
        f"3. Label definition A (AUC >= 0.90) AUROC: {format_metric(strict_mean)} +/- {format_metric(strict_std)} "
        f"(n_positives={int(strict_eval_df['label'].sum())})"
    )
    print(
        f"   Label definition B (AUC >= 0.88) AUROC: {format_metric(stratified_mean)} +/- {format_metric(stratified_std)} "
        f"(n_positives={baseline_n_positives})"
    )
    print(
        f"   Label definition C (AUC >= 0.80) AUROC: {format_metric(lenient_mean)} +/- {format_metric(lenient_std)} "
        f"(n_positives={int(lenient_eval_df['label'].sum())})"
    )
    print(
        f"4. Fold D validation (features <= {FOLD_D_FEATURE_CUTOFF_YEAR}, labels {FOLD_D_LABEL_START_YEAR}-{FOLD_D_LABEL_END_YEAR}) "
        f"AUROC: {format_metric(fold_d_mean)} +/- {format_metric(fold_d_std)} "
        f"(n_positives={int(fold_d_eval_df['label'].sum())})"
    )
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Heatmap saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
