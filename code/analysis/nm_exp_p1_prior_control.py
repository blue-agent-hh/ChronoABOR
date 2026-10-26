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
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_year_external import external_year_lookup

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p1_prior_control.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp_p1_prior_control.png"

RANDOM_SEED = 42
FEATURE_CUTOFF_YEAR = 2021
LABEL_START_YEAR = 2022
LABEL_END_YEAR = 2025
N_SPLITS = 5
PERMUTATION_COUNT = 1000
CLASSIC_MARKERS = {"APP", "APOE", "MAPT", "GFAP", "NEFL"}
SECRETED_GENES = {
    "NEFL",
    "GFAP",
    "APP",
    "MAPT",
    "APOE",
    "CLU",
    "A2M",
    "ACE",
    "BDNF",
    "CRP",
    "IL6",
    "TNF",
    "VEGFA",
    "FGF21",
    "GDF15",
    "LGALS3",
    "S100B",
    "CHI3L1",
    "LGALS3BP",
    "CD14",
    "CXCL13",
    "NPTX2",
    "VGF",
    "SNAP25",
    "NRGN",
    "SYPL1",
    "GAP43",
    "SNCB",
    "YWHAG",
    "YWHAZ",
    "YWHAE",
    "NEFM",
    "NEFH",
    "SMOC1",
    "DYRK1A",
    "MOB1A",
    "PTPRN2",
    "BASP1",
    "MMP9",
    "MMP10",
    "SPP1",
    "TREM2",
    "CST3",
    "BIN1",
    "PICALM",
    "CR1",
    "FCGR3B",
    "CD33",
    "AKT1",
    "INSR",
}

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
TEMPORAL_FEATURES = [
    "sig_count_all",
    "sig_count_peripheral",
    "sig_max_auc",
    "sig_count_high_auc",
]
PRIOR_FEATURES = [
    "is_secreted",
    "snp_count",
    "cpg_count",
    "deg_count",
    "hub_count",
    "mirna_count",
    "layer_count",
]
FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]


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
    if 1900 <= year <= LABEL_END_YEAR:
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


def build_first_peripheral_year(signature_df: pd.DataFrame) -> dict[str, int]:
    first_year: dict[str, int] = {}
    peripheral_mask = signature_df["Tissue"].apply(is_peripheral_tissue)
    high_auc_mask = signature_df["AUC_num"].ge(0.88)
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
) -> pd.DataFrame:
    matured_by_label_end = {gene for gene, year in first_peripheral_year.items() if year <= LABEL_END_YEAR}
    positive_genes = {
        gene
        for gene, year in first_peripheral_year.items()
        if LABEL_START_YEAR <= year <= LABEL_END_YEAR
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


def evaluate_feature_set(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    splitter_seed: int = RANDOM_SEED,
) -> dict:
    subset_df = eval_df.loc[eval_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    if subset_df.empty:
        return {
            "features": feature_columns,
            "n_eval_genes": 0,
            "n_positives": 0,
            "n_negatives": 0,
            "mean_auroc": float("nan"),
            "std_auroc": float("nan"),
            "fold_aurocs": [],
            "valid_folds": 0,
            "skipped_folds": N_SPLITS,
            "split_error": None,
        }

    subset_df["label"] = subset_df["label"].astype(int)
    X = subset_df[feature_columns].to_numpy(dtype=float)
    y = subset_df["label"].to_numpy(dtype=int)

    n_positives = int(y.sum())
    n_negatives = int(len(y) - n_positives)
    if len(np.unique(y)) < 2:
        return {
            "features": feature_columns,
            "n_eval_genes": int(len(subset_df)),
            "n_positives": n_positives,
            "n_negatives": n_negatives,
            "mean_auroc": float("nan"),
            "std_auroc": float("nan"),
            "fold_aurocs": [],
            "valid_folds": 0,
            "skipped_folds": N_SPLITS,
            "split_error": "only_one_class_in_dataset",
        }

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=splitter_seed)
    fold_aurocs: list[float] = []
    skipped_folds = 0
    split_error = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            split_indices = list(splitter.split(X, y))
    except ValueError as exc:
        split_error = str(exc)
        split_indices = []

    for train_idx, test_idx in split_indices:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped_folds += 1
            continue
        model = make_xgb()
        model.fit(X[train_idx], y_train)
        test_probs = model.predict_proba(X[test_idx])[:, 1]
        fold_aurocs.append(float(roc_auc_score(y_test, test_probs)))

    if split_error is not None and not fold_aurocs:
        skipped_folds = N_SPLITS
    else:
        skipped_folds = max(skipped_folds, N_SPLITS - len(fold_aurocs))

    mean_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else float("nan")
    if len(fold_aurocs) > 1:
        std_auroc = float(np.std(fold_aurocs, ddof=1))
    elif len(fold_aurocs) == 1:
        std_auroc = 0.0
    else:
        std_auroc = float("nan")

    return {
        "features": feature_columns,
        "n_eval_genes": int(len(subset_df)),
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "fold_aurocs": fold_aurocs,
        "valid_folds": int(len(fold_aurocs)),
        "skipped_folds": int(skipped_folds),
        "split_error": split_error,
    }


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_permutations: int,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    null_aurocs = np.full(shape=n_permutations, fill_value=np.nan, dtype=float)

    for perm_idx in range(n_permutations):
        y_perm = rng.permutation(y)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED + perm_idx + 1,
        )
        fold_aurocs = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                split_indices = list(splitter.split(X, y_perm))
        except ValueError:
            split_indices = []
        for train_idx, test_idx in split_indices:
            y_train = y_perm[train_idx]
            y_test = y_perm[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            model = make_xgb()
            model.fit(X[train_idx], y_train)
            perm_probs = model.predict_proba(X[test_idx])[:, 1]
            fold_aurocs.append(float(roc_auc_score(y_test, perm_probs)))
        if fold_aurocs:
            null_aurocs[perm_idx] = float(np.mean(fold_aurocs))

        if (perm_idx + 1) % 100 == 0:
            print(f"Permutation {perm_idx + 1}/{n_permutations} complete")

    return null_aurocs


def summarize_null_distribution(null_aurocs: np.ndarray) -> dict:
    finite_nulls = null_aurocs[np.isfinite(null_aurocs)]
    if len(finite_nulls) == 0:
        return {
            "count": int(len(null_aurocs)),
            "finite_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(null_aurocs)),
        "finite_count": int(len(finite_nulls)),
        "mean": float(np.mean(finite_nulls)),
        "std": float(np.std(finite_nulls, ddof=1)) if len(finite_nulls) > 1 else 0.0,
        "median": float(np.median(finite_nulls)),
        "min": float(np.min(finite_nulls)),
        "max": float(np.max(finite_nulls)),
        "p05": float(np.quantile(finite_nulls, 0.05)),
        "p95": float(np.quantile(finite_nulls, 0.95)),
    }


def plot_results(
    experiment_rows: list[dict],
    null_mean: float,
    permutation_p_value: float,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    labels = [row["label"] for row in experiment_rows]
    values = [row["mean_auroc"] for row in experiment_rows]
    errors = [row["std_auroc"] if math.isfinite(row["std_auroc"]) else 0.0 for row in experiment_rows]
    colors = ["#2A6F97", "#6C9A8B", "#E07A5F", "#3D405B", "#81B29A"]
    plot_values = [0.0 if not math.isfinite(value) else value for value in values]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(
        labels,
        plot_values,
        yerr=errors,
        capsize=5,
        color=colors[: len(labels)],
        edgecolor="#1F2933",
        linewidth=1.0,
    )

    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            bar.set_hatch("//")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "NaN",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    if math.isfinite(null_mean):
        ax.axhline(
            null_mean,
            color="#C1121F",
            linestyle="--",
            linewidth=2,
            label=f"Permutation null mean = {null_mean:.3f}",
        )

    ax.set_title("Prior-Control Experiments with 2021 Feature Cutoff")
    ax.set_ylabel("Mean 5-fold AUROC")
    ax.set_ylim(0.0, max([0.6] + [value + error + 0.08 for value, error in zip(plot_values, errors)]))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.text(
        0.99,
        0.02,
        f"Permutation p = {permutation_p_value:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D7DE", "alpha": 0.95},
    )
    if math.isfinite(null_mean):
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    first_peripheral_year = build_first_peripheral_year(tables["signature"])

    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)
    label_df = assign_labels(gene_universe, evidence_by_cutoff, first_peripheral_year)

    full_df = feature_df.merge(label_df, on="gene", how="left")
    eval_df = full_df.loc[full_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["label"].astype(int)

    secreted_eval_df = eval_df.loc[eval_df["is_secreted"] == 1].copy().reset_index(drop=True)
    secreted_temporal = evaluate_feature_set(secreted_eval_df, TEMPORAL_FEATURES, splitter_seed=RANDOM_SEED)
    secreted_prior = evaluate_feature_set(secreted_eval_df, PRIOR_FEATURES, splitter_seed=RANDOM_SEED)
    secreted_full = evaluate_feature_set(secreted_eval_df, FULL_FEATURES, splitter_seed=RANDOM_SEED)

    no_classic_df = full_df.copy()
    no_classic_mask = no_classic_df["gene"].isin(CLASSIC_MARKERS) & no_classic_df["label"].eq(1)
    no_classic_df.loc[no_classic_mask, "label"] = np.nan
    no_classic_eval_df = no_classic_df.loc[no_classic_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    no_classic_eval_df["label"] = no_classic_eval_df["label"].astype(int)
    no_classic_full = evaluate_feature_set(no_classic_eval_df, FULL_FEATURES, splitter_seed=RANDOM_SEED)

    full_model = evaluate_feature_set(eval_df, FULL_FEATURES, splitter_seed=RANDOM_SEED)

    print("Running 1000-label permutation test on the full evaluation set.")
    null_aurocs = permutation_test(
        X=eval_df[FULL_FEATURES].to_numpy(dtype=float),
        y=eval_df["label"].to_numpy(dtype=int),
        n_splits=N_SPLITS,
        n_permutations=PERMUTATION_COUNT,
    )
    permutation_p_value = float((1 + np.sum(null_aurocs >= full_model["mean_auroc"])) / (PERMUTATION_COUNT + 1))
    null_stats = summarize_null_distribution(null_aurocs)

    experiment_rows = [
        {
            "label": "Secreted temporal",
            "mean_auroc": secreted_temporal["mean_auroc"],
            "std_auroc": secreted_temporal["std_auroc"],
        },
        {
            "label": "Secreted prior",
            "mean_auroc": secreted_prior["mean_auroc"],
            "std_auroc": secreted_prior["std_auroc"],
        },
        {
            "label": "Secreted full",
            "mean_auroc": secreted_full["mean_auroc"],
            "std_auroc": secreted_full["std_auroc"],
        },
        {
            "label": "Full model all genes",
            "mean_auroc": full_model["mean_auroc"],
            "std_auroc": full_model["std_auroc"],
        },
        {
            "label": "Full model no classic",
            "mean_auroc": no_classic_full["mean_auroc"],
            "std_auroc": no_classic_full["std_auroc"],
        },
    ]
    plot_results(experiment_rows, null_stats["mean"], permutation_p_value, OUTPUT_PLOT)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "label_start_year": LABEL_START_YEAR,
        "label_end_year": LABEL_END_YEAR,
        "n_genes_universe": int(len(gene_universe)),
        "n_eval_genes": int(len(eval_df)),
        "n_positives": int(eval_df["label"].sum()),
        "n_negatives": int(len(eval_df) - eval_df["label"].sum()),
        "n_historical_positives_excluded": int((full_df["label_status"] == "historical_positive_excluded").sum()),
        "n_unlabeled_genes": int((full_df["label_status"] == "unlabeled").sum()),
        "secreted_temporal_auroc": secreted_temporal["mean_auroc"],
        "secreted_prior_auroc": secreted_prior["mean_auroc"],
        "secreted_full_auroc": secreted_full["mean_auroc"],
        "no_classic_full_auroc": no_classic_full["mean_auroc"],
        "full_model_auroc": full_model["mean_auroc"],
        "permutation_p_value": permutation_p_value,
        "null_distribution_stats": null_stats,
        "secreted_subset": {
            "n_eval_genes": int(len(secreted_eval_df)),
            "n_positives": int(secreted_eval_df["label"].sum()) if len(secreted_eval_df) else 0,
            "n_negatives": int(len(secreted_eval_df) - secreted_eval_df["label"].sum()) if len(secreted_eval_df) else 0,
            "temporal": secreted_temporal,
            "prior": secreted_prior,
            "full": secreted_full,
        },
        "drop_classic_markers": {
            "classic_markers_removed_from_positive_eval": sorted(CLASSIC_MARKERS),
            "n_eval_genes": int(len(no_classic_eval_df)),
            "n_positives": int(no_classic_eval_df["label"].sum()),
            "n_negatives": int(len(no_classic_eval_df) - no_classic_eval_df["label"].sum()),
            "full": no_classic_full,
        },
        "full_model": full_model,
        "permutation_test": {
            "feature_columns": FULL_FEATURES,
            "permutation_count": PERMUTATION_COUNT,
            "observed_auroc": full_model["mean_auroc"],
            "empirical_p_value": permutation_p_value,
            "null_distribution_stats": null_stats,
            "null_aurocs": null_aurocs.tolist(),
        },
        "feature_groups": {
            "temporal_features": TEMPORAL_FEATURES,
            "prior_features": PRIOR_FEATURES,
            "full_features": FULL_FEATURES,
        },
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
        "signature_pmid_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print()
    print("NM Prior-Control Experiment complete")
    print(f"Feature cutoff year: {FEATURE_CUTOFF_YEAR}")
    print(f"Label window: {LABEL_START_YEAR}-{LABEL_END_YEAR}")
    print(f"Gene universe: {len(gene_universe)}")
    print(f"Eval set: {len(eval_df)} genes ({int(eval_df['label'].sum())} positives, {int(len(eval_df) - eval_df['label'].sum())} negatives)")
    print(
        "Secreted subset: "
        f"{len(secreted_eval_df)} genes "
        f"({int(secreted_eval_df['label'].sum()) if len(secreted_eval_df) else 0} positives, "
        f"{int(len(secreted_eval_df) - secreted_eval_df['label'].sum()) if len(secreted_eval_df) else 0} negatives)"
    )
    print(f"secreted_temporal_auroc: {secreted_temporal['mean_auroc']:.6f}")
    print(f"secreted_prior_auroc: {secreted_prior['mean_auroc']:.6f}")
    print(f"secreted_full_auroc: {secreted_full['mean_auroc']:.6f}")
    print(f"full_model_auroc: {full_model['mean_auroc']:.6f}")
    print(f"no_classic_full_auroc: {no_classic_full['mean_auroc']:.6f}")
    print(f"permutation_p_value: {permutation_p_value:.6f}")
    print(
        "null_distribution_stats: "
        f"mean={null_stats['mean']:.6f}, std={null_stats['std']:.6f}, "
        f"median={null_stats['median']:.6f}, p05={null_stats['p05']:.6f}, p95={null_stats['p95']:.6f}"
    )
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
