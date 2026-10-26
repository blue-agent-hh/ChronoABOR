#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl
from pub_year_external import external_year_lookup

_mpl.rcParams["svg.fonttype"] = "none"
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp4_full_universe.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp4_plot.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp4_plot.pdf"
NM_EXP1_JSON = RESULTS_DIR / "nm_exp1_clinicaltrials.json"

RANDOM_SEED = 42
FEATURE_CUTOFF_YEAR = 2019
LABEL_START_YEAR = 2020
LABEL_END_YEAR = 2025
N_SPLITS = 5
PERMUTATION_COUNT = 1000
TOP_K_VALUES = [10, 25, 50, 100]
KNOWN_POSITIVES = ["NEFL", "GFAP", "APOE", "APP", "MAPT"]

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


def cross_validated_predictions(
    feature_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X_eval = eval_df[feature_columns].to_numpy(dtype=float)
    y_eval = eval_df["label"].to_numpy(dtype=int)
    X_all = feature_df[feature_columns].to_numpy(dtype=float)

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.full(shape=len(eval_df), fill_value=np.nan, dtype=float)
    all_fold_probs = np.zeros(shape=(len(feature_df), N_SPLITS), dtype=float)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_eval, y_eval)):
        model = make_xgb()
        model.fit(X_eval[train_idx], y_eval[train_idx])
        test_probs = model.predict_proba(X_eval[test_idx])[:, 1]
        oof_probs[test_idx] = test_probs
        all_fold_probs[:, fold_idx] = model.predict_proba(X_all)[:, 1]
        fold_aurocs.append(float(roc_auc_score(y_eval[test_idx], test_probs)))

    if np.isnan(oof_probs).any():
        raise RuntimeError("Cross-validation failed to populate every held-out prediction.")

    return oof_probs, all_fold_probs.mean(axis=1), fold_aurocs


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_permutations: int,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    null_aurocs = np.zeros(shape=n_permutations, dtype=float)

    for perm_idx in range(n_permutations):
        y_perm = rng.permutation(y)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED + perm_idx + 1,
        )
        fold_aurocs = []
        for train_idx, test_idx in splitter.split(X, y_perm):
            model = make_xgb()
            model.fit(X[train_idx], y_perm[train_idx])
            perm_probs = model.predict_proba(X[test_idx])[:, 1]
            fold_aurocs.append(float(roc_auc_score(y_perm[test_idx], perm_probs)))
        null_aurocs[perm_idx] = float(np.mean(fold_aurocs))

        if (perm_idx + 1) % 100 == 0:
            print(f"Permutation {perm_idx + 1}/{n_permutations} complete")

    return null_aurocs


def compute_topk_metrics(ranked_df: pd.DataFrame, ks: list[int], base_rate: float) -> list[dict]:
    positive_total = int(ranked_df["label"].sum())
    rows = []
    for k in ks:
        if k > len(ranked_df):
            continue
        subset = ranked_df.head(k)
        tp = int(subset["label"].sum())
        precision = float(tp / k)
        recall = float(tp / positive_total) if positive_total > 0 else float("nan")
        lift = float(precision / base_rate) if base_rate > 0 else float("nan")
        nnt = float(k / tp) if tp > 0 else float("nan")
        rows.append(
            {
                "k": int(k),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": lift,
                "nnt_at_k": nnt,
                "n_true_positives_in_top_k": tp,
            }
        )
    return rows


def merge_topk_rows(model_rows: list[dict], baseline_rows: list[dict], random_precision: float) -> list[dict]:
    baseline_lookup = {row["k"]: row for row in baseline_rows}
    merged = []
    for row in model_rows:
        baseline = baseline_lookup[row["k"]]
        merged.append(
            {
                "k": int(row["k"]),
                "chronoabor_precision_at_k": float(row["precision_at_k"]),
                "chronoabor_recall_at_k": float(row["recall_at_k"]),
                "chronoabor_lift_at_k": float(row["lift_at_k"]),
                "chronoabor_nnt_at_k": float(row["nnt_at_k"]),
                "chronoabor_true_positives_in_top_k": int(row["n_true_positives_in_top_k"]),
                "baseline_deg_count_precision_at_k": float(baseline["precision_at_k"]),
                "baseline_deg_count_recall_at_k": float(baseline["recall_at_k"]),
                "baseline_deg_count_lift_at_k": float(baseline["lift_at_k"]),
                "baseline_deg_count_nnt_at_k": float(baseline["nnt_at_k"]),
                "baseline_deg_count_true_positives_in_top_k": int(baseline["n_true_positives_in_top_k"]),
                "random_expected_precision_at_k": float(random_precision),
            }
        )
    return merged


def load_nm_exp1_trial_lookup(path: Path) -> tuple[dict[str, int], dict | None]:
    if not path.exists():
        return {}, None
    payload = json.loads(path.read_text())
    lookup = {}
    for group_name in ["top50", "null50"]:
        for record in payload.get(group_name, []):
            gene = str(record.get("gene", "")).upper().strip()
            if not gene:
                continue
            lookup[gene] = int(record.get("trial_count", 0))
    return lookup, payload.get("summary")


def build_top50_table(
    ranked_candidate_df: pd.DataFrame,
    full_rank_lookup: dict[str, int],
    trial_lookup: dict[str, int],
) -> list[dict]:
    top50 = ranked_candidate_df.head(50).copy()
    rows = []
    for _, row in top50.iterrows():
        gene = row["gene"]
        nm_exp1_trial_count = trial_lookup.get(gene)
        rows.append(
            {
                "rank": int(row["candidate_rank"]),
                "full_universe_rank": int(full_rank_lookup[gene]),
                "gene": gene,
                "cv_mean_probability": float(row["cv_mean_probability"]),
                "label_status": str(row["label_status"]),
                "first_peripheral_high_auc_year": row["first_peripheral_high_auc_year"],
                "has_evidence_by_cutoff": bool(row["has_evidence_by_cutoff"]),
                "nm_exp1_trial_count": None if nm_exp1_trial_count is None else int(nm_exp1_trial_count),
                "nm_exp1_trial_status": None
                if nm_exp1_trial_count is None
                else ("trial-linked" if nm_exp1_trial_count > 0 else "no-trials-found"),
                **{column: float(row[column]) if column == "sig_max_auc" else int(row[column]) for column in FEATURE_COLUMNS},
            }
        )
    return rows


def build_known_positive_rank_table(
    ranked_full_df: pd.DataFrame,
    ranked_candidate_df: pd.DataFrame,
    trial_lookup: dict[str, int],
) -> dict[str, dict]:
    full_rank_lookup = {gene: idx + 1 for idx, gene in enumerate(ranked_full_df["gene"].tolist())}
    candidate_rank_lookup = {gene: idx + 1 for idx, gene in enumerate(ranked_candidate_df["gene"].tolist())}
    info_lookup = ranked_full_df.set_index("gene").to_dict(orient="index")
    result = {}
    for gene in KNOWN_POSITIVES:
        row = info_lookup.get(gene)
        result[gene] = {
            "full_universe_rank": full_rank_lookup.get(gene),
            "candidate_rank_by_2019_evidence": candidate_rank_lookup.get(gene),
            "in_candidate_pool": None if row is None else bool(row["has_evidence_by_cutoff"]),
            "cv_mean_probability": None if row is None else float(row["cv_mean_probability"]),
            "label_status": None if row is None else str(row["label_status"]),
            "first_peripheral_high_auc_year": None if row is None else row["first_peripheral_high_auc_year"],
            "nm_exp1_trial_count": trial_lookup.get(gene),
        }
    return result


def plot_results(
    null_aurocs: np.ndarray,
    actual_auroc: float,
    permutation_p_value: float,
    topk_table: list[dict],
    base_rate: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].hist(null_aurocs, bins=30, color="#56B4E9", edgecolor="white")
    axes[0].axvline(actual_auroc, color="#D55E00", linewidth=2, label=f"Actual AUROC = {actual_auroc:.3f}")
    axes[0].set_title("Permutation Null AUROC")
    axes[0].set_xlabel("Mean fold AUROC")
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.03,
        0.97,
        f"Permutation p = {permutation_p_value:.4f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#b0bec5"},
    )
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(alpha=0.2)

    ks = [row["k"] for row in topk_table]
    chrono_precision = [row["chronoabor_precision_at_k"] for row in topk_table]
    baseline_precision = [row["baseline_deg_count_precision_at_k"] for row in topk_table]
    axes[1].plot(ks, chrono_precision, marker="o", linewidth=2, color="#56B4E9", label="ChronoABOR full-universe")
    axes[1].plot(ks, baseline_precision, marker="s", linewidth=2, color="#E69F00", label="Most-published DEG baseline")
    axes[1].axhline(base_rate, linestyle="--", color="#616161", linewidth=1.5, label="Random expectation")
    axes[1].set_title("Precision@k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Precision")
    axes[1].set_xticks(ks)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
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
    label_df = assign_labels(gene_universe, evidence_by_cutoff, first_peripheral_year)

    full_df = feature_df.merge(label_df, on="gene", how="left")
    eval_df = full_df.loc[full_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["label"].astype(int)

    n_positives = int(eval_df["label"].sum())
    n_negatives = int((1 - eval_df["label"]).sum())
    if n_positives < 10:
        print(f"WARNING: only {n_positives} positives remain after filtering; continuing anyway.")

    oof_probs, full_mean_probs, fold_aurocs = cross_validated_predictions(
        feature_df=full_df,
        eval_df=eval_df,
        feature_columns=FEATURE_COLUMNS,
    )

    eval_df["oof_probability"] = oof_probs
    full_df["cv_mean_probability"] = full_mean_probs

    cv_auroc_mean = float(np.mean(fold_aurocs))
    cv_auroc_std = float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
    cv_auroc_oof = float(roc_auc_score(eval_df["label"], eval_df["oof_probability"]))
    cv_auprc = float(average_precision_score(eval_df["label"], eval_df["oof_probability"]))
    brier = float(brier_score_loss(eval_df["label"], eval_df["oof_probability"]))

    base_rate = float(eval_df["label"].mean())
    ranked_eval_df = eval_df.sort_values(
        by=["oof_probability", "num_deg_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    baseline_eval_df = eval_df.sort_values(
        by=["num_deg_studies", "gene"],
        ascending=[False, True],
    ).reset_index(drop=True)

    chrono_topk = compute_topk_metrics(ranked_eval_df, TOP_K_VALUES, base_rate)
    baseline_topk = compute_topk_metrics(baseline_eval_df, TOP_K_VALUES, base_rate)
    topk_table = merge_topk_rows(chrono_topk, baseline_topk, base_rate)

    baseline_deg_count_auroc = float(roc_auc_score(eval_df["label"], eval_df["num_deg_studies"]))
    baseline_deg_count_auprc = float(average_precision_score(eval_df["label"], eval_df["num_deg_studies"]))

    print("Running 1000-label permutation test. This is the slowest step.")
    null_aurocs = permutation_test(
        X=eval_df[FEATURE_COLUMNS].to_numpy(dtype=float),
        y=eval_df["label"].to_numpy(dtype=int),
        n_splits=N_SPLITS,
        n_permutations=PERMUTATION_COUNT,
    )
    permutation_p_value = float((1 + np.sum(null_aurocs >= cv_auroc_mean)) / (len(null_aurocs) + 1))

    full_df = full_df.sort_values(
        by=["cv_mean_probability", "num_deg_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    full_df["rank"] = np.arange(1, len(full_df) + 1)
    full_rank_lookup = {gene: idx + 1 for idx, gene in enumerate(full_df["gene"].tolist())}

    candidate_df = full_df.loc[full_df["has_evidence_by_cutoff"]].copy().reset_index(drop=True)
    candidate_df["candidate_rank"] = np.arange(1, len(candidate_df) + 1)

    trial_lookup, nm_exp1_summary = load_nm_exp1_trial_lookup(NM_EXP1_JSON)
    top50_table = build_top50_table(candidate_df, full_rank_lookup, trial_lookup)
    top50_genes = [row["gene"] for row in top50_table]
    known_positives_rank = build_known_positive_rank_table(full_df, candidate_df, trial_lookup)

    lookup_annotated = [row for row in top50_table if row["nm_exp1_trial_count"] is not None]
    lookup_trial_hits = [row for row in top50_table if row["nm_exp1_trial_count"] is not None and row["nm_exp1_trial_count"] > 0]
    candidate_lookup_counts = [trial_lookup[gene] for gene in candidate_df["gene"] if gene in trial_lookup]
    candidate_lookup_hit_rate = (
        float(np.mean([count > 0 for count in candidate_lookup_counts])) if candidate_lookup_counts else None
    )
    top50_lookup_counts = [row["nm_exp1_trial_count"] for row in lookup_annotated]
    top50_lookup_hit_rate = float(np.mean([count > 0 for count in top50_lookup_counts])) if top50_lookup_counts else None
    lookup_hit_rate_enrichment = (
        float(top50_lookup_hit_rate / candidate_lookup_hit_rate)
        if top50_lookup_hit_rate is not None and candidate_lookup_hit_rate not in (None, 0.0)
        else None
    )
    candidate_lookup_mean_trials = float(np.mean(candidate_lookup_counts)) if candidate_lookup_counts else None
    top50_lookup_mean_trials = float(np.mean(top50_lookup_counts)) if top50_lookup_counts else None

    plot_results(
        null_aurocs=null_aurocs,
        actual_auroc=cv_auroc_mean,
        permutation_p_value=permutation_p_value,
        topk_table=topk_table,
        base_rate=base_rate,
    )

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "label_window": f"{LABEL_START_YEAR}-{LABEL_END_YEAR}",
        "n_genes_universe": int(len(gene_universe)),
        "n_eval_genes": int(len(eval_df)),
        "n_candidate_genes_by_cutoff": int(len(candidate_df)),
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "n_historical_positives_excluded": int((full_df["label_status"] == "historical_positive_excluded").sum()),
        "n_unlabeled_genes": int((full_df["label_status"] == "unlabeled").sum()),
        "cv_auroc_mean": cv_auroc_mean,
        "cv_auroc_std": cv_auroc_std,
        "cv_auroc_oof": cv_auroc_oof,
        "cv_auprc": cv_auprc,
        "brier_score": brier,
        "permutation_p_value": permutation_p_value,
        "baseline_deg_count_auroc": baseline_deg_count_auroc,
        "baseline_deg_count_auprc": baseline_deg_count_auprc,
        "topk_table": topk_table,
        "top50_ranking_population": "genes with any ABOR evidence by 2019",
        "top50_genes": top50_genes,
        "top50_table": top50_table,
        "known_positives_rank": known_positives_rank,
        "historical_positive_genes_excluded": sorted(
            full_df.loc[full_df["label_status"] == "historical_positive_excluded", "gene"].tolist()
        ),
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
        "signature_pmid_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "clinicaltrials_lookup_summary": {
            "source_file": str(NM_EXP1_JSON) if NM_EXP1_JSON.exists() else None,
            "nm_exp1_summary": nm_exp1_summary,
            "lookup_method": "reuse nm_exp1 trial counts only; no fresh API calls in this experiment",
            "n_candidate_genes_with_nm_exp1_annotation": int(len(candidate_lookup_counts)),
            "candidate_hit_rate_within_nm_exp1_lookup": candidate_lookup_hit_rate,
            "candidate_mean_trial_count_within_nm_exp1_lookup": candidate_lookup_mean_trials,
            "n_top50_genes_with_nm_exp1_annotation": int(len(lookup_annotated)),
            "n_top50_genes_with_trials_in_nm_exp1": int(len(lookup_trial_hits)),
            "top50_hit_rate_within_nm_exp1_lookup": top50_lookup_hit_rate,
            "top50_mean_trial_count_within_nm_exp1_lookup": top50_lookup_mean_trials,
            "top50_vs_candidate_hit_rate_enrichment_within_nm_exp1_lookup": lookup_hit_rate_enrichment,
            "annotated_trial_genes": [row["gene"] for row in lookup_trial_hits],
        },
        "feature_columns": FEATURE_COLUMNS,
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print()
    print("NM Experiment 4: Full-universe temporal evaluation complete")
    print(f"Gene universe: {len(gene_universe)}")
    print(f"Eval set: {len(eval_df)} genes ({n_positives} positives, {n_negatives} negatives)")
    print(f"Forecast-time candidate pool (evidence by 2019): {len(candidate_df)} genes")
    print(f"Historical positives excluded from negatives: {results['n_historical_positives_excluded']}")
    print(f"Feature cutoff year: {FEATURE_CUTOFF_YEAR}")
    print(f"Label window: {LABEL_START_YEAR}-{LABEL_END_YEAR}")
    print(f"CV AUROC (mean ± std): {cv_auroc_mean:.4f} ± {cv_auroc_std:.4f}")
    print(f"CV AUROC (OOF): {cv_auroc_oof:.4f}")
    print(f"CV AUPRC: {cv_auprc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Permutation p-value: {permutation_p_value:.6f}")
    print(f"Baseline DEG-count AUROC: {baseline_deg_count_auroc:.4f}")
    print("Known positive ranks:")
    for gene in KNOWN_POSITIVES:
        info = known_positives_rank[gene]
        print(
            f"  {gene}: full_rank={info['full_universe_rank']}, "
            f"candidate_rank={info['candidate_rank_by_2019_evidence']}, "
            f"prob={info['cv_mean_probability']}, label_status={info['label_status']}, "
            f"first_year={info['first_peripheral_high_auc_year']}"
        )
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
