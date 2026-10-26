#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
import inspect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

GEMINI_HTTP_PROXY = "http://172.31.22.23:15001"
os.environ["HTTPS_PROXY"] = GEMINI_HTTP_PROXY
os.environ["HTTP_PROXY"] = GEMINI_HTTP_PROXY
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

try:
    import google.generativeai as genai
except Exception as exc:
    genai = None
    GEMINI_IMPORT_ERROR = str(exc)
else:
    GEMINI_IMPORT_ERROR = None

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = str(exc)
else:
    LIGHTGBM_IMPORT_ERROR = None

try:
    from catboost import CatBoostClassifier
except Exception as exc:
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = str(exc)
else:
    CATBOOST_IMPORT_ERROR = None

try:
    from tabpfn import TabPFNClassifier
except Exception as exc:
    TabPFNClassifier = None
    TABPFN_IMPORT_ERROR = str(exc)
else:
    TABPFN_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp5_llm_vs_ml_benchmark.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp5_comparison_plot.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp5_comparison_plot.pdf"
LLM_CACHE_JSON = RESULTS_DIR / "nm_exp5_llm_cache.json"
LLM_CACHE_VERSION = "nm_exp5_v3_geneblind"

RANDOM_SEED = 42
FEATURE_CUTOFF_YEAR = 2021
LABEL_START_YEAR = 2022
LABEL_END_YEAR = 2025
INTERNAL_CV_REQUEST = "LOOCV"
INTERNAL_LOO_MAX_N = 80
INTERNAL_SKF_SPLITS = 5
LLM_SAVE_EVERY = 25
LLM_BATCH_SIZE = 500
PROXY_LLM_MAX_WORKERS = 8
GEMINI_LLM_MAX_WORKERS = 1
PROXY_LLM_SLEEP_SECONDS = 0.0
GEMINI_LLM_SLEEP_SECONDS = 0.5
LLM_GENE_BLIND_DEFAULT = True
ANONYMOUS_GENE_ID = "[ANONYMOUS_GENE_ID]"

PROXY_MODELS = ["gpt-5.2", "gpt-5.4"]
PROXY_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
PROXY_API_KEY = os.environ.get("OPENAI_API_KEY")

GEMINI_MODELS = ["gemini-2.5-pro", "gemini-3.1-pro-preview"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ALL_LLM_MODELS = PROXY_MODELS + GEMINI_MODELS
LLM_MODEL_PROVIDERS = {
    "gpt-5.2": "proxy",
    "gpt-5.4": "proxy",
    "gemini-2.5-pro": "gemini",
    "gemini-3.1-pro-preview": "gemini",
}
LLM_MODEL_DISPLAY_NAMES = {
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
    "gemini-2.5-pro": "Gemini-2.5-pro",
    "gemini-3.1-pro-preview": "Gemini-3.1-pro-preview",
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
    "is_secreted",
    "n_peripheral_high_auc",
    "has_plasma_sig",
]

EXTERNAL_BASELINE_FEATURES = [
    "is_secreted",
    "sig_count_all",
    "sig_count_peripheral",
    "sig_max_auc",
    "n_peripheral_high_auc",
    "layer_count",
    "num_studies",
]

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
    "IL8",
    "CXCL8",
    "MMP9",
    "MMP10",
    "SPP1",
    "TREM2",
    "CST3",
    "FCGR3B",
    "CD33",
    "BIN1",
    "PICALM",
    "CR1",
}

NM_CORPUS_GENES = [
    "NEFL",
    "GFAP",
    "APOE",
    "APP",
    "MAPT",
    "SNAP25",
    "NRGN",
    "SYPL1",
    "GAP43",
    "PTPRN2",
    "BASP1",
    "SNCB",
    "VGF",
    "NPTX2",
    "YWHAG",
    "YWHAZ",
    "YWHAE",
    "NEFM",
    "NEFH",
    "SMOC1",
    "DYRK1A",
    "MOB1A",
    "IL6",
    "TNF",
    "CRP",
    "FGF21",
    "GDF15",
    "CXCL13",
    "CHI3L1",
    "LGALS3BP",
    "CD14",
    "S100B",
]

LLM_SYSTEM_PROMPT = (
    "You are an expert Alzheimer's disease genomics researcher evaluating "
    "candidate blood biomarker genes. You will assess genes based on their "
    "multi-omics evidence profile."
)

LLM_USER_PROMPT_TEMPLATE = (
    "Based on the following multi-omics evidence profile, estimate the "
    "probability (0.0 to 1.0) that this gene will emerge as a validated "
    "Alzheimer's disease blood biomarker within the next 3 years.\n\n"
    "{evidence_card}\n\n"
    "Respond with ONLY a single floating-point number between 0.0 and 1.0. "
    "Do not include any explanation."
)


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


def make_xgb(y_train: np.ndarray | None = None) -> XGBClassifier:
    scale_pos_weight = None
    if y_train is not None:
        positives = int(np.sum(y_train))
        negatives = int(len(y_train) - positives)
        if positives > 0 and negatives > 0:
            scale_pos_weight = float(negatives / positives)
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )


def make_lgbm() -> LGBMClassifier:
    if LGBMClassifier is None:
        raise RuntimeError(LIGHTGBM_IMPORT_ERROR or "lightgbm is not installed")
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        objective="binary",
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=-1,
    )


def make_catboost() -> CatBoostClassifier:
    if CatBoostClassifier is None:
        raise RuntimeError(CATBOOST_IMPORT_ERROR or "catboost is not installed")
    return CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=0,
        allow_writing_files=False,
        thread_count=1,
    )


def make_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def make_tabpfn() -> TabPFNClassifier:
    if TabPFNClassifier is None:
        raise RuntimeError(TABPFN_IMPORT_ERROR or "tabpfn is not installed")
    return TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=True,
        random_state=RANDOM_SEED,
        n_preprocessing_jobs=1,
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
    pmid_conflicts = {
        pmid: len(set(years))
        for pmid, years in pmid_to_years.items()
        if len(set(years)) > 1
    }
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
            "n_peripheral_high_auc": 0,
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
            if peripheral and pd.notna(auc) and float(auc) >= 0.88:
                record["n_peripheral_high_auc"] += 1
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
                "is_secreted": int(gene in SECRETED_GENES),
                "n_peripheral_high_auc": int(record["n_peripheral_high_auc"]),
                "has_plasma_sig": int(record["sig_count_peripheral"] > 0),
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


def choose_internal_splitter(n_rows: int):
    if n_rows <= INTERNAL_LOO_MAX_N:
        return LeaveOneOut(), INTERNAL_CV_REQUEST
    return (
        StratifiedKFold(
            n_splits=INTERNAL_SKF_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
        f"StratifiedKFold({INTERNAL_SKF_SPLITS})_fallback_large_n",
    )


def predict_positive_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(float)
        return probs.reshape(-1).astype(float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise RuntimeError(f"{type(model).__name__} does not expose predict_proba or decision_function")


def instantiate_model(model_factory, y_train: np.ndarray):
    try:
        signature = inspect.signature(model_factory)
    except (TypeError, ValueError):
        return model_factory()
    if len(signature.parameters) == 0:
        return model_factory()
    return model_factory(y_train)


def cross_validated_predictions(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, str]:
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both classes for internal cross-validation")

    splitter, cv_name = choose_internal_splitter(len(y))
    oof_probs = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    split_iter = splitter.split(X, y) if isinstance(splitter, StratifiedKFold) else splitter.split(X)
    for train_idx, test_idx in split_iter:
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError("A training fold contains only one class")
        model = instantiate_model(model_factory, y_train)
        model.fit(X[train_idx], y_train)
        oof_probs[test_idx] = predict_positive_probability(model, X[test_idx])

    if np.isnan(oof_probs).any():
        raise RuntimeError("Cross-validation failed to populate all held-out predictions")
    return oof_probs, cv_name


def cross_validated_predictions_with_splitter(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    cv_name: str,
) -> tuple[np.ndarray, str]:
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both classes for cross-validation")

    oof_probs = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    split_iter = splitter.split(X, y) if isinstance(splitter, StratifiedKFold) else splitter.split(X)
    for train_idx, test_idx in split_iter:
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError("A training fold contains only one class")
        model = instantiate_model(model_factory, y_train)
        model.fit(X[train_idx], y_train)
        oof_probs[test_idx] = predict_positive_probability(model, X[test_idx])

    if np.isnan(oof_probs).any():
        raise RuntimeError("Cross-validation failed to populate all held-out predictions")
    return oof_probs, cv_name


def fit_and_score_full(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_score: np.ndarray,
) -> np.ndarray:
    model = instantiate_model(model_factory, y_train)
    model.fit(X_train, y_train)
    return predict_positive_probability(model, X_score)


def clip_probability(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def load_llm_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return {}
    if raw.get("cache_version") != LLM_CACHE_VERSION:
        return {}
    model_payload = raw.get("models", {})
    cache: dict[str, dict[str, float]] = {}
    for model_name, model_scores in model_payload.items():
        cache[model_name] = {gene: float(score) for gene, score in model_scores.items()}
    return cache


def save_llm_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    payload = {
        "cache_version": LLM_CACHE_VERSION,
        "models": cache,
    }
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def build_evidence_card(row: pd.Series, gene_blind: bool = True) -> str:
    gene_label = ANONYMOUS_GENE_ID if gene_blind else str(row["gene"])
    return (
        f"Gene: {gene_label}\n"
        f"ABOR Evidence (as of {FEATURE_CUTOFF_YEAR}):\n"
        f"- Secreted/blood-detectable protein: {'yes' if int(row['is_secreted']) == 1 else 'no'}\n"
        f"- SNP associations: {int(row['snp_count'])}\n"
        f"- CpG methylation sites: {int(row['cpg_count'])}\n"
        f"- Differentially expressed: {int(row['deg_count'])} studies "
        f"(up: {int(row['deg_up_count'])}, down: {int(row['deg_down_count'])}, "
        f"tissue breadth: {int(row['deg_tissue_breadth'])})\n"
        f"- Protein-protein hub interactions: {int(row['hub_count'])}\n"
        f"- miRNA regulatory targets: {int(row['mirna_count'])}\n"
        f"- Biomarker signature studies: {int(row['sig_count_all'])} "
        f"(peripheral blood/plasma: {int(row['sig_count_peripheral'])}, "
        f"max AUC: {float(row['sig_max_auc']):.3f}, "
        f"high-AUC studies: {int(row['sig_count_high_auc'])})\n"
        f"- Peripheral blood signatures (high AUC >=0.88): {int(row['n_peripheral_high_auc'])}\n"
        f"- Total evidence layers: {int(row['layer_count'])}, total studies: {int(row['num_studies'])}"
    )


def evidence_card_from_row(row: pd.Series, gene_blind: bool = LLM_GENE_BLIND_DEFAULT) -> str:
    return build_evidence_card(row, gene_blind=gene_blind)


def build_llm_prompt_text(evidence_card: str) -> str:
    return f"{LLM_SYSTEM_PROMPT}\n\n{LLM_USER_PROMPT_TEMPLATE.format(evidence_card=evidence_card)}"


def extract_chat_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def extract_gemini_content(response: object) -> str:
    try:
        text = getattr(response, "text", "")
    except Exception:
        text = ""
    return "" if text is None else str(text)


def parse_probability_text(content: str) -> float:
    stripped = content.strip()
    try:
        return clip_probability(float(stripped))
    except Exception:
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)", stripped)
        if match is None:
            raise ValueError(f"Could not parse probability from response: {content!r}") from None
        return clip_probability(float(match.group(0)))


def provider_sleep_seconds(provider_name: str) -> float:
    if provider_name == "proxy":
        return PROXY_LLM_SLEEP_SECONDS
    if provider_name == "gemini":
        return GEMINI_LLM_SLEEP_SECONDS
    return 0.0


def call_proxy_llm_probability(client: OpenAI, model_name: str, evidence_card: str) -> float:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=6,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": LLM_USER_PROMPT_TEMPLATE.format(evidence_card=evidence_card),
            },
        ],
    )
    content = extract_chat_content(response.choices[0].message.content).strip()
    try:
        return parse_probability_text(content)
    except Exception:
        return 0.5


def call_gemini_probability(model_name: str, evidence_card: str) -> float:
    if genai is None:
        raise RuntimeError(GEMINI_IMPORT_ERROR or "google-generativeai is not installed")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(build_llm_prompt_text(evidence_card))
    content = extract_gemini_content(response).strip()
    try:
        return parse_probability_text(content)
    except Exception:
        return 0.5


def call_llm_probability(
    provider_name: str,
    model_name: str,
    evidence_card: str,
    client: OpenAI | None = None,
) -> float:
    if provider_name == "proxy":
        if client is None:
            raise RuntimeError("Proxy client unavailable")
        return call_proxy_llm_probability(client, model_name, evidence_card)
    if provider_name == "gemini":
        return call_gemini_probability(model_name, evidence_card)
    raise ValueError(f"Unsupported provider: {provider_name}")


def score_gene_with_new_client(provider_name: str, model_name: str, evidence_card: str) -> float:
    sleep_seconds = provider_sleep_seconds(provider_name)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    if provider_name == "proxy":
        worker_client = OpenAI(base_url=PROXY_BASE_URL, api_key=PROXY_API_KEY)
        return call_proxy_llm_probability(worker_client, model_name, evidence_card)
    if provider_name == "gemini":
        return call_gemini_probability(model_name, evidence_card)
    raise ValueError(f"Unsupported provider: {provider_name}")


def compute_external_metrics(
    ranked_df: pd.DataFrame,
    score_column: str,
    positive_genes: set[str],
) -> dict:
    eval_df = ranked_df.copy()
    eval_df["label_ext"] = eval_df["gene"].isin(positive_genes).astype(int)
    eval_df = eval_df.sort_values(
        by=[score_column, "num_deg_studies", "num_studies", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    base_rate = float(eval_df["label_ext"].mean())
    metrics = {
        "auroc": float(roc_auc_score(eval_df["label_ext"], eval_df[score_column])),
        "p_at_10": None,
        "ef_at_10": None,
        "p_at_20": None,
        "ef_at_20": None,
        "top_10_genes": eval_df.head(10)["gene"].tolist(),
        "top_20_genes": eval_df.head(20)["gene"].tolist(),
        "n_external_positives": int(eval_df["label_ext"].sum()),
        "n_external_universe": int(len(eval_df)),
        "external_base_rate": base_rate,
    }

    for k, p_key, ef_key in [(10, "p_at_10", "ef_at_10"), (20, "p_at_20", "ef_at_20")]:
        if len(eval_df) < k:
            continue
        precision = float(eval_df.head(k)["label_ext"].mean())
        metrics[p_key] = precision
        metrics[ef_key] = float(precision / base_rate) if base_rate > 0 else None

    return metrics


def format_metric(value: float | None, width: int = 7, digits: int = 3) -> str:
    if value is None:
        return " " * (width - 2) + "NA"
    return f"{value:{width}.{digits}f}"


def resolve_llm_models() -> list[str]:
    requested = os.environ.get("NM_EXP5_LLM_MODELS", "").strip()
    if not requested:
        return list(ALL_LLM_MODELS)

    active_models: list[str] = []
    seen: set[str] = set()
    for raw_item in requested.split(","):
        model_name = raw_item.strip()
        if not model_name or model_name in seen:
            continue
        if model_name not in ALL_LLM_MODELS:
            raise ValueError(
                f"Unknown model in NM_EXP5_LLM_MODELS: {model_name}. "
                f"Expected one of {ALL_LLM_MODELS}."
            )
        active_models.append(model_name)
        seen.add(model_name)

    if not active_models:
        raise ValueError("NM_EXP5_LLM_MODELS did not contain any valid model names")
    return active_models


def env_flag_enabled(env_name: str) -> bool:
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def compute_external_corpus_diagnostics(
    ranking_df: pd.DataFrame,
    positive_genes: set[str],
) -> dict:
    eval_df = ranking_df.copy()
    eval_df["label_ext"] = eval_df["gene"].isin(positive_genes).astype(int)
    positive_df = eval_df.loc[eval_df["label_ext"] == 1].copy()

    status_counts = {
        str(status): int(count)
        for status, count in positive_df["label_status"].value_counts(dropna=False).items()
    }
    genes_by_status = {
        str(status): sorted(group["gene"].tolist())
        for status, group in positive_df.groupby("label_status", dropna=False)
    }

    baseline_feature_aurocs = {}
    if positive_df.shape[0] > 0 and positive_df.shape[0] < eval_df.shape[0]:
        for feature_name in EXTERNAL_BASELINE_FEATURES:
            series = pd.to_numeric(eval_df[feature_name], errors="coerce").fillna(0.0)
            if series.nunique(dropna=False) < 2:
                continue
            baseline_feature_aurocs[feature_name] = float(roc_auc_score(eval_df["label_ext"], series))

    return {
        "n_external_positives": int(positive_df.shape[0]),
        "positive_status_counts": status_counts,
        "positive_genes_by_status": genes_by_status,
        "n_external_positive_internal_negatives": int(status_counts.get("negative", 0)),
        "n_external_positive_historical_positives": int(status_counts.get("historical_positive_excluded", 0)),
        "n_external_positive_internal_positives": int(status_counts.get("positive", 0)),
        "is_directly_comparable_to_internal": (
            int(status_counts.get("negative", 0)) == 0
            and int(status_counts.get("historical_positive_excluded", 0)) == 0
        ),
        "baseline_feature_aurocs": baseline_feature_aurocs,
    }


def print_external_corpus_diagnostics(diagnostics: dict) -> None:
    counts = diagnostics.get("positive_status_counts", {})
    counts_text = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(
        "External NM corpus diagnostics:",
        counts_text or "no external positives",
    )
    if not diagnostics.get("is_directly_comparable_to_internal", True):
        print(
            "Warning: External NM corpus mixes internal negatives and/or historical positives; "
            "treat AUROC_nm as corpus retrieval, not as a directly comparable prospective metric."
        )

    baseline_aurocs = diagnostics.get("baseline_feature_aurocs", {})
    if baseline_aurocs:
        top_features = sorted(
            baseline_aurocs.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        print(
            "Simple external baselines:",
            ", ".join(f"{feature}={score:.3f}" for feature, score in top_features),
        )


def print_comparison_table(method_results: list[dict]) -> None:
    method_width = max(18, max(len(row["display_name"]) for row in method_results)) + 2
    print()
    print(
        f"{'Method':<{method_width}}"
        "AUROC_int  AUROC_nm  P@10   EF@10  P@20   EF@20"
    )
    print("-" * (method_width + 52))
    for row in method_results:
        print(
            f"{row['display_name']:<{method_width}}"
            f"{format_metric(row.get('internal_auroc'), width=10)}"
            f"{format_metric(row.get('external_auroc'), width=10)}"
            f"{format_metric(row.get('p_at_10'), width=8)}"
            f"{format_metric(row.get('ef_at_10'), width=8, digits=1)}"
            f"{format_metric(row.get('p_at_20'), width=8)}"
            f"{format_metric(row.get('ef_at_20'), width=8, digits=1)}"
        )


def build_plot(method_results: list[dict]) -> None:
    labels = [row["display_name"] for row in method_results]
    internal_aurocs = [0.0 if row.get("internal_auroc") is None else row["internal_auroc"] for row in method_results]
    aurocs = [0.0 if row.get("external_auroc") is None else row["external_auroc"] for row in method_results]
    colors = ["#56B4E9" if row["kind"] == "ml" else "#E69F00" for row in method_results]
    hatches = ["//" if row.get("status") != "ok" else "" for row in method_results]

    fig_width = max(7.2, len(labels) * 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx, (value, color, hatch) in enumerate(zip(internal_aurocs, colors, hatches, strict=True)):
        axes[0].bar(idx, value, color=color, alpha=0.9, hatch=hatch)
    axes[0].set_title("Internal ABOR AUROC")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    for idx, (value, color, hatch) in enumerate(zip(aurocs, colors, hatches, strict=True)):
        axes[1].bar(idx, value, color=color, alpha=0.9, hatch=hatch)
    axes[1].set_title("External NM Corpus AUROC")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_ml_method(
    display_name: str,
    model_factory,
    eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    external_positive_genes: set[str],
) -> dict:
    print(f"[ML] {display_name}: internal CV + external ranking")
    X_eval = eval_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_eval = eval_df["label"].to_numpy(dtype=int)
    external_eval_df = ranking_df.copy()
    external_eval_df["label_ext"] = external_eval_df["gene"].isin(external_positive_genes).astype(int)
    X_external = external_eval_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    result = {
        "method": display_name,
        "display_name": display_name,
        "kind": "ml",
        "status": "ok",
        "skip_reason": None,
    }
    try:
        oof_probs, cv_name = cross_validated_predictions(model_factory, X_eval, y_eval)
        internal_auroc = float(roc_auc_score(y_eval, oof_probs))
        external_scores = fit_and_score_full(
            model_factory=model_factory,
            X_train=X_eval,
            y_train=y_eval,
            X_score=X_external,
        )
        ranking_with_scores = external_eval_df.copy()
        ranking_with_scores["score"] = external_scores
        external = compute_external_metrics(ranking_with_scores, "score", external_positive_genes)
        result.update(
            {
                "internal_auroc": internal_auroc,
                "auroc_internal": internal_auroc,
                "internal_cv_method": cv_name,
                "external_cv_method": "fit_full_internal_labels",
                "external_auroc": external["auroc"],
                "auroc_external": external["auroc"],
                "p_at_10": external["p_at_10"],
                "ef_at_10": external["ef_at_10"],
                "p_at_20": external["p_at_20"],
                "ef_at_20": external["ef_at_20"],
                "external_top_10_genes": external["top_10_genes"],
                "external_top_20_genes": external["top_20_genes"],
                "score_summary": {
                    "min": float(np.min(external_scores)),
                    "max": float(np.max(external_scores)),
                    "mean": float(np.mean(external_scores)),
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "skipped",
                "skip_reason": str(exc),
                "internal_auroc": None,
                "auroc_internal": None,
                "internal_cv_method": None,
                "external_cv_method": None,
                "external_auroc": None,
                "auroc_external": None,
                "p_at_10": None,
                "ef_at_10": None,
                "p_at_20": None,
                "ef_at_20": None,
                "external_top_10_genes": [],
                "external_top_20_genes": [],
            }
        )
    return result


def evaluate_llm_method(
    display_name: str,
    model_name: str,
    provider_name: str,
    provider_runtime: dict,
    llm_score_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    external_positive_genes: set[str],
    cache: dict[str, dict[str, float]],
    gene_blind: bool = True,
) -> dict:
    result = {
        "method": display_name,
        "display_name": display_name,
        "kind": "llm",
        "status": "ok",
        "skip_reason": None,
        "provider": provider_name,
        "gene_blind": bool(gene_blind),
        "internal_cv_method": "zero_shot_gene_blind" if gene_blind else "zero_shot_gene_aware",
    }

    if provider_runtime.get("status") != "ok":
        result.update(
            {
                "status": "skipped",
                "skip_reason": provider_runtime.get("skip_reason", f"{provider_name} provider unavailable"),
                "internal_auroc": None,
                "auroc_internal": None,
                "external_auroc": None,
                "auroc_external": None,
                "p_at_10": None,
                "ef_at_10": None,
                "p_at_20": None,
                "ef_at_20": None,
                "external_top_10_genes": [],
                "external_top_20_genes": [],
            }
        )
        return result

    client = provider_runtime.get("client")
    available_model_ids = provider_runtime.get("available_model_ids")
    max_workers = int(provider_runtime.get("max_workers", 1))

    if provider_name == "proxy" and (available_model_ids is None or model_name not in available_model_ids):
        result.update(
            {
                "status": "skipped",
                "skip_reason": f"{model_name} not exposed by {PROXY_BASE_URL}",
                "internal_auroc": None,
                "auroc_internal": None,
                "external_auroc": None,
                "auroc_external": None,
                "p_at_10": None,
                "ef_at_10": None,
                "p_at_20": None,
                "ef_at_20": None,
                "external_top_10_genes": [],
                "external_top_20_genes": [],
            }
        )
        return result

    model_cache = cache.setdefault(model_name, {})
    llm_max_genes = os.environ.get("NM_EXP5_LLM_MAX_GENES")
    llm_max_genes_int = int(llm_max_genes) if llm_max_genes else None
    llm_work_df = llm_score_df.copy()
    if llm_max_genes_int is not None:
        llm_work_df = llm_work_df.head(llm_max_genes_int).copy()

    pending_rows = [row for _, row in llm_work_df.iterrows() if row["gene"] not in model_cache]
    total_needed = len(llm_work_df)
    print(
        f"[LLM] {display_name}: {total_needed} genes to cover for scored subset, "
        f"{len(pending_rows)} uncached"
    )

    writes_since_flush = 0
    try:
        for batch_start in range(0, len(pending_rows), LLM_BATCH_SIZE):
            batch_end = min(batch_start + LLM_BATCH_SIZE, len(pending_rows))
            batch_rows = pending_rows[batch_start:batch_end]
            batch_index = (batch_start // LLM_BATCH_SIZE) + 1
            batch_total = math.ceil(len(pending_rows) / LLM_BATCH_SIZE) if pending_rows else 0
            if batch_rows:
                print(
                    f"  {display_name}: batch {batch_index}/{batch_total}, "
                    f"genes {batch_start + 1}-{batch_end}"
                )
            if provider_name != "proxy" or max_workers <= 1 or len(batch_rows) <= 1:
                for idx, row in enumerate(batch_rows, start=batch_start + 1):
                    score = call_llm_probability(
                        provider_name=provider_name,
                        model_name=model_name,
                        evidence_card=evidence_card_from_row(row, gene_blind=gene_blind),
                        client=client,
                    )
                    model_cache[row["gene"]] = score
                    writes_since_flush += 1
                    if idx % 50 == 0 or idx == len(pending_rows):
                        print(f"  {display_name}: scored {idx}/{len(pending_rows)} new genes")
                    if writes_since_flush >= LLM_SAVE_EVERY:
                        save_llm_cache(LLM_CACHE_JSON, cache)
                        writes_since_flush = 0
                    sleep_seconds = provider_sleep_seconds(provider_name)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
            else:
                batch_payload = [
                    (
                        str(row["gene"]),
                        evidence_card_from_row(row, gene_blind=gene_blind),
                    )
                    for row in batch_rows
                ]
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_gene = {
                        executor.submit(score_gene_with_new_client, provider_name, model_name, evidence_card): gene
                        for gene, evidence_card in batch_payload
                    }
                    for completed_count, future in enumerate(as_completed(future_to_gene), start=batch_start + 1):
                        gene = future_to_gene[future]
                        score = future.result()
                        model_cache[gene] = score
                        writes_since_flush += 1
                        if completed_count % 50 == 0 or completed_count == len(pending_rows):
                            print(f"  {display_name}: scored {completed_count}/{len(pending_rows)} new genes")
                        if writes_since_flush >= LLM_SAVE_EVERY:
                            save_llm_cache(LLM_CACHE_JSON, cache)
                            writes_since_flush = 0
            if batch_rows:
                save_llm_cache(LLM_CACHE_JSON, cache)
        save_llm_cache(LLM_CACHE_JSON, cache)
    except Exception as exc:
        save_llm_cache(LLM_CACHE_JSON, cache)
        result.update(
            {
                "status": "skipped",
                "skip_reason": str(exc),
                "internal_auroc": None,
                "auroc_internal": None,
                "external_auroc": None,
                "auroc_external": None,
                "p_at_10": None,
                "ef_at_10": None,
                "p_at_20": None,
                "ef_at_20": None,
                "external_top_10_genes": [],
                "external_top_20_genes": [],
            }
        )
        return result

    if llm_max_genes_int is not None and llm_max_genes_int < len(llm_score_df):
        result.update(
            {
                "status": "skipped",
                "skip_reason": (
                    f"Partial LLM run only: NM_EXP5_LLM_MAX_GENES={llm_max_genes_int}; "
                    "rerun without the cap for full metrics"
                ),
                "internal_auroc": None,
                "auroc_internal": None,
                "external_auroc": None,
                "auroc_external": None,
                "p_at_10": None,
                "ef_at_10": None,
                "p_at_20": None,
                "ef_at_20": None,
                "external_top_10_genes": [],
                "external_top_20_genes": [],
            }
        )
        return result

    score_lookup = {gene: float(score) for gene, score in model_cache.items()}
    eval_scores = eval_df["gene"].map(score_lookup).fillna(0.5).to_numpy(dtype=float)
    rank_scores = ranking_df["gene"].map(score_lookup).fillna(0.5).to_numpy(dtype=float)
    ranking_with_scores = ranking_df.copy()
    ranking_with_scores["score"] = rank_scores
    external = compute_external_metrics(ranking_with_scores, "score", external_positive_genes)

    result.update(
        {
            "internal_auroc": float(roc_auc_score(eval_df["label"], eval_scores)),
            "auroc_internal": float(roc_auc_score(eval_df["label"], eval_scores)),
            "external_auroc": external["auroc"],
            "auroc_external": external["auroc"],
            "p_at_10": external["p_at_10"],
            "ef_at_10": external["ef_at_10"],
            "p_at_20": external["p_at_20"],
            "ef_at_20": external["ef_at_20"],
            "external_top_10_genes": external["top_10_genes"],
            "external_top_20_genes": external["top_20_genes"],
            "score_summary": {
                "min": float(np.min(rank_scores)),
                "max": float(np.max(rank_scores)),
                "mean": float(np.mean(rank_scores)),
            },
        }
    )
    return result


def initialize_proxy_runtime() -> dict:
    runtime = {
        "provider": "proxy",
        "status": "ok",
        "skip_reason": None,
        "client": None,
        "available_model_ids": None,
        "max_workers": PROXY_LLM_MAX_WORKERS,
    }
    try:
        client = OpenAI(base_url=PROXY_BASE_URL, api_key=PROXY_API_KEY)
        available_model_ids = {model.id for model in client.models.list().data}
        runtime.update(
            {
                "client": client,
                "available_model_ids": available_model_ids,
            }
        )
        print(f"Proxy LLM endpoint online: {len(available_model_ids)} models listed")
    except Exception as exc:
        runtime.update(
            {
                "status": "skipped",
                "skip_reason": f"Proxy endpoint unavailable: {exc}",
            }
        )
        print(runtime["skip_reason"])
    return runtime


def initialize_gemini_runtime() -> dict:
    runtime = {
        "provider": "gemini",
        "status": "ok",
        "skip_reason": None,
        "client": None,
        "available_model_ids": None,
        "max_workers": GEMINI_LLM_MAX_WORKERS,
    }
    if genai is None:
        runtime.update(
            {
                "status": "skipped",
                "skip_reason": GEMINI_IMPORT_ERROR or "google-generativeai is not installed",
            }
        )
        print(runtime["skip_reason"])
        return runtime

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured")
    except Exception as exc:
        runtime.update(
            {
                "status": "skipped",
                "skip_reason": f"Gemini API unavailable: {exc}",
            }
        )
        print(runtime["skip_reason"])
    return runtime


def prepare_data() -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Counter],
    dict,
    pd.DataFrame,
]:
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

    ranking_df = full_df.loc[full_df["has_evidence_by_cutoff"]].copy().reset_index(drop=True)
    llm_score_df = full_df.copy().reset_index(drop=True)

    nm_set = set(NM_CORPUS_GENES)
    universe_set = set(gene_universe)
    ranking_set = set(ranking_df["gene"])
    nm_in_abor_universe = sorted(nm_set & universe_set)
    nm_evaluable = sorted(nm_set & ranking_set)
    nm_absent_from_abor = sorted(nm_set - universe_set)
    nm_present_but_no_cutoff_evidence = sorted((nm_set & universe_set) - ranking_set)
    nm_internal_positive_overlap = sorted(
        eval_df.loc[
            (eval_df["label"] == 1) & (eval_df["gene"].isin(nm_set)),
            "gene",
        ].tolist()
    )

    data_summary = {
        "n_genes_universe": int(len(gene_universe)),
        "n_genes_with_evidence_by_cutoff": int(len(ranking_df)),
        "n_internal_eval_genes": int(len(eval_df)),
        "n_internal_positives": int(eval_df["label"].sum()),
        "n_internal_negatives": int((1 - eval_df["label"]).sum()),
        "n_historical_positives_excluded": int((full_df["label_status"] == "historical_positive_excluded").sum()),
        "n_unlabeled_genes": int((full_df["label_status"] == "unlabeled").sum()),
        "nm_corpus_total_genes": int(len(NM_CORPUS_GENES)),
        "nm_genes_in_abor_universe": nm_in_abor_universe,
        "nm_genes_evaluable_at_cutoff": nm_evaluable,
        "nm_genes_absent_from_abor_universe": nm_absent_from_abor,
        "nm_genes_present_but_without_2021_evidence": nm_present_but_no_cutoff_evidence,
        "nm_internal_positive_overlap": nm_internal_positive_overlap,
        "n_nm_internal_positive_overlap": int(len(nm_internal_positive_overlap)),
        "n_internal_eval_genes_nm_holdout": int(len(eval_df) - len(nm_internal_positive_overlap)),
        "n_internal_positives_nm_holdout": int(eval_df["label"].sum() - len(nm_internal_positive_overlap)),
        "signature_pmid_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
    }
    extras = {
        "gene_universe": gene_universe,
        "nm_evaluable_set": set(nm_evaluable),
        "nm_internal_positive_overlap_set": set(nm_internal_positive_overlap),
        "pmid_conflicts": pmid_conflicts,
    }
    return data_summary, full_df, eval_df, ranking_df, year_source_counts, extras, llm_score_df


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    active_llm_models = resolve_llm_models()
    skip_ml = env_flag_enabled("NM_EXP5_SKIP_ML")

    (
        data_summary,
        full_df,
        eval_df,
        ranking_df,
        year_source_counts,
        extras,
        llm_score_df,
    ) = prepare_data()

    print(
        "Prepared Fold C data:",
        f"cutoff<={FEATURE_CUTOFF_YEAR},",
        f"internal labels={LABEL_START_YEAR}-{LABEL_END_YEAR},",
        f"internal n={data_summary['n_internal_eval_genes']},",
        f"external evaluable NM genes={len(extras['nm_evaluable_set'])}/{len(NM_CORPUS_GENES)}",
    )
    print(
        "NM holdout positives excluded from training:",
        f"{data_summary['n_nm_internal_positive_overlap']}",
        data_summary["nm_internal_positive_overlap"],
    )
    print("Active LLM models:", active_llm_models)
    if skip_ml:
        print("Skipping ML evaluation because NM_EXP5_SKIP_ML is enabled")

    external_corpus_diagnostics = compute_external_corpus_diagnostics(
        ranking_df=ranking_df,
        positive_genes=extras["nm_evaluable_set"],
    )
    print_external_corpus_diagnostics(external_corpus_diagnostics)

    eval_df_nm_holdout = (
        eval_df.loc[
            ~(
                (eval_df["label"] == 1)
                & (eval_df["gene"].isin(extras["nm_internal_positive_overlap_set"]))
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    ml_method_specs = [
        ("XGBoost", make_xgb),
        ("LightGBM", make_lgbm),
        ("CatBoost", make_catboost),
        ("Logistic Regression", make_logistic_regression),
        ("Random Forest", make_random_forest),
        ("TabPFN", make_tabpfn),
    ]

    method_results: list[dict] = []
    if not skip_ml:
        for display_name, factory in ml_method_specs:
            if display_name == "LightGBM" and LGBMClassifier is None:
                method_results.append(
                    {
                        "method": display_name,
                        "display_name": display_name,
                        "kind": "ml",
                        "status": "skipped",
                        "skip_reason": LIGHTGBM_IMPORT_ERROR or "lightgbm not installed",
                        "internal_auroc": None,
                        "auroc_internal": None,
                        "internal_cv_method": None,
                        "external_auroc": None,
                        "auroc_external": None,
                        "p_at_10": None,
                        "ef_at_10": None,
                        "p_at_20": None,
                        "ef_at_20": None,
                        "external_top_10_genes": [],
                        "external_top_20_genes": [],
                    }
                )
                continue
            if display_name == "CatBoost" and CatBoostClassifier is None:
                method_results.append(
                    {
                        "method": display_name,
                        "display_name": display_name,
                        "kind": "ml",
                        "status": "skipped",
                        "skip_reason": CATBOOST_IMPORT_ERROR or "catboost not installed",
                        "internal_auroc": None,
                        "auroc_internal": None,
                        "internal_cv_method": None,
                        "external_auroc": None,
                        "auroc_external": None,
                        "p_at_10": None,
                        "ef_at_10": None,
                        "p_at_20": None,
                        "ef_at_20": None,
                        "external_top_10_genes": [],
                        "external_top_20_genes": [],
                    }
                )
                continue
            if display_name == "TabPFN" and TabPFNClassifier is None:
                method_results.append(
                    {
                        "method": display_name,
                        "display_name": display_name,
                        "kind": "ml",
                        "status": "skipped",
                        "skip_reason": TABPFN_IMPORT_ERROR or "tabpfn not installed",
                        "internal_auroc": None,
                        "auroc_internal": None,
                        "internal_cv_method": None,
                        "external_auroc": None,
                        "auroc_external": None,
                        "p_at_10": None,
                        "ef_at_10": None,
                        "p_at_20": None,
                        "ef_at_20": None,
                        "external_top_10_genes": [],
                        "external_top_20_genes": [],
                    }
                )
                continue

            method_results.append(
                evaluate_ml_method(
                    display_name=display_name,
                    model_factory=factory,
                    eval_df=eval_df,
                    ranking_df=ranking_df,
                    external_positive_genes=extras["nm_evaluable_set"],
                )
            )
            if display_name == "XGBoost":
                holdout_result = evaluate_ml_method(
                    display_name="XGBoost (NM-holdout)",
                    model_factory=factory,
                    eval_df=eval_df_nm_holdout,
                    ranking_df=ranking_df,
                    external_positive_genes=extras["nm_evaluable_set"],
                )
                holdout_result["training_positive_exclusions"] = data_summary["nm_internal_positive_overlap"]
                method_results.append(holdout_result)

    llm_cache = load_llm_cache(LLM_CACHE_JSON)
    llm_provider_runtimes: dict[str, dict] = {}
    active_providers = {LLM_MODEL_PROVIDERS[model_name] for model_name in active_llm_models}
    if "proxy" in active_providers:
        llm_provider_runtimes["proxy"] = initialize_proxy_runtime()
    if "gemini" in active_providers:
        llm_provider_runtimes["gemini"] = initialize_gemini_runtime()

    for model_name in active_llm_models:
        provider_name = LLM_MODEL_PROVIDERS[model_name]
        display_name = f"{LLM_MODEL_DISPLAY_NAMES[model_name]} (gene-blind)"
        method_results.append(
            evaluate_llm_method(
                display_name=display_name,
                model_name=model_name,
                provider_name=provider_name,
                provider_runtime=llm_provider_runtimes[provider_name],
                llm_score_df=llm_score_df,
                eval_df=eval_df,
                ranking_df=ranking_df,
                external_positive_genes=extras["nm_evaluable_set"],
                cache=llm_cache,
                gene_blind=LLM_GENE_BLIND_DEFAULT,
            )
        )

    print_comparison_table(method_results)
    build_plot(method_results)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "label_start_year": LABEL_START_YEAR,
        "label_end_year": LABEL_END_YEAR,
        "internal_cv_request": INTERNAL_CV_REQUEST,
        "internal_cv_note": (
            "The requested LOO-CV is used only when the internal evaluation set is small. "
            "With the full-universe negative pool implied by the task, the script falls back "
            "to stratified 5-fold CV when n exceeds the configured LOO threshold."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "nm_corpus_genes_requested": NM_CORPUS_GENES,
        "data_summary": data_summary,
        "external_nm_corpus_diagnostics": external_corpus_diagnostics,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
        "methods": method_results,
        "environment": {
            "llm_base_url": PROXY_BASE_URL,
            "llm_models_all": ALL_LLM_MODELS,
            "llm_models_requested": active_llm_models,
            "proxy_base_url": PROXY_BASE_URL,
            "proxy_models_requested": PROXY_MODELS,
            "gemini_models_requested": GEMINI_MODELS,
            "gemini_http_proxy": GEMINI_HTTP_PROXY,
            "llm_batch_size": LLM_BATCH_SIZE,
            "proxy_llm_max_workers": PROXY_LLM_MAX_WORKERS,
            "gemini_llm_max_workers": GEMINI_LLM_MAX_WORKERS,
            "proxy_llm_sleep_seconds": PROXY_LLM_SLEEP_SECONDS,
            "gemini_llm_sleep_seconds": GEMINI_LLM_SLEEP_SECONDS,
            "llm_gene_blind_default": LLM_GENE_BLIND_DEFAULT,
            "llm_cache_version": LLM_CACHE_VERSION,
            "lightgbm_installed": LGBMClassifier is not None,
            "catboost_installed": CatBoostClassifier is not None,
            "tabpfn_installed": TabPFNClassifier is not None,
            "gemini_package_installed": genai is not None,
            "gemini_import_error": GEMINI_IMPORT_ERROR,
            "llm_provider_status": {
                provider_name: {
                    "status": runtime.get("status"),
                    "skip_reason": runtime.get("skip_reason"),
                    "available_model_ids": (
                        None
                        if runtime.get("available_model_ids") is None
                        else sorted(runtime["available_model_ids"])
                    ),
                }
                for provider_name, runtime in llm_provider_runtimes.items()
            },
            "llm_cache_path": str(LLM_CACHE_JSON),
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    xgb_result = next((row for row in method_results if row["display_name"] == "XGBoost"), None)
    xgb_holdout_result = next((row for row in method_results if row["display_name"] == "XGBoost (NM-holdout)"), None)
    print()
    print("NM Experiment 5 complete")
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Comparison plot: {OUTPUT_PLOT}")
    print(f"LLM cache: {LLM_CACHE_JSON}")
    print(f"Internal eval genes: {data_summary['n_internal_eval_genes']}")
    print(
        "NM evaluable overlap at cutoff:",
        f"{len(data_summary['nm_genes_evaluable_at_cutoff'])}/{len(NM_CORPUS_GENES)}",
    )
    if xgb_result is not None:
        print(
            "XGBoost external AUROC:",
            "NA" if xgb_result.get("external_auroc") is None else f"{xgb_result['external_auroc']:.4f}",
        )
    if xgb_holdout_result is not None:
        print(
            "XGBoost (NM-holdout) external AUROC:",
            "NA"
            if xgb_holdout_result.get("external_auroc") is None
            else f"{xgb_holdout_result['external_auroc']:.4f}",
        )


if __name__ == "__main__":
    main()
