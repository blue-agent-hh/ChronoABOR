#!/usr/bin/env python3

from __future__ import annotations

import inspect
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBClassifier

from nm_exp4_full_universe import (
    FEATURE_COLUMNS,
    LAYER_FILES,
    RANDOM_SEED,
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    extract_year_from_doi_like,
    is_peripheral_tissue,
    load_tables,
    make_xgb as make_nm_exp4_xgb,
    normalize_gene_tokens,
    normalize_pmid,
    resolve_year,
    to_jsonable,
)
from nm_exp5_llm_vs_ml_benchmark import make_logistic_regression as make_elastic_net_pipeline
from nm_exp_p10_second_validation import build_scored_universe
from nm_exp_p1_prior_control import SECRETED_GENES
from nm_exp_p6_secreted_ablation import build_fold_c_frame as build_within_secreted_fold_c_frame
from nm_exp_p5_external_validation import normalize_external_gene_symbol

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TARGET_CSV_PATH = BASE_DIR / "target.csv"
NM_BENCHMARK_TEX_PATH = BASE_DIR / "paper" / "supplementary_table_s1.tex"
FRICK_WORKBOOK_PATH = BASE_DIR / "external_data" / "Frick2024_SupplementaryTables.xlsx"

FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]
EXTERNAL_FOLD_SPECS = {
    "C": {"cutoff_year": 2021, "label_year_start": 2021, "label_year_end": 2024},
    "D": {"cutoff_year": 2022, "label_year_start": 2022, "label_year_end": 2025},
}
FEATURE_LAYER_ORDER = ["snp", "cpg", "deg", "hub", "mirna", "signature"]
ALL_LAYER_SPECS = {
    "deg": {"filename": "degs.csv", "gene_column": "Gene_symbol", "study_column": "PMID_DOI", "id_field_label": "PMID_DOI"},
    "snp": {"filename": "snp.csv", "gene_column": "Gene_symbol", "study_column": "PMID", "id_field_label": "PMID"},
    "cpg": {"filename": "cpg.csv", "gene_column": "Gene_symbol", "study_column": "PMID", "id_field_label": "PMID"},
    "hub": {"filename": "hubgene.csv", "gene_column": "Gene_symbl", "study_column": "PMID", "id_field_label": "PMID"},
    "mirna": {"filename": "miRNA.csv", "gene_column": "Gene_symbol", "study_column": "PMID_DOI", "id_field_label": "PMID_DOI"},
    "signature": {"filename": "signature.csv", "gene_column": "Gene_symbol", "study_column": "PMID", "id_field_label": "PMID"},
    "target": {"filename": "target.csv", "gene_column": "GeneSymbol", "study_column": "References", "id_field_label": "References"},
}
URL_PMID_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d+)", flags=re.IGNORECASE)
LITERAL_PMID_RE = re.compile(r"^\s*PMID\s*:?\s*(\d+)\s*$", flags=re.IGNORECASE)
URL_DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", flags=re.IGNORECASE)
LATEX_TEXTIT_RE = re.compile(r"\\textit\{([^}]+)\}")
SPECIAL_NM_ALIASES = {"VILIP1": "VSNL1", "NSE": "ENO2"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def provenance(script_name: str, parameters: dict, input_paths: list[Path | str]) -> dict:
    return {
        "script_name": script_name,
        "timestamp_utc": utc_timestamp(),
        "seed": RANDOM_SEED,
        "input_files": [str(Path(path)) for path in input_paths],
        "parameters": to_jsonable(parameters),
    }


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_tables() -> dict[str, pd.DataFrame]:
    tables = load_tables()
    tables["signature"] = tables["signature"].copy()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")
    tables["target"] = pd.read_csv(TARGET_CSV_PATH)
    return tables


def build_all_layer_gene_universe(tables: dict[str, pd.DataFrame] | None = None) -> list[str]:
    if tables is None:
        tables = load_all_tables()
    universe: set[str] = set()
    for layer, spec in ALL_LAYER_SPECS.items():
        if layer not in tables:
            continue
        for genes in tables[layer][spec["gene_column"]].apply(normalize_gene_tokens):
            universe.update(genes)
    return sorted(universe)


def build_feature_context(
    cutoff_year: int,
    tables: dict[str, pd.DataFrame] | None = None,
    gene_universe: list[str] | None = None,
) -> dict:
    if tables is None:
        tables = load_all_tables()
    if gene_universe is None:
        gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=cutoff_year,
    )
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)
    return {
        "tables": tables,
        "gene_universe": gene_universe,
        "pmid_year_map": pmid_year_map,
        "pmid_conflicts": pmid_conflicts,
        "feature_df": feature_df,
        "evidence_by_cutoff": evidence_by_cutoff,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
    }


def load_nm_corpus_table(tex_path: Path | None = None) -> list[dict]:
    path = NM_BENCHMARK_TEX_PATH if tex_path is None else tex_path
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith(r"\textit{"):
            continue
        parts = [part.strip() for part in line.split("&")]
        if len(parts) < 7:
            raise ValueError(f"Malformed NM corpus row: {line}")
        match = LATEX_TEXTIT_RE.search(parts[0])
        if match is None:
            raise ValueError(f"Could not parse NM corpus gene symbol from row: {line}")
        raw_gene = match.group(1).replace(r"\-", "").strip().upper()
        canonical_gene = SPECIAL_NM_ALIASES.get(raw_gene, raw_gene)
        year = int(parts[4])
        fold_c = parts[5].replace("\\", "").strip() == "Y"
        fold_d = parts[6].replace("\\", "").strip() == "Y"
        rows.append(
            {
                "raw_gene": raw_gene,
                "canonical_gene": canonical_gene,
                "year": year,
                "fold_c": fold_c,
                "fold_d": fold_d,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No NM corpus rows found in {path}")
    return rows


def build_external_fold_dataset(
    fold_key: str,
    tables: dict[str, pd.DataFrame] | None = None,
    gene_universe: list[str] | None = None,
    keep_only_genes_with_full_corpus_cutoff_evidence: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    fold_key = fold_key.upper()
    if fold_key not in EXTERNAL_FOLD_SPECS:
        raise ValueError(f"Unsupported fold key: {fold_key}")
    spec = EXTERNAL_FOLD_SPECS[fold_key]
    context = build_feature_context(cutoff_year=spec["cutoff_year"], tables=tables, gene_universe=gene_universe)
    nm_rows = load_nm_corpus_table()
    fold_flag = "fold_c" if fold_key == "C" else "fold_d"
    positive_genes = sorted({row["canonical_gene"] for row in nm_rows if row[fold_flag]})
    eval_df = context["feature_df"].copy()
    if keep_only_genes_with_full_corpus_cutoff_evidence:
        eval_df = eval_df.loc[eval_df["has_evidence_by_cutoff"]].copy().reset_index(drop=True)
    positive_set = set(positive_genes)
    eval_df["label"] = eval_df["gene"].isin(positive_set).astype(int)
    positive_in_universe = sorted(positive_set & set(context["gene_universe"]))
    positive_in_eval = sorted(positive_set & set(eval_df["gene"]))
    summary = {
        "fold_key": fold_key,
        "cutoff_year": spec["cutoff_year"],
        "label_year_start": spec["label_year_start"],
        "label_year_end": spec["label_year_end"],
        "n_nm_rows_in_checked_in_tex_table": int(len(nm_rows)),
        "n_nm_canonical_genes_in_checked_in_tex_table": int(len({row["canonical_gene"] for row in nm_rows})),
        "n_fold_positive_genes_requested": int(len(positive_genes)),
        "fold_positive_genes_requested": positive_genes,
        "n_fold_positive_genes_in_feature_universe": int(len(positive_in_universe)),
        "fold_positive_genes_in_feature_universe": positive_in_universe,
        "n_fold_positive_genes_in_eval_set": int(len(positive_in_eval)),
        "fold_positive_genes_in_eval_set": positive_in_eval,
        "fold_positive_genes_missing_from_eval_set": sorted(positive_set - set(eval_df["gene"])),
        "eval_gene_count": int(len(eval_df)),
        "eval_positive_count": int(eval_df["label"].sum()),
        "eval_negative_count": int(len(eval_df) - eval_df["label"].sum()),
        "feature_columns_used": FULL_FEATURES,
        "benchmark_source_note": (
            "Fold labels were parsed from paper/supplementary_table_s1.tex because this is the only checked-in "
            "per-gene Fold C/Fold D definition in the repo."
        ),
    }
    return eval_df.reset_index(drop=True), summary, context


def make_fixed_xgb(
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def make_nm_exp5_style_xgb(y_train: np.ndarray | None = None) -> XGBClassifier:
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


def make_l2_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _build_model(model_builder, y_train: np.ndarray):
    signature = inspect.signature(model_builder)
    if len(signature.parameters) == 0:
        return model_builder()
    if "y_train" in signature.parameters:
        return model_builder(y_train=y_train)
    return model_builder()


def cross_validated_scores(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict:
    X = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_scores = np.full(len(eval_df), np.nan, dtype=float)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"Fold {fold_idx} training set has a single class.")
        if len(np.unique(y_test)) < 2:
            raise RuntimeError(f"Fold {fold_idx} test set has a single class.")
        model = _build_model(model_builder, y_train)
        model.fit(X[train_idx], y_train)
        scores = model.predict_proba(X[test_idx])[:, 1]
        oof_scores[test_idx] = scores
        fold_aurocs.append(float(roc_auc_score(y_test, scores)))

    if np.isnan(oof_scores).any():
        raise RuntimeError("Cross-validation did not populate every held-out score.")

    return {
        "oof_scores": oof_scores,
        "fold_aurocs": fold_aurocs,
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0,
        "auroc_oof": float(roc_auc_score(y, oof_scores)),
        "auprc_oof": float(average_precision_score(y, oof_scores)),
    }


def full_fit_scores(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> np.ndarray:
    X = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    model = _build_model(model_builder, y)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def summarize_distribution(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p975": float(np.quantile(values, 0.975)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def stratified_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        pos_idx = rng.choice(positives, size=len(positives), replace=True)
        neg_idx = rng.choice(negatives, size=len(negatives), replace=True)
        sample_idx = np.concatenate([pos_idx, neg_idx])
        aucs[idx] = float(roc_auc_score(labels[sample_idx], scores[sample_idx]))

    return {
        "distribution": aucs,
        "summary": summarize_distribution(aucs),
        "ci95": {
            "lower": float(np.quantile(aucs, 0.025)),
            "upper": float(np.quantile(aucs, 0.975)),
        },
    }


def paired_stratified_bootstrap_auc_delta(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        pos_idx = rng.choice(positives, size=len(positives), replace=True)
        neg_idx = rng.choice(negatives, size=len(negatives), replace=True)
        sample_idx = np.concatenate([pos_idx, neg_idx])
        auc_a = roc_auc_score(labels[sample_idx], score_a[sample_idx])
        auc_b = roc_auc_score(labels[sample_idx], score_b[sample_idx])
        deltas[idx] = float(auc_a - auc_b)

    p_left = float(np.mean(deltas <= 0.0))
    p_right = float(np.mean(deltas >= 0.0))
    return {
        "observed_delta": float(roc_auc_score(labels, score_a) - roc_auc_score(labels, score_b)),
        "distribution": deltas,
        "summary": summarize_distribution(deltas),
        "ci95": {
            "lower": float(np.quantile(deltas, 0.025)),
            "upper": float(np.quantile(deltas, 0.975)),
        },
        "p_value_two_sided": float(min(1.0, 2.0 * min(p_left, p_right))),
    }


def rank_dataframe(eval_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    ranked_df = eval_df[["gene", "label", "num_studies"]].copy()
    ranked_df["score"] = scores
    return ranked_df.sort_values(
        by=["score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_rank_metrics_from_scores(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict:
    ranked_df = rank_dataframe(eval_df, scores)
    prevalence = float(eval_df["label"].mean())
    metrics = {
        "ranking_mode": "score_desc_num_studies_desc_gene_asc",
        "base_rate": prevalence,
        "top_genes": ranked_df.head(max(ks))["gene"].tolist(),
    }
    for k in ks:
        if k > len(ranked_df):
            continue
        top_df = ranked_df.head(k)
        hits = int(top_df["label"].sum())
        precision = float(hits / k)
        ef = float(precision / prevalence) if prevalence > 0 else float("nan")
        metrics[f"hits_at_{k}"] = hits
        metrics[f"p_at_{k}"] = precision
        metrics[f"ef_at_{k}"] = ef
    return metrics


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(sorted_values):
        end = start
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    restored = np.empty(len(values), dtype=float)
    restored[order] = midranks
    return restored


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for row_idx in range(k):
        tx[row_idx] = compute_midrank(positive_examples[row_idx])
        ty[row_idx] = compute_midrank(negative_examples[row_idx])
        tz[row_idx] = compute_midrank(predictions_sorted_transposed[row_idx])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.asarray([[float(sx)]], dtype=float)
    if np.ndim(sy) == 0:
        sy = np.asarray([[float(sy)]], dtype=float)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    order = np.argsort(-labels)
    label_1_count = int(labels.sum())
    predictions = np.vstack([score_a, score_b])[:, order]
    aucs, delong_cov = fast_delong(predictions, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ delong_cov @ contrast.T)
    if variance <= 0:
        raise RuntimeError("Computed non-positive DeLong variance.")
    z_score = float(abs(aucs[0] - aucs[1]) / math.sqrt(variance))
    p_value = float(2.0 * stats.norm.sf(z_score))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": float(aucs[0] - aucs[1]),
        "z_score": z_score,
        "p_value_two_sided": p_value,
        "variance": variance,
    }


def shap_mean_abs_for_features(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
    target_features: list[str],
) -> dict:
    X = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    model = _build_model(model_builder, y)
    model.fit(X, y)
    dmatrix = DMatrix(X, feature_names=feature_columns)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    result = {}
    for feature in target_features:
        if feature not in feature_columns:
            raise KeyError(f"Feature {feature} is not in the model feature set.")
        feature_idx = feature_columns.index(feature)
        result[feature] = float(np.mean(np.abs(contributions[:, feature_idx])))
    return result


def canonicalize_url(raw_value: object) -> str:
    if pd.isna(raw_value):
        raise ValueError("Cannot canonicalize a missing URL study identifier.")
    text = str(raw_value).strip()
    if not text:
        raise ValueError("Cannot canonicalize an empty URL study identifier.")
    decoded = unquote(text)
    parts = urlsplit(decoded)
    normalized_path = re.sub(r"/+", "/", parts.path.rstrip("/"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, parts.query, ""))


def canonicalize_study_identifier(raw_value: object, layer: str) -> tuple[str, str, str]:
    if pd.isna(raw_value):
        raise ValueError(f"Missing study identifier in layer {layer}.")
    if layer == "target":
        text = str(raw_value).strip()
        if not text:
            raise ValueError("Empty References value in target.csv.")
        pmid_match = URL_PMID_RE.search(text) or LITERAL_PMID_RE.match(text)
        if pmid_match is not None:
            pmid = pmid_match.group(1)
            return pmid, "PMID", pmid
        if text.isdigit():
            return text, "PMID", text
        doi_match = URL_DOI_RE.search(unquote(text))
        if doi_match is not None:
            doi = doi_match.group(1).upper()
            return doi, "DOI", doi
        normalized_url = canonicalize_url(text)
        return normalized_url, "URL", normalized_url

    pmid = normalize_pmid(raw_value)
    if pmid is not None:
        return pmid, "PMID", pmid
    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"Empty study identifier in layer {layer}.")
    if text.lower().startswith(("http://", "https://")):
        normalized_url = canonicalize_url(text)
        return normalized_url, "URL", normalized_url
    return text.upper(), "DOI", text.upper()


def resolve_record_year(
    layer: str,
    row: pd.Series,
    pmid_year_map: dict[str, int],
) -> tuple[int, str]:
    raw_value = row[ALL_LAYER_SPECS[layer]["study_column"]]
    direct_year = None
    if layer == "signature" and pd.notna(row.get("Year_int")):
        direct_year = int(row["Year_int"])
    _, _, year_input = canonicalize_study_identifier(raw_value, layer)
    return resolve_year(year_input, pmid_year_map, direct_year=direct_year)


def annotate_tables_with_study_metadata(
    tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, int]]:
    if tables is None:
        tables = load_all_tables()
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    annotated: dict[str, pd.DataFrame] = {}
    for layer, spec in ALL_LAYER_SPECS.items():
        df = tables[layer].copy()
        study_ids: list[str] = []
        id_types: list[str] = []
        resolve_sources: list[str] = []
        resolved_years: list[int] = []
        for _, row in df.iterrows():
            study_id, id_type, _ = canonicalize_study_identifier(row[spec["study_column"]], layer)
            year, source = resolve_record_year(layer, row, pmid_year_map)
            study_ids.append(study_id)
            id_types.append(id_type)
            resolved_years.append(int(year))
            resolve_sources.append(source)
        df["_study_id"] = study_ids
        df["_study_id_type"] = id_types
        df["_resolved_year"] = resolved_years
        df["_year_resolution_source"] = resolve_sources
        annotated[layer] = df
    return annotated, pmid_year_map, pmid_conflicts


def subset_feature_tables_by_studies(
    annotated_tables: dict[str, pd.DataFrame],
    sampled_study_ids: set[str],
) -> dict[str, pd.DataFrame]:
    subset = {}
    for layer in FEATURE_LAYER_ORDER:
        if layer not in annotated_tables:
            raise KeyError(f"Missing annotated table for layer {layer}")
        subset[layer] = annotated_tables[layer].loc[annotated_tables[layer]["_study_id"].isin(sampled_study_ids)].copy()
    return subset


def build_sheet_with_detected_header(raw_df: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
    headers = []
    seen: Counter[str] = Counter()
    for value in raw_df.iloc[header_row_idx].tolist():
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            text = "unnamed"
        seen[text] += 1
        headers.append(text if seen[text] == 1 else f"{text}__{seen[text]}")
    body = raw_df.iloc[header_row_idx + 1 :].copy().reset_index(drop=True)
    body.columns = headers
    return body


def find_header_row(raw_df: pd.DataFrame, required_tokens: list[str]) -> int:
    normalized_tokens = [token.lower() for token in required_tokens]
    for idx in range(len(raw_df)):
        row_text = ["" if pd.isna(value) else str(value).strip().lower() for value in raw_df.iloc[idx].tolist()]
        if all(any(token == cell for cell in row_text) for token in normalized_tokens):
            return idx
    raise ValueError(f"Could not find a header row containing all tokens: {required_tokens}")


def get_forward_filled_superheader(raw_df: pd.DataFrame, row_idx: int) -> pd.Series:
    return raw_df.iloc[row_idx].ffill()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2))
