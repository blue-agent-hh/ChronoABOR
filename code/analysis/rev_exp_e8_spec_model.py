#!/usr/bin/env python3

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

from nm_exp4_full_universe import build_gene_universe, normalize_gene_tokens
from nm_exp_p1_prior_control import SECRETED_GENES as LEGACY_SECRETED_GENES
from rev_exp_annotations import build_annotation_bundle
from rev_exp_common import (
    BASE_DIR,
    NM_BENCHMARK_TEX_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    _build_model,
    annotate_tables_with_study_metadata,
    build_all_layer_gene_universe,
    cross_validated_scores,
    delong_test,
    ensure_results_dir,
    full_fit_scores,
    load_all_tables,
    provenance,
    shap_mean_abs_for_features,
    summarize_distribution,
    write_json,
)
from rev_exp_e7_external_benchmark import (
    RETROSPECTIVE_REFERENCE_GENES,
    build_tvl_lookup,
    cross_validated_full_universe_predictions,
    direct_score_metrics,
    run_label_permutation_null,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / "rev_exp_e8_spec_model.json"
OUTPUT_MATRIX_CSV = RESULTS_DIR / "rev_exp_e8_feature_matrix_2021.csv"
OUTPUT_MD = RESULTS_DIR / "rev_exp_e8_replacement_map.md"

CUTOFF_YEARS = [2018, 2019, 2020, 2021, 2022]
PRIMARY_WITHIN_SECRETED_CUTOFF = 2021
ROLLING_CUTOFFS = [2019, 2020, 2021]
LEARNING_RATES = [0.05, 0.1]
N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000
PERMUTATION_ITERATIONS = 1000

SCORING_LAYERS = ["snp", "cpg", "deg", "hub", "mirna", "signature", "target"]
FEATURE_BEARING_LAYERS = ["snp", "cpg", "deg", "hub", "mirna", "signature"]
FOLD_SPECS = {
    "C": {"cutoff_year": 2021, "fold_column": "fold_c"},
    "D": {"cutoff_year": 2022, "fold_column": "fold_d"},
}
MANUSCRIPT_TEN_FEATURES = [
    "is_secreted",
    "is_ad_driver",
    "deg_lfc_mean",
    "deg_lfc_max",
    "n_evidence_total",
    "num_deg_studies",
    "num_modalities",
    "has_methylation",
    "has_snp",
    "directionality_consistency",
]
PRIOR_FEATURES = ["is_secreted", "is_ad_driver"]
TEMPORAL_FEATURES = [
    "deg_lfc_mean",
    "deg_lfc_max",
    "n_evidence_total",
    "num_deg_studies",
    "num_modalities",
    "has_methylation",
    "has_snp",
    "directionality_consistency",
]
NO_EVIDENCE_COUNT_BLOCK = [
    "n_evidence_total",
    "num_deg_studies",
    "num_modalities",
]
NO_DEG_FEATURES = [
    "deg_lfc_mean",
    "deg_lfc_max",
    "num_deg_studies",
    "directionality_consistency",
]
GENE_COLUMN_BY_LAYER = {
    "snp": "Gene_symbol",
    "cpg": "Gene_symbol",
    "deg": "Gene_symbol",
    "hub": "Gene_symbl",
    "mirna": "Gene_symbol",
    "signature": "Gene_symbol",
    "target": "GeneSymbol",
}
METHOD_LABELS = {
    "full_model": "ChronoABOR full model",
    "temporal_only": "Temporal-only model",
    "prior_only": "Prior-only model",
    "logistic_regression": "Logistic regression",
    "elastic_net": "Elastic net",
    "n_evidence_total": "Evidence count (n_evidence_total)",
    "num_modalities": "Modality count (num_modalities)",
    "is_secreted": "is_secreted",
    "is_ad_driver": "is_ad_driver",
    "random": "Random",
}
REPLACEMENT_OLD_VALUES = {
    "abstract_corpus_size": "243 studies; 4,576 entries",
    "abstract_external_aurocs": "0.983 and 0.958",
    "abstract_within_secreted": "AUROC 0.759 [0.605, 0.902]; P@5=P@10=1.0",
    "abstract_rolling_origin": "Mean AUROC 0.771 ± 0.021",
    "abstract_llm_claim": "ChronoABOR exceeds all five frontier LLMs on the within-secreted benchmark",
    "abstract_proteomics": "Indiana/Knight NES 1.635, p=0.001; Frick NES 1.526, p=0.006",
    "conclusions_resource_size": "4,576 entries; 243 studies; 2,894 genes",
    "conclusions_external": "AUROC up to 0.983; AUPRC 0.412; top-10 precision 100%; 123-fold enrichment",
    "conclusions_retrospective": "All eight established genes recovered within the top 8",
    "conclusions_prior_refinement": "57% → 100% top-10 precision",
    "conclusions_llm_claim": "Gene-blind LLMs fall to chance-level P@10 while ChronoABOR remains superior in the secreted stratum",
    "conclusions_proteomics": "External proteomics enrichment validates the ranking biologically",
    "table1_full": "Fold C 0.983 ± 0.014; Fold D 0.958 ± 0.069",
    "table1_lr": "Fold C 0.971 ± 0.019; Fold D 0.944 ± 0.081",
    "table1_en": "Fold C 0.968 ± 0.022; Fold D 0.941 ± 0.084",
    "table1_is_secreted": "Fold C 0.926; Fold D 0.926",
    "table1_is_ad_driver": "Fold C 0.814; Fold D 0.809",
    "table1_prior_only": "Fold C 0.953; Fold D 0.950",
    "table1_citation": "Fold C 0.752; Fold D 0.726 (`n_evidence_total`)",
    "table1_tvl": "Fold C 0.631; Fold D 0.632",
    "table1_random": "0.500 / 0.500",
    "table1_llm_rows": "GPT-5.2 / GPT-5.4 / Gemini full-universe rows as printed in Table 1",
    "secreted_full": "0.759 [0.605, 0.902]; P@5 1.000; P@10 1.000; EF@10 2.692",
    "secreted_prior_only": "0.647 [0.465, 0.809]; P@10 0.700",
    "secreted_layer_count": "0.638 [0.455, 0.809]; P@10 0.500",
    "secreted_sig_max_auc": "0.635 [0.468, 0.783]; P@10 0.500",
    "secreted_num_studies": "0.577 [0.385, 0.773]; P@10 0.400",
    "secreted_temporal_only": "0.388 [0.189, 0.582]; P@10 0.600",
    "secreted_is_secreted": "0.500 [0.500, 0.500]; P@10 0.371",
    "b5_mean": "0.503",
    "b5_sd": "0.068",
    "b5_p95": "0.732",
    "b5_p99": "0.759",
    "b5_max": "0.791",
    "b5_observed": "0.985",
    "b5_p": "<0.001",
    "b6_fold_counts": "32 genes; Fold C 21; Fold D 21; shared 11",
    "b6_label_windows": "Fold C 2021-2024 first reports; Fold D 2022-2025 first reports",
    "b6_n_total": "Fold C 2,585; Fold D 2,611",
    "b6_n_pos": "Fold C 21; Fold D 21",
    "b6_static_prior": "UniProt 2019-11; KEGG 2019-Q4; DisGeNET v7.0",
    "b7_fold_c_top5": "P@5 1.00 [0.60, 1.00]; EF >100x; Hypergeom 2.1e-11; Permutation <0.001",
    "b7_fold_c_top10": "P@10 1.00 [0.67, 1.00]; EF 123x; Hypergeom 9.8e-23; Permutation <0.001",
    "b7_fold_c_top20": "P@20 0.95 [0.75, 1.00]; EF 117x; Hypergeom 7.9e-45; Permutation <0.001",
    "b7_fold_c_top50": "P@50 0.42 [0.28, 0.56]; EF 52x; Permutation <0.001",
    "b7_random_rate": "0.0081",
    "b7_auprc": "Full 0.412; citation 0.094; is_secreted 0.287",
    "b8_title": "Full 2,894-gene universe; all eight in top 8; P=5.9e-13",
    "b8_MAPT": "1",
    "b8_GFAP": "2",
    "b8_APOE": "3",
    "b8_APP": "4",
    "b8_BACE1": "5",
    "b8_TREM2": "6",
    "b8_SNAP25": "7",
    "b8_NEFL": "8",
    "b9_is_secreted": "3.647",
    "b9_deg_lfc_mean": "1.748",
    "b9_deg_lfc_max": "1.219",
    "b9_is_ad_driver": "0.877",
    "b9_n_evidence_total": "0.760",
    "b10_full": "0.985",
    "b10_no_count": "0.980",
    "b10_no_deg": "0.962",
    "b10_prior": "0.953",
    "b10_temporal": "0.677",
    "b10_note": "57% → 100%",
}


def lr_key(learning_rate: float) -> str:
    return f"lr_{learning_rate:.2f}"


def make_spec_xgb(learning_rate: float, y_train: np.ndarray | None = None) -> XGBClassifier:
    scale_pos_weight = None
    if y_train is not None:
        positives = int(np.sum(y_train))
        negatives = int(len(y_train) - positives)
        if positives > 0 and negatives > 0:
            scale_pos_weight = float(negatives / positives)
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )


def make_xgb_builder(learning_rate: float):
    def builder(y_train: np.ndarray | None = None) -> XGBClassifier:
        return make_spec_xgb(learning_rate=learning_rate, y_train=y_train)

    return builder


def make_spec_l2_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=0.1,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_spec_elastic_net() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=0.1,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def load_nm_corpus_v2() -> pd.DataFrame:
    path = BASE_DIR / "nm_corpus_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing authoritative NM corpus file: {path}")
    df = pd.read_csv(path)
    expected_columns = {"gene", "protein_name", "evidence_type", "ref_key", "year", "fold_c", "fold_d"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise KeyError(f"nm_corpus_v2.csv is missing required columns: {sorted(missing)}")
    return df.sort_values("gene").reset_index(drop=True)


def prepare_annotated_tables() -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, int]]:
    tables = load_all_tables()
    annotated_tables, pmid_year_map, pmid_conflicts = annotate_tables_with_study_metadata(tables)
    annotated_tables["deg"] = annotated_tables["deg"].copy()
    annotated_tables["deg"]["logFC_num"] = pd.to_numeric(annotated_tables["deg"]["logFC"], errors="raise")
    return annotated_tables, pmid_year_map, pmid_conflicts


def initialize_feature_state(scored_universe: list[str]) -> dict[str, dict]:
    state: dict[str, dict] = {}
    for gene in scored_universe:
        state[gene] = {
            "deg_abs_sum": 0.0,
            "deg_abs_max": 0.0,
            "n_evidence_total": 0,
            "num_deg_studies": 0,
            "layer_set": set(),
            "has_methylation": 0,
            "has_snp": 0,
            "deg_up_count": 0,
            "deg_down_count": 0,
            "has_temporal_evidence_by_cutoff": False,
            "has_feature_layer_evidence_by_cutoff": False,
            "target_evidence_count": 0,
        }
    return state


def add_feature_row(
    state: dict[str, dict],
    gene: str,
    layer: str,
    row: pd.Series,
) -> None:
    record = state[gene]
    record["n_evidence_total"] += 1
    record["layer_set"].add(layer)
    record["has_temporal_evidence_by_cutoff"] = True
    if layer in FEATURE_BEARING_LAYERS:
        record["has_feature_layer_evidence_by_cutoff"] = True
    if layer == "target":
        record["target_evidence_count"] += 1
    if layer == "snp":
        record["has_snp"] = 1
    elif layer == "cpg":
        record["has_methylation"] = 1
    elif layer == "deg":
        record["num_deg_studies"] += 1
        logfc = float(row["logFC_num"])
        abs_logfc = abs(logfc)
        record["deg_abs_sum"] += abs_logfc
        record["deg_abs_max"] = max(record["deg_abs_max"], abs_logfc)
        alteration = str(row.get("Alteration", "")).strip().lower()
        has_up = "up" in alteration
        has_down = "down" in alteration
        if has_up and has_down:
            raise RuntimeError(f"Ambiguous DEG alteration value: {row.get('Alteration')}")
        if not has_up and not has_down:
            raise RuntimeError(f"Unexpected DEG alteration value: {row.get('Alteration')}")
        if has_up:
            record["deg_up_count"] += 1
        if has_down:
            record["deg_down_count"] += 1


def build_base_feature_matrix(
    cutoff_year: int,
    annotated_tables: dict[str, pd.DataFrame],
    scored_universe: list[str],
    ad_driver_gene_set: set[str],
) -> dict:
    state = initialize_feature_state(scored_universe)

    for layer in SCORING_LAYERS:
        df = annotated_tables[layer]
        gene_column = GENE_COLUMN_BY_LAYER[layer]
        eligible_df = df.loc[df["_resolved_year"] <= cutoff_year]
        for _, row in eligible_df.iterrows():
            genes = normalize_gene_tokens(row[gene_column])
            for gene in genes:
                if gene not in state:
                    continue
                add_feature_row(state, gene, layer, row)

    rows = []
    for gene in scored_universe:
        record = state[gene]
        deg_count = int(record["num_deg_studies"])
        directionality_denom = record["deg_up_count"] + record["deg_down_count"]
        directionality = (
            abs(record["deg_up_count"] - record["deg_down_count"]) / directionality_denom
            if directionality_denom > 0
            else 0.0
        )
        rows.append(
            {
                "gene": gene,
                "is_ad_driver": int(gene in ad_driver_gene_set),
                "deg_lfc_mean": float(record["deg_abs_sum"] / deg_count) if deg_count > 0 else 0.0,
                "deg_lfc_max": float(record["deg_abs_max"]),
                "n_evidence_total": int(record["n_evidence_total"]),
                "num_deg_studies": deg_count,
                "num_modalities": int(len(record["layer_set"])),
                "has_methylation": int(record["has_methylation"]),
                "has_snp": int(record["has_snp"]),
                "directionality_consistency": float(directionality),
                "has_temporal_evidence_by_cutoff": bool(record["has_temporal_evidence_by_cutoff"]),
                "has_feature_layer_evidence_by_cutoff": bool(record["has_feature_layer_evidence_by_cutoff"]),
                "target_evidence_count": int(record["target_evidence_count"]),
            }
        )

    feature_df = pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)
    added_via_target_only = feature_df.loc[
        feature_df["has_temporal_evidence_by_cutoff"] & ~feature_df["has_feature_layer_evidence_by_cutoff"],
        "gene",
    ].tolist()
    return {
        "feature_df": feature_df,
        "cutoff_year": cutoff_year,
        "n_genes": int(len(feature_df)),
        "n_with_any_temporal_evidence": int(feature_df["has_temporal_evidence_by_cutoff"].sum()),
        "n_with_feature_layer_evidence": int(feature_df["has_feature_layer_evidence_by_cutoff"].sum()),
        "n_added_via_target_only": int(len(added_via_target_only)),
        "genes_added_via_target_only": added_via_target_only,
    }


def attach_secreted_definition(base_feature_df: pd.DataFrame, secreted_gene_set: set[str]) -> pd.DataFrame:
    feature_df = base_feature_df.copy()
    feature_df["is_secreted"] = feature_df["gene"].isin(secreted_gene_set).astype(int)
    ordered_columns = ["gene"] + MANUSCRIPT_TEN_FEATURES + [
        "has_temporal_evidence_by_cutoff",
        "has_feature_layer_evidence_by_cutoff",
        "target_evidence_count",
    ]
    return feature_df[ordered_columns]


def summarize_feature_matrix(feature_df: pd.DataFrame) -> dict:
    summary = {}
    for feature in MANUSCRIPT_TEN_FEATURES:
        values = feature_df[feature].to_numpy(dtype=float)
        nonzero_count = int(np.count_nonzero(values))
        summary[feature] = {
            "nonzero_count": nonzero_count,
            "nonzero_rate": float(nonzero_count / len(feature_df)),
            "distribution": summarize_distribution(values),
        }
    return summary


def compute_cpg_year_summary(annotated_tables: dict[str, pd.DataFrame]) -> dict:
    cpg_df = annotated_tables["cpg"]
    earliest = int(cpg_df["_resolved_year"].min())
    latest = int(cpg_df["_resolved_year"].max())
    by_source = cpg_df["_year_resolution_source"].value_counts().sort_index().to_dict()
    return {
        "n_rows": int(len(cpg_df)),
        "earliest_resolved_year": earliest,
        "latest_resolved_year": latest,
        "year_resolution_source_counts": {str(k): int(v) for k, v in by_source.items()},
        "all_zero_at_or_before_2020_claim_supported": bool(earliest > 2020),
    }


def rank_eval_frame(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    secondary_column: str = "n_evidence_total",
) -> pd.DataFrame:
    if secondary_column not in eval_df.columns:
        raise KeyError(f"Missing ranking secondary column: {secondary_column}")
    ranked_df = eval_df[["gene", "label", secondary_column]].copy()
    ranked_df["score"] = scores
    return ranked_df.sort_values(
        by=["score", secondary_column, "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def constant_rank_metrics(n_total: int, n_pos: int, ks: tuple[int, ...]) -> dict:
    base_rate = float(n_pos / n_total)
    result = {
        "ranking_mode": "expected_random_tie",
        "base_rate": base_rate,
        "top_genes": None,
    }
    for k in ks:
        result[f"hits_at_{k}"] = float(base_rate * k)
        result[f"p_at_{k}"] = base_rate
        result[f"ef_at_{k}"] = 1.0
    return result


def compute_rank_metrics_from_scores_spec(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    secondary_column: str = "n_evidence_total",
    allow_constant_random_tie: bool = False,
) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    if allow_constant_random_tie and np.allclose(scores, scores[0]):
        return constant_rank_metrics(len(eval_df), int(labels.sum()), ks)

    ranked_df = rank_eval_frame(eval_df, scores, secondary_column=secondary_column)
    base_rate = float(labels.mean())
    result = {
        "ranking_mode": f"score_desc_{secondary_column}_desc_gene_asc",
        "base_rate": base_rate,
        "top_genes": ranked_df.head(max(ks))["gene"].tolist(),
    }
    for k in ks:
        if k > len(ranked_df):
            continue
        top_df = ranked_df.head(k)
        hits = int(top_df["label"].sum())
        precision = float(hits / k)
        ef = float(precision / base_rate) if base_rate > 0 else float("nan")
        result[f"hits_at_{k}"] = hits
        result[f"p_at_{k}"] = precision
        result[f"ef_at_{k}"] = ef
    return result


def stratified_bootstrap_auc_one_sided(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
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
        "summary": summarize_distribution(aucs),
        "ci95": {
            "lower": float(np.quantile(aucs, 0.025)),
            "upper": float(np.quantile(aucs, 0.975)),
        },
    }


def paired_stratified_bootstrap_auc_delta_one_sided(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
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
        auc_a = float(roc_auc_score(labels[sample_idx], score_a[sample_idx]))
        auc_b = float(roc_auc_score(labels[sample_idx], score_b[sample_idx]))
        deltas[idx] = auc_a - auc_b

    observed_delta = float(roc_auc_score(labels, score_a) - roc_auc_score(labels, score_b))
    return {
        "observed_delta": observed_delta,
        "summary": summarize_distribution(deltas),
        "ci95": {
            "lower": float(np.quantile(deltas, 0.025)),
            "upper": float(np.quantile(deltas, 0.975)),
        },
        "p_value_one_sided_full_not_better": float((1 + np.sum(deltas <= 0.0)) / (n_bootstrap + 1)),
    }


def bootstrap_topk_precision_intervals_spec(
    ranked_df: pd.DataFrame,
    ks: list[int],
    n_bootstrap: int,
    seed: int,
    secondary_column: str,
) -> dict[int, dict]:
    labels = ranked_df["label"].to_numpy(dtype=int)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    precision_dists = {k: np.empty(n_bootstrap, dtype=float) for k in ks}

    for iteration in range(n_bootstrap):
        pos_idx = rng.choice(positives, size=len(positives), replace=True)
        neg_idx = rng.choice(negatives, size=len(negatives), replace=True)
        sample_df = ranked_df.iloc[np.concatenate([pos_idx, neg_idx])].copy()
        sample_ranked = sample_df.sort_values(
            by=["score", secondary_column, "gene"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for k in ks:
            precision_dists[k][iteration] = float(sample_ranked.head(k)["label"].mean())

    result = {}
    for k, values in precision_dists.items():
        result[k] = {
            "summary": summarize_distribution(values),
            "ci95": {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            },
        }
    return result


def permutation_topk_hit_distributions(
    ranked_labels: np.ndarray,
    ks: list[int],
    n_permutations: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    hits = {k: np.empty(n_permutations, dtype=int) for k in ks}
    for iteration in range(n_permutations):
        shuffled = rng.permutation(ranked_labels)
        cumsum = np.cumsum(shuffled)
        for k in ks:
            hits[k][iteration] = int(cumsum[k - 1])
    return hits


def summarize_topk_enrichment_spec(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    ks: list[int],
    bootstrap_seed: int,
    permutation_seed: int,
    secondary_column: str = "n_evidence_total",
) -> dict:
    ranked_df = rank_eval_frame(eval_df, scores, secondary_column=secondary_column)
    labels = ranked_df["label"].to_numpy(dtype=int)
    n_total = int(len(ranked_df))
    n_pos = int(labels.sum())
    base_rate = float(n_pos / n_total)
    bootstrap = bootstrap_topk_precision_intervals_spec(
        ranked_df,
        ks=ks,
        n_bootstrap=BOOTSTRAP_ITERATIONS,
        seed=bootstrap_seed,
        secondary_column=secondary_column,
    )
    permutation_hits = permutation_topk_hit_distributions(
        labels,
        ks=ks,
        n_permutations=PERMUTATION_ITERATIONS,
        seed=permutation_seed,
    )
    rows = []
    for k in ks:
        top_df = ranked_df.head(k)
        hits = int(top_df["label"].sum())
        precision = float(hits / k)
        recall = float(hits / n_pos)
        ef = float(precision / base_rate)
        hypergeom_p = float(stats.hypergeom.sf(hits - 1, n_total, n_pos, k))
        perm_p = float((1 + np.sum(permutation_hits[k] >= hits)) / (PERMUTATION_ITERATIONS + 1))
        rows.append(
            {
                "k": k,
                "hits": hits,
                "precision_at_k": precision,
                "precision_bootstrap_ci95": bootstrap[k]["ci95"],
                "precision_bootstrap_summary": bootstrap[k]["summary"],
                "recall_at_k": recall,
                "enrichment_factor_at_k": ef,
                "hypergeometric_p_upper_tail": hypergeom_p,
                "permutation_p_upper_tail": perm_p,
                "random_background_rate": base_rate,
            }
        )

    return {
        "score_source": "full_fit_probability",
        "ranking_mode": f"score_desc_{secondary_column}_desc_gene_asc",
        "n_total": n_total,
        "n_pos": n_pos,
        "random_background_rate": base_rate,
        "top_genes_head": ranked_df.head(max(ks))["gene"].tolist(),
        "rows": rows,
    }


def evaluate_direct_method(
    eval_df: pd.DataFrame,
    method_name: str,
    scores: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    random_tie: bool = False,
) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    auc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    bootstrap = stratified_bootstrap_auc_one_sided(labels, scores)
    rank_metrics = compute_rank_metrics_from_scores_spec(
        eval_df,
        scores,
        ks=ks,
        allow_constant_random_tie=random_tie,
    )
    return {
        "method": method_name,
        "label": METHOD_LABELS.get(method_name, method_name),
        "score_source": "direct_feature",
        "auroc": auc,
        "auprc": auprc,
        "bootstrap_ci95": bootstrap["ci95"],
        "bootstrap_summary": bootstrap["summary"],
        "rank_metrics": rank_metrics,
        "scores": scores,
    }


def evaluate_learned_method(
    eval_df: pd.DataFrame,
    method_name: str,
    feature_columns: list[str],
    model_builder,
    ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict:
    cv = cross_validated_scores(eval_df, feature_columns, model_builder, n_splits=N_SPLITS, random_state=RANDOM_SEED)
    labels = eval_df["label"].to_numpy(dtype=int)
    bootstrap = stratified_bootstrap_auc_one_sided(labels, cv["oof_scores"])
    full_scores = full_fit_scores(eval_df, feature_columns, model_builder)
    rank_metrics = compute_rank_metrics_from_scores_spec(eval_df, full_scores, ks=ks)
    return {
        "method": method_name,
        "label": METHOD_LABELS.get(method_name, method_name),
        "feature_columns": feature_columns,
        "score_source": "cross_validated_oof_for_auroc_full_fit_for_ranking",
        "cv_auroc_mean": cv["auroc_mean"],
        "cv_auroc_std": cv["auroc_std"],
        "cv_auroc_oof": cv["auroc_oof"],
        "cv_auprc_oof": cv["auprc_oof"],
        "bootstrap_ci95": bootstrap["ci95"],
        "bootstrap_summary": bootstrap["summary"],
        "rank_metrics": rank_metrics,
        "oof_scores": cv["oof_scores"],
        "full_fit_scores": full_scores,
        "fold_aurocs": cv["fold_aurocs"],
    }


def fit_scores_on_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> np.ndarray:
    model = _build_model(model_builder, train_df["label"].to_numpy(dtype=int))
    model.fit(
        train_df[feature_columns].to_numpy(dtype=float),
        train_df["label"].to_numpy(dtype=int),
    )
    return model.predict_proba(eval_df[feature_columns].to_numpy(dtype=float))[:, 1]


def run_jackknife(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
    secondary_column: str = "n_evidence_total",
) -> dict:
    labels = eval_df["label"].to_numpy(dtype=int)
    positive_genes = sorted(eval_df.loc[eval_df["label"] == 1, "gene"].tolist())
    rows = []
    held_out_ranks = []
    aurocs = []

    for held_out_gene in positive_genes:
        train_df = eval_df.loc[eval_df["gene"] != held_out_gene].copy().reset_index(drop=True)
        scores = fit_scores_on_eval(train_df, eval_df, feature_columns, model_builder)
        ranked_df = rank_eval_frame(eval_df, scores, secondary_column=secondary_column)
        held_out_rank = int(ranked_df.index[ranked_df["gene"] == held_out_gene][0]) + 1
        held_out_score = float(ranked_df.loc[ranked_df["gene"] == held_out_gene, "score"].iloc[0])
        jackknife_auc = float(roc_auc_score(labels, scores))
        rows.append(
            {
                "held_out_gene": held_out_gene,
                "held_out_rank": held_out_rank,
                "held_out_score": held_out_score,
                "jackknife_auroc": jackknife_auc,
                "top_10_genes": ranked_df.head(10)["gene"].tolist(),
            }
        )
        held_out_ranks.append(held_out_rank)
        aurocs.append(jackknife_auc)

    rank_array = np.asarray(held_out_ranks, dtype=float)
    auc_array = np.asarray(aurocs, dtype=float)
    return {
        "summary": {
            "n_iterations": int(len(rows)),
            "mean_rank": float(np.mean(rank_array)),
            "median_rank": float(np.median(rank_array)),
            "std_rank": float(np.std(rank_array, ddof=1)) if len(rank_array) > 1 else 0.0,
            "fraction_top_5": float(np.mean(rank_array <= 5)),
            "fraction_top_10": float(np.mean(rank_array <= 10)),
            "mean_auroc": float(np.mean(auc_array)),
            "std_auroc": float(np.std(auc_array, ddof=1)) if len(auc_array) > 1 else 0.0,
        },
        "iterations": rows,
    }


def evaluate_within_secreted_definition(
    definition_name: str,
    feature_matrices_by_cutoff: dict[int, pd.DataFrame],
    corpus_gene_set: set[str],
    learning_rates: list[float],
    include_rolling_origin: bool,
) -> dict:
    subset_gene_mask = feature_matrices_by_cutoff[PRIMARY_WITHIN_SECRETED_CUTOFF]["is_secreted"] == 1
    subset_genes = feature_matrices_by_cutoff[PRIMARY_WITHIN_SECRETED_CUTOFF].loc[subset_gene_mask, "gene"].tolist()
    primary_df = feature_matrices_by_cutoff[PRIMARY_WITHIN_SECRETED_CUTOFF].loc[
        feature_matrices_by_cutoff[PRIMARY_WITHIN_SECRETED_CUTOFF]["gene"].isin(subset_genes)
    ].copy()
    primary_df["label"] = primary_df["gene"].isin(corpus_gene_set).astype(int)
    subset_summary = {
        "definition": definition_name,
        "subset_size": int(len(primary_df)),
        "n_positive": int(primary_df["label"].sum()),
        "n_unlabeled": int(len(primary_df) - primary_df["label"].sum()),
        "positive_genes": sorted(primary_df.loc[primary_df["label"] == 1, "gene"].tolist()),
        "unlabeled_genes": sorted(primary_df.loc[primary_df["label"] == 0, "gene"].tolist()),
        "n_genes_without_pre_cutoff_temporal_evidence": int((~primary_df["has_temporal_evidence_by_cutoff"]).sum()),
        "genes_without_pre_cutoff_temporal_evidence": sorted(
            primary_df.loc[~primary_df["has_temporal_evidence_by_cutoff"], "gene"].tolist()
        ),
        "note": "Subset membership is fixed by the static secreted prior within the 2,843-gene scored universe; genes without pre-cutoff temporal evidence remain in the subset with zero-filled temporal features.",
    }

    direct_methods = {
        "is_secreted": evaluate_direct_method(
            primary_df,
            "is_secreted",
            np.ones(len(primary_df), dtype=float),
            ks=(5, 10),
            random_tie=True,
        ),
        "is_ad_driver": evaluate_direct_method(
            primary_df,
            "is_ad_driver",
            primary_df["is_ad_driver"].to_numpy(dtype=float),
            ks=(5, 10),
        ),
        "n_evidence_total": evaluate_direct_method(
            primary_df,
            "n_evidence_total",
            primary_df["n_evidence_total"].to_numpy(dtype=float),
            ks=(5, 10),
        ),
        "num_modalities": evaluate_direct_method(
            primary_df,
            "num_modalities",
            primary_df["num_modalities"].to_numpy(dtype=float),
            ks=(5, 10),
        ),
        "prior_only": evaluate_learned_method(
            primary_df,
            "prior_only",
            PRIOR_FEATURES,
            make_spec_l2_logistic,
            ks=(5, 10),
        ),
    }

    xgb_results = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        builder = make_xgb_builder(learning_rate)
        full_result = evaluate_learned_method(
            primary_df,
            "full_model",
            MANUSCRIPT_TEN_FEATURES,
            builder,
            ks=(5, 10),
        )
        temporal_result = evaluate_learned_method(
            primary_df,
            "temporal_only",
            TEMPORAL_FEATURES,
            builder,
            ks=(5, 10),
        )
        labels = primary_df["label"].to_numpy(dtype=int)
        full_oof = full_result["oof_scores"]
        baseline_deltas = {}
        comparison_score_map = {
            "is_secreted": direct_methods["is_secreted"]["scores"],
            "is_ad_driver": direct_methods["is_ad_driver"]["scores"],
            "n_evidence_total": direct_methods["n_evidence_total"]["scores"],
            "num_modalities": direct_methods["num_modalities"]["scores"],
            "prior_only": direct_methods["prior_only"]["oof_scores"],
            "temporal_only": temporal_result["oof_scores"],
        }
        for idx, (baseline_name, baseline_scores) in enumerate(comparison_score_map.items(), start=1):
            baseline_deltas[baseline_name] = paired_stratified_bootstrap_auc_delta_one_sided(
                labels,
                full_oof,
                baseline_scores,
                seed=RANDOM_SEED + int(learning_rate * 1000) + idx,
            )
        xgb_results[key] = {
            "learning_rate": learning_rate,
            "full_model": full_result,
            "temporal_only": temporal_result,
            "paired_bootstrap_full_vs_baselines": baseline_deltas,
            "jackknife_leave_one_positive_out": run_jackknife(primary_df, MANUSCRIPT_TEN_FEATURES, builder),
        }

    rolling_origin = None
    if include_rolling_origin:
        rolling_origin = evaluate_rolling_origin(subset_genes, corpus_gene_set, feature_matrices_by_cutoff, learning_rates)

    return {
        "subset_summary": subset_summary,
        "direct_methods": direct_methods,
        "xgb_results_by_learning_rate": xgb_results,
        "rolling_origin": rolling_origin,
        "replacement_note": "The manuscript's sig_max_auc and layer_count baselines are not part of the specified ten features; n_evidence_total and num_modalities are reported as the evidence-breadth comparators instead.",
    }


def evaluate_rolling_origin(
    subset_genes: list[str],
    corpus_gene_set: set[str],
    feature_matrices_by_cutoff: dict[int, pd.DataFrame],
    learning_rates: list[float],
) -> dict:
    folds = []
    for cutoff_year in ROLLING_CUTOFFS:
        eval_df = feature_matrices_by_cutoff[cutoff_year].loc[
            feature_matrices_by_cutoff[cutoff_year]["gene"].isin(subset_genes)
        ].copy()
        eval_df["label"] = eval_df["gene"].isin(corpus_gene_set).astype(int)
        fold_payload = {
            "cutoff_year": cutoff_year,
            "n_total": int(len(eval_df)),
            "n_positive": int(eval_df["label"].sum()),
            "n_unlabeled": int(len(eval_df) - eval_df["label"].sum()),
            "baselines": {
                "is_secreted": evaluate_direct_method(
                    eval_df,
                    "is_secreted",
                    np.ones(len(eval_df), dtype=float),
                    ks=(10,),
                    random_tie=True,
                ),
                "n_evidence_total": evaluate_direct_method(
                    eval_df,
                    "n_evidence_total",
                    eval_df["n_evidence_total"].to_numpy(dtype=float),
                    ks=(10,),
                ),
                "num_modalities": evaluate_direct_method(
                    eval_df,
                    "num_modalities",
                    eval_df["num_modalities"].to_numpy(dtype=float),
                    ks=(10,),
                ),
                "prior_only": evaluate_learned_method(
                    eval_df,
                    "prior_only",
                    PRIOR_FEATURES,
                    make_spec_l2_logistic,
                    ks=(10,),
                ),
            },
            "xgb_results_by_learning_rate": {},
        }
        for learning_rate in learning_rates:
            key = lr_key(learning_rate)
            builder = make_xgb_builder(learning_rate)
            fold_payload["xgb_results_by_learning_rate"][key] = {
                "learning_rate": learning_rate,
                "full_model": evaluate_learned_method(
                    eval_df,
                    "full_model",
                    MANUSCRIPT_TEN_FEATURES,
                    builder,
                    ks=(10,),
                ),
                "temporal_only": evaluate_learned_method(
                    eval_df,
                    "temporal_only",
                    TEMPORAL_FEATURES,
                    builder,
                    ks=(10,),
                ),
            }
        folds.append(fold_payload)

    pooled_deltas = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        labels = np.concatenate(
            [
                feature_matrices_by_cutoff[fold["cutoff_year"]]
                .loc[feature_matrices_by_cutoff[fold["cutoff_year"]]["gene"].isin(subset_genes), "gene"]
                .isin(corpus_gene_set)
                .astype(int)
                .to_numpy(dtype=int)
                for fold in folds
            ]
        )
        full_scores = np.concatenate([fold["xgb_results_by_learning_rate"][key]["full_model"]["oof_scores"] for fold in folds])
        pooled_deltas[key] = {}
        baseline_map = {
            "n_evidence_total": np.concatenate([fold["baselines"]["n_evidence_total"]["scores"] for fold in folds]),
            "num_modalities": np.concatenate([fold["baselines"]["num_modalities"]["scores"] for fold in folds]),
            "prior_only": np.concatenate([fold["baselines"]["prior_only"]["oof_scores"] for fold in folds]),
            "temporal_only": np.concatenate([fold["xgb_results_by_learning_rate"][key]["temporal_only"]["oof_scores"] for fold in folds]),
        }
        for idx, (baseline_name, baseline_scores) in enumerate(baseline_map.items(), start=1):
            pooled_deltas[key][baseline_name] = paired_stratified_bootstrap_auc_delta_one_sided(
                labels,
                full_scores,
                baseline_scores,
                seed=RANDOM_SEED + 20000 + idx + int(learning_rate * 100),
            )
            pooled_deltas[key][baseline_name]["n_pooled_instances"] = int(len(labels))
            pooled_deltas[key][baseline_name]["n_positive_instances"] = int(labels.sum())
            pooled_deltas[key][baseline_name]["n_negative_instances"] = int(len(labels) - labels.sum())

    return {
        "cutoffs": folds,
        "pooled_paired_bootstrap_full_vs_baselines": pooled_deltas,
    }


def build_external_positive_status(
    fold_key: str,
    corpus_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    scored_universe_set: set[str],
    seven_layer_universe_set: set[str],
    target_only_set: set[str],
) -> list[dict]:
    fold_column = FOLD_SPECS[fold_key]["fold_column"]
    evidence_genes = set(feature_df.loc[feature_df["has_temporal_evidence_by_cutoff"], "gene"])
    status_rows = []
    for _, row in corpus_df.loc[corpus_df[fold_column]].sort_values("gene").iterrows():
        gene = row["gene"]
        in_scored_universe = gene in scored_universe_set
        in_seven_layer_universe = gene in seven_layer_universe_set
        retained = gene in evidence_genes
        if retained:
            drop_reason = None
        elif gene in target_only_set:
            drop_reason = "present only in target.csv; excluded from the 2,843-gene scored universe"
        elif not in_seven_layer_universe:
            drop_reason = "absent from all checked-in ABOR layers"
        elif not in_scored_universe:
            drop_reason = "outside feature-bearing layers and therefore outside the scored universe"
        else:
            drop_reason = "in the scored universe but lacks any pre-cutoff temporal evidence across the seven ABOR layers"
        status_rows.append(
            {
                "gene": gene,
                "year": int(row["year"]),
                "retained_in_eval_set": retained,
                "in_scored_universe": in_scored_universe,
                "in_seven_layer_universe": in_seven_layer_universe,
                "target_only_gene": gene in target_only_set,
                "drop_reason": drop_reason,
            }
        )
    return status_rows


def evaluate_external_fold(
    fold_key: str,
    corpus_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    scored_universe_set: set[str],
    seven_layer_universe_set: set[str],
    target_only_set: set[str],
    tvl_lookup: dict[str, float],
    learning_rates: list[float],
) -> dict:
    fold_column = FOLD_SPECS[fold_key]["fold_column"]
    positive_set = set(corpus_df.loc[corpus_df[fold_column], "gene"])
    eval_df = feature_df.loc[feature_df["has_temporal_evidence_by_cutoff"]].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["gene"].isin(positive_set).astype(int)
    status_rows = build_external_positive_status(
        fold_key,
        corpus_df,
        feature_df,
        scored_universe_set,
        seven_layer_universe_set,
        target_only_set,
    )

    logistic = evaluate_learned_method(
        eval_df,
        "logistic_regression",
        MANUSCRIPT_TEN_FEATURES,
        make_spec_l2_logistic,
        ks=(5, 10, 20, 50),
    )
    elastic = evaluate_learned_method(
        eval_df,
        "elastic_net",
        MANUSCRIPT_TEN_FEATURES,
        make_spec_elastic_net,
        ks=(5, 10, 20, 50),
    )
    prior_only = evaluate_learned_method(
        eval_df,
        "prior_only",
        PRIOR_FEATURES,
        make_spec_l2_logistic,
        ks=(5, 10, 20, 50),
    )
    baselines = {
        "is_secreted": evaluate_direct_method(
            eval_df,
            "is_secreted",
            eval_df["is_secreted"].to_numpy(dtype=float),
        ),
        "is_ad_driver": evaluate_direct_method(
            eval_df,
            "is_ad_driver",
            eval_df["is_ad_driver"].to_numpy(dtype=float),
        ),
        "n_evidence_total": evaluate_direct_method(
            eval_df,
            "n_evidence_total",
            eval_df["n_evidence_total"].to_numpy(dtype=float),
        ),
        "random": evaluate_direct_method(
            eval_df,
            "random",
            np.ones(len(eval_df), dtype=float),
            random_tie=True,
        ),
    }
    xgb_results = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        builder = make_xgb_builder(learning_rate)
        learned = evaluate_learned_method(
            eval_df,
            "full_model",
            MANUSCRIPT_TEN_FEATURES,
            builder,
            ks=(5, 10, 20, 50),
        )
        learned["topk_enrichment_full_fit"] = summarize_topk_enrichment_spec(
            eval_df,
            learned["full_fit_scores"],
            ks=[5, 10, 20, 50],
            bootstrap_seed=RANDOM_SEED + 1000 + int(learning_rate * 100) + (0 if fold_key == "C" else 2000),
            permutation_seed=RANDOM_SEED + 3000 + int(learning_rate * 100) + (0 if fold_key == "C" else 2000),
        )
        learned["delong_vs_n_evidence_total"] = delong_test(
            eval_df["label"].to_numpy(dtype=int),
            learned["oof_scores"],
            baselines["n_evidence_total"]["scores"],
        )
        xgb_results[key] = learned

    tvl_subset = eval_df.loc[eval_df["gene"].isin(tvl_lookup)].copy().reset_index(drop=True)
    tvl_subset["tvl"] = tvl_subset["gene"].map(tvl_lookup)
    if tvl_subset["label"].nunique() != 2:
        raise RuntimeError(f"TVL subset for Fold {fold_key} is not two-class.")
    tvl_metrics = direct_score_metrics(
        tvl_subset["label"].to_numpy(dtype=int),
        tvl_subset["tvl"].to_numpy(dtype=float),
    )

    return {
        "fold_key": fold_key,
        "cutoff_year": FOLD_SPECS[fold_key]["cutoff_year"],
        "n_total_eval_genes": int(len(eval_df)),
        "n_pos_eval_genes": int(eval_df["label"].sum()),
        "n_neg_eval_genes": int(len(eval_df) - eval_df["label"].sum()),
        "positive_gene_status": status_rows,
        "dropped_positive_genes": [row["gene"] for row in status_rows if not row["retained_in_eval_set"]],
        "n_added_to_eval_only_by_target_layer": int(
            eval_df.loc[~eval_df["has_feature_layer_evidence_by_cutoff"], "gene"].nunique()
        ),
        "genes_added_to_eval_only_by_target_layer": sorted(
            eval_df.loc[~eval_df["has_feature_layer_evidence_by_cutoff"], "gene"].tolist()
        ),
        "xgb_results_by_learning_rate": xgb_results,
        "logistic_regression": logistic,
        "elastic_net": elastic,
        "prior_only": prior_only,
        "deterministic_baselines": {
            "is_secreted": {k: v for k, v in baselines["is_secreted"].items() if k != "scores"},
            "is_ad_driver": {k: v for k, v in baselines["is_ad_driver"].items() if k != "scores"},
            "n_evidence_total": {k: v for k, v in baselines["n_evidence_total"].items() if k != "scores"},
            "random": {k: v for k, v in baselines["random"].items() if k != "scores"},
        },
        "baseline_score_arrays": {
            "is_secreted": baselines["is_secreted"]["scores"],
            "is_ad_driver": baselines["is_ad_driver"]["scores"],
            "n_evidence_total": baselines["n_evidence_total"]["scores"],
            "random": baselines["random"]["scores"],
        },
        "tvl_baseline": {
            "subset_n_total": int(len(tvl_subset)),
            "subset_n_pos": int(tvl_subset["label"].sum()),
            "auroc": tvl_metrics["auroc"],
            "auprc": tvl_metrics["auprc"],
            "note": "TVL is defined only for genes with target.csv entries, so the AUROC is evaluated on that subset only.",
        },
    }


def evaluate_external_suite(
    definition_name: str,
    feature_matrices_by_cutoff: dict[int, pd.DataFrame],
    corpus_df: pd.DataFrame,
    scored_universe: list[str],
    seven_layer_universe: list[str],
    tvl_lookup: dict[str, float],
    learning_rates: list[float],
) -> dict:
    scored_universe_set = set(scored_universe)
    seven_layer_universe_set = set(seven_layer_universe)
    target_only_set = seven_layer_universe_set - scored_universe_set
    folds = {}
    for fold_key, fold_spec in FOLD_SPECS.items():
        feature_df = feature_matrices_by_cutoff[fold_spec["cutoff_year"]]
        folds[fold_key] = evaluate_external_fold(
            fold_key,
            corpus_df,
            feature_df,
            scored_universe_set,
            seven_layer_universe_set,
            target_only_set,
            tvl_lookup,
            learning_rates,
        )
    return {
        "definition": definition_name,
        "folds": folds,
    }


def evaluate_ablation(
    fold_c_eval_df: pd.DataFrame,
    learning_rates: list[float],
    within_secreted_primary: dict,
) -> dict:
    results = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        builder = make_xgb_builder(learning_rate)
        rows = []
        specs = [
            ("full_model", MANUSCRIPT_TEN_FEATURES),
            ("no_evidence_count_block", [feature for feature in MANUSCRIPT_TEN_FEATURES if feature not in NO_EVIDENCE_COUNT_BLOCK]),
            ("no_deg_features", [feature for feature in MANUSCRIPT_TEN_FEATURES if feature not in NO_DEG_FEATURES]),
            ("prior_only", PRIOR_FEATURES),
            ("temporal_only", TEMPORAL_FEATURES),
        ]
        for model_name, feature_columns in specs:
            model_builder = make_spec_l2_logistic if model_name == "prior_only" else builder
            cv = cross_validated_scores(fold_c_eval_df, feature_columns, model_builder, n_splits=N_SPLITS, random_state=RANDOM_SEED)
            rows.append(
                {
                    "model_configuration": model_name,
                    "feature_columns": feature_columns,
                    "cv_auroc_mean": cv["auroc_mean"],
                    "cv_auroc_std": cv["auroc_std"],
                    "cv_auroc_oof": cv["auroc_oof"],
                }
            )
        full_row = next(row for row in rows if row["model_configuration"] == "full_model")
        for row in rows:
            row["delta_vs_full"] = float(row["cv_auroc_mean"] - full_row["cv_auroc_mean"])
        prior_top10 = within_secreted_primary["direct_methods"]["prior_only"]["rank_metrics"]["p_at_10"]
        full_top10 = within_secreted_primary["xgb_results_by_learning_rate"][key]["full_model"]["rank_metrics"]["p_at_10"]
        results[key] = {
            "learning_rate": learning_rate,
            "rows": rows,
            "within_secreted_top10_precision_prior_only_vs_full": {
                "prior_only_p_at_10": float(prior_top10),
                "full_model_p_at_10": float(full_top10),
                "difference": float(full_top10 - prior_top10),
            },
        }
    return results


def evaluate_shap(fold_c_eval_df: pd.DataFrame, learning_rates: list[float]) -> dict:
    result = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        shap_values = shap_mean_abs_for_features(
            fold_c_eval_df,
            MANUSCRIPT_TEN_FEATURES,
            make_xgb_builder(learning_rate),
            MANUSCRIPT_TEN_FEATURES,
        )
        ranked = sorted(shap_values.items(), key=lambda item: (-item[1], item[0]))
        result[key] = {
            "learning_rate": learning_rate,
            "mean_abs_shap_by_feature": shap_values,
            "ranked_features": [{"feature": feature, "mean_abs_shap": value} for feature, value in ranked],
        }
    return result


def evaluate_permutation_null(fold_c_eval_df: pd.DataFrame, learning_rates: list[float]) -> dict:
    results = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        null_aurocs = run_label_permutation_null(fold_c_eval_df, MANUSCRIPT_TEN_FEATURES, make_xgb_builder(learning_rate))
        observed = cross_validated_scores(
            fold_c_eval_df,
            MANUSCRIPT_TEN_FEATURES,
            make_xgb_builder(learning_rate),
            n_splits=N_SPLITS,
            random_state=RANDOM_SEED,
        )["auroc_mean"]
        results[key] = {
            "learning_rate": learning_rate,
            "n_permutations": PERMUTATION_ITERATIONS,
            "null_distribution_summary": summarize_distribution(null_aurocs),
            "percentiles": {
                "p95": float(np.quantile(null_aurocs, 0.95)),
                "p99": float(np.quantile(null_aurocs, 0.99)),
                "max": float(np.max(null_aurocs)),
            },
            "observed_cv_auroc_mean": float(observed),
            "empirical_p_value": float((1 + np.sum(null_aurocs >= observed)) / (len(null_aurocs) + 1)),
        }
    return results


def evaluate_retrospective(
    feature_df_2018: pd.DataFrame,
    fold_c_positive_set: set[str],
    learning_rates: list[float],
) -> dict:
    retrospective_feature_df = feature_df_2018.copy()
    retrospective_eval_df = retrospective_feature_df.loc[
        retrospective_feature_df["has_temporal_evidence_by_cutoff"]
    ].copy().reset_index(drop=True)
    retrospective_eval_df["label"] = retrospective_eval_df["gene"].isin(fold_c_positive_set).astype(int)
    result = {}
    for learning_rate in learning_rates:
        key = lr_key(learning_rate)
        cv = cross_validated_full_universe_predictions(
            retrospective_feature_df,
            retrospective_eval_df,
            MANUSCRIPT_TEN_FEATURES,
            make_xgb_builder(learning_rate),
        )
        ranked_full_df = retrospective_feature_df[["gene", "n_evidence_total"]].copy()
        ranked_full_df["label"] = ranked_full_df["gene"].isin(RETROSPECTIVE_REFERENCE_GENES).astype(int)
        ranked_full_df["score"] = cv["full_universe_mean_scores"]
        ranked_full_df = ranked_full_df.sort_values(
            by=["score", "n_evidence_total", "gene"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        reference_ranks = {
            gene: int(ranked_full_df.index[ranked_full_df["gene"] == gene][0]) + 1 for gene in RETROSPECTIVE_REFERENCE_GENES
        }
        top_k_needed = max(reference_ranks.values())
        hypergeom_p = float(
            stats.hypergeom.sf(
                len(RETROSPECTIVE_REFERENCE_GENES) - 1,
                len(ranked_full_df),
                len(RETROSPECTIVE_REFERENCE_GENES),
                top_k_needed,
            )
        )
        result[key] = {
            "learning_rate": learning_rate,
            "n_total_scored_universe": int(len(retrospective_feature_df)),
            "n_eval_genes_with_pre_2019_temporal_evidence": int(len(retrospective_eval_df)),
            "n_pos_eval_genes": int(retrospective_eval_df["label"].sum()),
            "cv_auroc_mean": cv["auroc_mean"],
            "cv_auroc_std": cv["auroc_std"],
            "cv_auroc_oof": cv["auroc_oof"],
            "reference_gene_ranks": reference_ranks,
            "all_reference_genes_within_top_k": int(top_k_needed),
            "hypergeometric_p_upper_tail": hypergeom_p,
        }
    return result


def build_feature_matrix_csv(
    feature_df_2021: pd.DataFrame,
    corpus_df: pd.DataFrame,
    external_primary: dict,
) -> pd.DataFrame:
    fold_c_results = external_primary["folds"]["C"]
    fold_c_positive_set = set(corpus_df.loc[corpus_df["fold_c"], "gene"])
    fold_d_positive_set = set(corpus_df.loc[corpus_df["fold_d"], "gene"])
    eval_df = feature_df_2021.loc[feature_df_2021["has_temporal_evidence_by_cutoff"]].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["gene"].isin(fold_c_positive_set).astype(int)

    score_columns = {}
    for learning_rate in LEARNING_RATES:
        key = lr_key(learning_rate)
        cv = cross_validated_full_universe_predictions(
            feature_df_2021,
            eval_df,
            MANUSCRIPT_TEN_FEATURES,
            make_xgb_builder(learning_rate),
        )
        score_columns[f"score_fold_c_cv_mean_{key}"] = cv["full_universe_mean_scores"]
        score_columns[f"score_fold_c_oof_{key}"] = np.nan
        oof_lookup = dict(zip(eval_df["gene"], cv["oof_scores"], strict=True))
        score_columns[f"score_fold_c_oof_{key}"] = feature_df_2021["gene"].map(oof_lookup)

    export_df = feature_df_2021[["gene"] + MANUSCRIPT_TEN_FEATURES].copy()
    export_df["in_nm_corpus"] = export_df["gene"].isin(set(corpus_df["gene"])).astype(int)
    export_df["fold_c"] = export_df["gene"].isin(fold_c_positive_set).astype(int)
    export_df["fold_d"] = export_df["gene"].isin(fold_d_positive_set).astype(int)
    export_df["has_temporal_evidence_by_cutoff"] = feature_df_2021["has_temporal_evidence_by_cutoff"].astype(int)
    export_df["has_feature_layer_evidence_by_cutoff"] = feature_df_2021["has_feature_layer_evidence_by_cutoff"].astype(int)
    for column_name, values in score_columns.items():
        export_df[column_name] = values
    return export_df.sort_values("gene").reset_index(drop=True)


def compare_conclusions(primary_external: dict, primary_within: dict, go_external: dict, go_within: dict) -> dict:
    primary_fold_c = primary_external["folds"]["C"]["xgb_results_by_learning_rate"]["lr_0.10"]
    go_fold_c = go_external["folds"]["C"]["xgb_results_by_learning_rate"]["lr_0.10"]
    primary_within_full = primary_within["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    go_within_full = go_within["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    return {
        "external_fold_c_delta_auroc_go_minus_kw": float(go_fold_c["cv_auroc_mean"] - primary_fold_c["cv_auroc_mean"]),
        "within_secreted_delta_auroc_go_minus_kw": float(go_within_full["cv_auroc_oof"] - primary_within_full["cv_auroc_oof"]),
        "within_secreted_delta_p10_go_minus_kw": float(
            go_within_full["rank_metrics"]["p_at_10"] - primary_within_full["rank_metrics"]["p_at_10"]
        ),
        "qualitative_summary": (
            "GO-based broadening is treated as a sensitivity analysis. The external and within-secreted conclusions are considered stable only if the AUROC and P@10 directionality remains similar under lr=0.10."
        ),
    }


def sanitize_for_json(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"scores", "oof_scores", "full_fit_scores", "baseline_score_arrays"}:
                continue
            sanitized[key] = sanitize_for_json(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_replacement_rows(payload: dict) -> list[dict]:
    external_primary = payload["primary_secreted_definition"]["external_suite"]
    within_primary = payload["primary_secreted_definition"]["within_secreted"]
    go_suite = payload["go_secreted_sensitivity"]["external_suite"]
    legacy_within = payload["legacy_curated_secreted_comparison"]["within_secreted"]
    fold_c_primary = external_primary["folds"]["C"]["xgb_results_by_learning_rate"]["lr_0.10"]
    fold_d_primary = external_primary["folds"]["D"]["xgb_results_by_learning_rate"]["lr_0.10"]
    fold_c_top10 = next(row for row in fold_c_primary["topk_enrichment_full_fit"]["rows"] if row["k"] == 10)
    retro = payload["retrospective_pre_2019_recovery"]["lr_0.10"]
    b5 = payload["fold_c_permutation_null"]["lr_0.10"]
    b9 = payload["fold_c_shap"]["lr_0.10"]["mean_abs_shap_by_feature"]
    b10_rows = {
        row["model_configuration"]: row for row in payload["fold_c_structured_ablation"]["lr_0.10"]["rows"]
    }
    within_full = within_primary["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    within_temporal = within_primary["xgb_results_by_learning_rate"]["lr_0.10"]["temporal_only"]
    within_prior = within_primary["direct_methods"]["prior_only"]
    within_counts = within_primary["subset_summary"]
    legacy_counts = legacy_within["subset_summary"]
    replacement_rows = [
        {
            "manuscript_location": "Abstract: ABOR corpus size",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_corpus_size"],
            "new_rederived_value": "242 studies; 4579 raw rows",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Abstract: external benchmark AUROCs",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_external_aurocs"],
            "new_rederived_value": (
                f"Fold C {fold_c_primary['cv_auroc_mean']:.3f} ± {fold_c_primary['cv_auroc_std']:.3f}; "
                f"Fold D {fold_d_primary['cv_auroc_mean']:.3f} ± {fold_d_primary['cv_auroc_std']:.3f} "
                f"with the specified ten-feature model (lr=0.10)"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Abstract: within-secreted primary endpoint",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_within_secreted"],
            "new_rederived_value": (
                f"UniProt KW-0964 subset n={within_counts['subset_size']} ({within_counts['n_positive']} positives, {within_counts['n_unlabeled']} unlabeled): "
                f"AUROC {within_full['cv_auroc_oof']:.3f} [{within_full['bootstrap_ci95']['lower']:.3f}, {within_full['bootstrap_ci95']['upper']:.3f}], "
                f"P@5 {within_full['rank_metrics']['p_at_5']:.3f}, P@10 {within_full['rank_metrics']['p_at_10']:.3f}; "
                f"legacy curated-list sensitivity n={legacy_counts['subset_size']} gives AUROC {legacy_within['xgb_results_by_learning_rate']['lr_0.10']['full_model']['cv_auroc_oof']:.3f}"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Abstract: rolling-origin summary",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_rolling_origin"],
            "new_rederived_value": (
                f"UniProt KW-0964 rolling-origin mean AUROC "
                f"{np.mean([fold['xgb_results_by_learning_rate']['lr_0.10']['full_model']['cv_auroc_oof'] for fold in within_primary['rolling_origin']['cutoffs']]):.3f} ± "
                f"{np.std([fold['xgb_results_by_learning_rate']['lr_0.10']['full_model']['cv_auroc_oof'] for fold in within_primary['rolling_origin']['cutoffs']], ddof=1):.3f}"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Abstract: within-secreted LLM comparison",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_llm_claim"],
            "new_rederived_value": "Not rerun under the specified-model secreted definitions in E8; withdraw until the LLM benchmark is recomputed on the new subsets.",
            "verdict": "WITHDRAWN",
        },
        {
            "manuscript_location": "Abstract: external proteomics enrichment",
            "old_published_value": REPLACEMENT_OLD_VALUES["abstract_proteomics"],
            "new_rederived_value": "Not rerun on the specified-model 2021 ranking in E8; withdraw until the proteomics enrichment is recomputed.",
            "verdict": "WITHDRAWN",
        },
        {
            "manuscript_location": "Conclusions: ABOR resource size",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_resource_size"],
            "new_rederived_value": "4579 raw rows; 242 studies; 2843 scored genes, with 2885 genes in the seven-layer union including target.csv",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Conclusions: external literature evaluation headline",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_external"],
            "new_rederived_value": (
                f"Fold C AUROC {fold_c_primary['cv_auroc_mean']:.3f} ± {fold_c_primary['cv_auroc_std']:.3f}, "
                f"OOF AUPRC {fold_c_primary['cv_auprc_oof']:.3f}, P@10 {fold_c_top10['precision_at_k']:.3f}, "
                f"EF@10 {fold_c_top10['enrichment_factor_at_k']:.1f}x"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Conclusions: retrospective recovery",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_retrospective"],
            "new_rederived_value": (
                f"All eight land within top {retro['all_reference_genes_within_top_k']}; "
                f"ranks {', '.join(f'{gene}={rank}' for gene, rank in retro['reference_gene_ranks'].items())}"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Conclusions: within-secreted prior-only refinement",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_prior_refinement"],
            "new_rederived_value": (
                f"{payload['fold_c_structured_ablation']['lr_0.10']['within_secreted_top10_precision_prior_only_vs_full']['prior_only_p_at_10']:.3f} → "
                f"{payload['fold_c_structured_ablation']['lr_0.10']['within_secreted_top10_precision_prior_only_vs_full']['full_model_p_at_10']:.3f} "
                f"under the UniProt KW-0964 subset"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Conclusions: within-secreted LLM claim",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_llm_claim"],
            "new_rederived_value": "Not rerun under the specified-model secreted definitions in E8; withdraw until recomputed.",
            "verdict": "WITHDRAWN",
        },
        {
            "manuscript_location": "Conclusions: external proteomics claim",
            "old_published_value": REPLACEMENT_OLD_VALUES["conclusions_proteomics"],
            "new_rederived_value": "Not rerun on the specified-model ranking in E8; withdraw until recomputed.",
            "verdict": "WITHDRAWN",
        },
        {
            "manuscript_location": "Table 1: ChronoABOR (full model)",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_full"],
            "new_rederived_value": f"Fold C {fold_c_primary['cv_auroc_mean']:.3f} ± {fold_c_primary['cv_auroc_std']:.3f}; Fold D {fold_d_primary['cv_auroc_mean']:.3f} ± {fold_d_primary['cv_auroc_std']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: Logistic regression (all features)",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_lr"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['logistic_regression']['cv_auroc_mean']:.3f} ± {external_primary['folds']['C']['logistic_regression']['cv_auroc_std']:.3f}; Fold D {external_primary['folds']['D']['logistic_regression']['cv_auroc_mean']:.3f} ± {external_primary['folds']['D']['logistic_regression']['cv_auroc_std']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: Elastic net (all features)",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_en"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['elastic_net']['cv_auroc_mean']:.3f} ± {external_primary['folds']['C']['elastic_net']['cv_auroc_std']:.3f}; Fold D {external_primary['folds']['D']['elastic_net']['cv_auroc_mean']:.3f} ± {external_primary['folds']['D']['elastic_net']['cv_auroc_std']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: is_secreted alone",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_is_secreted"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['deterministic_baselines']['is_secreted']['auroc']:.3f}; Fold D {external_primary['folds']['D']['deterministic_baselines']['is_secreted']['auroc']:.3f} using UniProt KW-0964",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: is_ad_driver alone",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_is_ad_driver"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['deterministic_baselines']['is_ad_driver']['auroc']:.3f}; Fold D {external_primary['folds']['D']['deterministic_baselines']['is_ad_driver']['auroc']:.3f} from KEGG hsa05010",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: prior-only (is_secreted + is_ad_driver)",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_prior_only"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['prior_only']['cv_auroc_mean']:.3f} ± {external_primary['folds']['C']['prior_only']['cv_auroc_std']:.3f}; Fold D {external_primary['folds']['D']['prior_only']['cv_auroc_mean']:.3f} ± {external_primary['folds']['D']['prior_only']['cv_auroc_std']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: citation frequency",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_citation"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['deterministic_baselines']['n_evidence_total']['auroc']:.3f}; Fold D {external_primary['folds']['D']['deterministic_baselines']['n_evidence_total']['auroc']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: TVL baseline",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_tvl"],
            "new_rederived_value": f"Subset-only control: Fold C {external_primary['folds']['C']['tvl_baseline']['auroc']:.3f} on n={external_primary['folds']['C']['tvl_baseline']['subset_n_total']}; Fold D {external_primary['folds']['D']['tvl_baseline']['auroc']:.3f} on n={external_primary['folds']['D']['tvl_baseline']['subset_n_total']}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "Table 1: random baseline",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_random"],
            "new_rederived_value": "0.500 / 0.500",
            "verdict": "CONFIRMED",
        },
        {
            "manuscript_location": "Table 1: GPT / Gemini rows",
            "old_published_value": REPLACEMENT_OLD_VALUES["table1_llm_rows"],
            "new_rederived_value": "Not rerun under the specified ten-feature model in E8; keep withdrawn until the LLM comparison is recomputed on the specified-model benchmark.",
            "verdict": "WITHDRAWN",
        },
        {
            "manuscript_location": "tab:secreted_subset: ChronoABOR full model",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_full"],
            "new_rederived_value": f"UniProt KW-0964 subset n={within_counts['subset_size']}: AUROC {within_full['cv_auroc_oof']:.3f} [{within_full['bootstrap_ci95']['lower']:.3f}, {within_full['bootstrap_ci95']['upper']:.3f}]; P@5 {within_full['rank_metrics']['p_at_5']:.3f}; P@10 {within_full['rank_metrics']['p_at_10']:.3f}; EF@10 {within_full['rank_metrics']['ef_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: prior-only model",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_prior_only"],
            "new_rederived_value": f"AUROC {within_prior['cv_auroc_oof']:.3f} [{within_prior['bootstrap_ci95']['lower']:.3f}, {within_prior['bootstrap_ci95']['upper']:.3f}]; P@10 {within_prior['rank_metrics']['p_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: layer_count baseline",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_layer_count"],
            "new_rederived_value": f"Replaced by num_modalities: AUROC {within_primary['direct_methods']['num_modalities']['auroc']:.3f} [{within_primary['direct_methods']['num_modalities']['bootstrap_ci95']['lower']:.3f}, {within_primary['direct_methods']['num_modalities']['bootstrap_ci95']['upper']:.3f}]; P@10 {within_primary['direct_methods']['num_modalities']['rank_metrics']['p_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: sig_max_auc baseline",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_sig_max_auc"],
            "new_rederived_value": f"The specified ten-feature model has no sig_max_auc feature; evidence comparators are n_evidence_total AUROC {within_primary['direct_methods']['n_evidence_total']['auroc']:.3f} and num_modalities AUROC {within_primary['direct_methods']['num_modalities']['auroc']:.3f}.",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: publication count",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_num_studies"],
            "new_rederived_value": f"Using n_evidence_total as the specified-model evidence-count comparator: AUROC {within_primary['direct_methods']['n_evidence_total']['auroc']:.3f} [{within_primary['direct_methods']['n_evidence_total']['bootstrap_ci95']['lower']:.3f}, {within_primary['direct_methods']['n_evidence_total']['bootstrap_ci95']['upper']:.3f}]; P@10 {within_primary['direct_methods']['n_evidence_total']['rank_metrics']['p_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: temporal-only model",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_temporal_only"],
            "new_rederived_value": f"AUROC {within_temporal['cv_auroc_oof']:.3f} [{within_temporal['bootstrap_ci95']['lower']:.3f}, {within_temporal['bootstrap_ci95']['upper']:.3f}]; P@10 {within_temporal['rank_metrics']['p_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "tab:secreted_subset: is_secreted constant",
            "old_published_value": REPLACEMENT_OLD_VALUES["secreted_is_secreted"],
            "new_rederived_value": f"AUROC {within_primary['direct_methods']['is_secreted']['auroc']:.3f}; expected random-tie P@10 {within_primary['direct_methods']['is_secreted']['rank_metrics']['p_at_10']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B4: has_methylation structural-zero claim",
            "old_published_value": "Earliest CpG record is 2021; has_methylation is zero for all genes at T<=2020",
            "new_rederived_value": "Unsupported under the executable year-resolution logic: all 425 CpG rows resolve to 2018 via default_2018, so has_methylation is non-zero for 210 genes at every cutoff from 2018 through 2022.",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: permutation mean AUROC",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_mean"],
            "new_rederived_value": f"{b5['null_distribution_summary']['mean']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: permutation SD",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_sd"],
            "new_rederived_value": f"{b5['null_distribution_summary']['std']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: 95th percentile",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_p95"],
            "new_rederived_value": f"{b5['percentiles']['p95']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: 99th percentile",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_p99"],
            "new_rederived_value": f"{b5['percentiles']['p99']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: permutation max",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_max"],
            "new_rederived_value": f"{b5['percentiles']['max']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: observed full-model AUROC",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_observed"],
            "new_rederived_value": f"{b5['observed_cv_auroc_mean']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B5: empirical p-value",
            "old_published_value": REPLACEMENT_OLD_VALUES["b5_p"],
            "new_rederived_value": f"{b5['empirical_p_value']:.3g}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B6: NM corpus size and fold counts",
            "old_published_value": REPLACEMENT_OLD_VALUES["b6_fold_counts"],
            "new_rederived_value": f"{len(payload['nm_corpus']['genes'])} genes; Fold C {int(payload['nm_corpus']['fold_counts']['fold_c'])}; Fold D {int(payload['nm_corpus']['fold_counts']['fold_d'])}; shared {int(payload['nm_corpus']['fold_counts']['shared'])}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B6: external label windows",
            "old_published_value": REPLACEMENT_OLD_VALUES["b6_label_windows"],
            "new_rederived_value": "The parsed fold_c/fold_d columns are authoritative. Positive row years span 2020-2025 in both columns, so the printed first-report windows are not self-consistent with the checked-in corpus table.",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B6: genes with pre-cutoff features (n_total)",
            "old_published_value": REPLACEMENT_OLD_VALUES["b6_n_total"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['n_total_eval_genes']}; Fold D {external_primary['folds']['D']['n_total_eval_genes']}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B6: benchmark positives surviving the filter (n_pos)",
            "old_published_value": REPLACEMENT_OLD_VALUES["b6_n_pos"],
            "new_rederived_value": f"Fold C {external_primary['folds']['C']['n_pos_eval_genes']}; Fold D {external_primary['folds']['D']['n_pos_eval_genes']}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B6: static-prior snapshot dates",
            "old_published_value": REPLACEMENT_OLD_VALUES["b6_static_prior"],
            "new_rederived_value": (
                f"UniProt current release {payload['annotation_bundle']['uniprot']['manifest']['release']} accessed {payload['annotation_bundle']['uniprot']['manifest']['accessed_at_utc']}; "
                f"KEGG hsa05010 current REST pull; DisGeNET unavailable so is_ad_driver is KEGG-only"
            ),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C top-5",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_fold_c_top5"],
            "new_rederived_value": _format_topk_row(fold_c_primary["topk_enrichment_full_fit"], 5),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C top-10",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_fold_c_top10"],
            "new_rederived_value": _format_topk_row(fold_c_primary["topk_enrichment_full_fit"], 10),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C top-20",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_fold_c_top20"],
            "new_rederived_value": _format_topk_row(fold_c_primary["topk_enrichment_full_fit"], 20),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C top-50",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_fold_c_top50"],
            "new_rederived_value": _format_topk_row(fold_c_primary["topk_enrichment_full_fit"], 50),
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C random background rate",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_random_rate"],
            "new_rederived_value": f"{fold_c_primary['topk_enrichment_full_fit']['random_background_rate']:.4f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B7: Fold C AUPRC note",
            "old_published_value": REPLACEMENT_OLD_VALUES["b7_auprc"],
            "new_rederived_value": f"Full {fold_c_primary['cv_auprc_oof']:.3f}; citation {external_primary['folds']['C']['deterministic_baselines']['n_evidence_total']['auprc']:.3f}; is_secreted {external_primary['folds']['C']['deterministic_baselines']['is_secreted']['auprc']:.3f}",
            "verdict": "REPLACED",
        },
        {
            "manuscript_location": "B8: title/note",
            "old_published_value": REPLACEMENT_OLD_VALUES["b8_title"],
            "new_rederived_value": f"Full 2,843-gene scored universe; all eight in top {retro['all_reference_genes_within_top_k']}; P={retro['hypergeometric_p_upper_tail']:.3g}",
            "verdict": "REPLACED",
        },
    ]
    for gene in RETROSPECTIVE_REFERENCE_GENES:
        replacement_rows.append(
            {
                "manuscript_location": f"B8: rank of {gene}",
                "old_published_value": REPLACEMENT_OLD_VALUES[f"b8_{gene}"],
                "new_rederived_value": str(retro["reference_gene_ranks"][gene]),
                "verdict": "REPLACED",
            }
        )
    replacement_rows.extend(
        [
            {
                "manuscript_location": "B9: is_secreted mean|SHAP|",
                "old_published_value": REPLACEMENT_OLD_VALUES["b9_is_secreted"],
                "new_rederived_value": f"{b9['is_secreted']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B9: deg_lfc_mean mean|SHAP|",
                "old_published_value": REPLACEMENT_OLD_VALUES["b9_deg_lfc_mean"],
                "new_rederived_value": f"{b9['deg_lfc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B9: deg_lfc_max mean|SHAP|",
                "old_published_value": REPLACEMENT_OLD_VALUES["b9_deg_lfc_max"],
                "new_rederived_value": f"{b9['deg_lfc_max']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B9: is_ad_driver mean|SHAP|",
                "old_published_value": REPLACEMENT_OLD_VALUES["b9_is_ad_driver"],
                "new_rederived_value": f"{b9['is_ad_driver']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B9: n_evidence_total mean|SHAP|",
                "old_published_value": REPLACEMENT_OLD_VALUES["b9_n_evidence_total"],
                "new_rederived_value": f"{b9['n_evidence_total']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10: full model",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_full"],
                "new_rederived_value": f"{b10_rows['full_model']['cv_auroc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10: no evidence-count feature block",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_no_count"],
                "new_rederived_value": f"{b10_rows['no_evidence_count_block']['cv_auroc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10: no DEG features",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_no_deg"],
                "new_rederived_value": f"{b10_rows['no_deg_features']['cv_auroc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10: prior only (is_secreted + is_ad_driver)",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_prior"],
                "new_rederived_value": f"{b10_rows['prior_only']['cv_auroc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10: temporal only",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_temporal"],
                "new_rederived_value": f"{b10_rows['temporal_only']['cv_auroc_mean']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "B10 note: within-secreted top-10 precision",
                "old_published_value": REPLACEMENT_OLD_VALUES["b10_note"],
                "new_rederived_value": f"{payload['fold_c_structured_ablation']['lr_0.10']['within_secreted_top10_precision_prior_only_vs_full']['prior_only_p_at_10']:.3f} → {payload['fold_c_structured_ablation']['lr_0.10']['within_secreted_top10_precision_prior_only_vs_full']['full_model_p_at_10']:.3f}",
                "verdict": "REPLACED",
            },
            {
                "manuscript_location": "Sensitivity note: GO-based secreted definition",
                "old_published_value": "Not reported",
                "new_rederived_value": (
                    f"Fold C AUROC {go_suite['folds']['C']['xgb_results_by_learning_rate']['lr_0.10']['cv_auroc_mean']:.3f} ± "
                    f"{go_suite['folds']['C']['xgb_results_by_learning_rate']['lr_0.10']['cv_auroc_std']:.3f}; "
                    f"within-secreted AUROC {payload['go_secreted_sensitivity']['within_secreted']['xgb_results_by_learning_rate']['lr_0.10']['full_model']['cv_auroc_oof']:.3f}"
                ),
                "verdict": "ADDED",
            },
        ]
    )
    return replacement_rows


def _format_topk_row(topk_payload: dict, k: int) -> str:
    row = next(item for item in topk_payload["rows"] if item["k"] == k)
    return (
        f"P@{k} {row['precision_at_k']:.3f} [{row['precision_bootstrap_ci95']['lower']:.3f}, {row['precision_bootstrap_ci95']['upper']:.3f}]; "
        f"EF {row['enrichment_factor_at_k']:.1f}x; Hypergeom {row['hypergeometric_p_upper_tail']:.3g}; "
        f"Permutation {row['permutation_p_upper_tail']:.3g}"
    )


def write_replacement_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# ChronoABOR Specified-Model Replacement Map",
        "",
        "| Manuscript location | Old published value | New specified-model value | Verdict |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['manuscript_location']} | {row['old_published_value']} | {row['new_rederived_value']} | {row['verdict']} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    start = time.time()
    ensure_results_dir()
    corpus_df = load_nm_corpus_v2()
    annotation_bundle = build_annotation_bundle()
    annotated_tables, pmid_year_map, pmid_conflicts = prepare_annotated_tables()
    scored_universe = build_gene_universe(load_all_tables())
    seven_layer_universe = build_all_layer_gene_universe(load_all_tables())
    target_only_genes = sorted(set(seven_layer_universe) - set(scored_universe))
    tvl_lookup = build_tvl_lookup(load_all_tables()["target"])

    primary_secreted_gene_set = set(annotation_bundle["uniprot"]["secreted_kw_genes"])
    go_secreted_gene_set = set(annotation_bundle["uniprot"]["secreted_go_genes"])
    ad_driver_gene_set = set(annotation_bundle["kegg"]["ad_driver_genes"])
    legacy_secreted_gene_set = set(LEGACY_SECRETED_GENES)
    corpus_gene_set = set(corpus_df["gene"])
    fold_c_positive_set = set(corpus_df.loc[corpus_df["fold_c"], "gene"])

    base_matrices = {cutoff: build_base_feature_matrix(cutoff, annotated_tables, scored_universe, ad_driver_gene_set) for cutoff in CUTOFF_YEARS}
    primary_matrices = {cutoff: attach_secreted_definition(base_matrices[cutoff]["feature_df"], primary_secreted_gene_set) for cutoff in CUTOFF_YEARS}
    go_matrices = {cutoff: attach_secreted_definition(base_matrices[cutoff]["feature_df"], go_secreted_gene_set) for cutoff in CUTOFF_YEARS}
    legacy_matrices = {cutoff: attach_secreted_definition(base_matrices[cutoff]["feature_df"], legacy_secreted_gene_set) for cutoff in CUTOFF_YEARS}

    feature_availability = {
        cutoff: summarize_feature_matrix(primary_matrices[cutoff]) for cutoff in CUTOFF_YEARS
    }
    cpg_year_summary = compute_cpg_year_summary(annotated_tables)

    external_primary = evaluate_external_suite(
        "uniprot_kw_0964",
        primary_matrices,
        corpus_df,
        scored_universe,
        seven_layer_universe,
        tvl_lookup,
        LEARNING_RATES,
    )
    within_primary = evaluate_within_secreted_definition(
        "uniprot_kw_0964",
        primary_matrices,
        corpus_gene_set,
        LEARNING_RATES,
        include_rolling_origin=True,
    )
    external_go = evaluate_external_suite(
        "uniprot_kw_or_go_extracellular",
        go_matrices,
        corpus_df,
        scored_universe,
        seven_layer_universe,
        tvl_lookup,
        LEARNING_RATES,
    )
    within_go = evaluate_within_secreted_definition(
        "uniprot_kw_or_go_extracellular",
        go_matrices,
        corpus_gene_set,
        LEARNING_RATES,
        include_rolling_origin=False,
    )
    within_legacy = evaluate_within_secreted_definition(
        "legacy_curated_secreted_genes",
        legacy_matrices,
        corpus_gene_set,
        LEARNING_RATES,
        include_rolling_origin=False,
    )

    fold_c_eval_df = primary_matrices[2021].loc[primary_matrices[2021]["has_temporal_evidence_by_cutoff"]].copy().reset_index(drop=True)
    fold_c_eval_df["label"] = fold_c_eval_df["gene"].isin(fold_c_positive_set).astype(int)

    fold_c_ablation = evaluate_ablation(fold_c_eval_df, LEARNING_RATES, within_primary)
    fold_c_shap = evaluate_shap(fold_c_eval_df, LEARNING_RATES)
    fold_c_permutation = evaluate_permutation_null(fold_c_eval_df, LEARNING_RATES)
    retrospective = evaluate_retrospective(primary_matrices[2018], fold_c_positive_set, LEARNING_RATES)

    feature_matrix_csv = build_feature_matrix_csv(primary_matrices[2021], corpus_df, external_primary)
    feature_matrix_csv.to_csv(OUTPUT_MATRIX_CSV, index=False)

    payload = {
        "provenance": provenance(
            SCRIPT_NAME,
            {
                "cutoff_years": CUTOFF_YEARS,
                "primary_within_secreted_cutoff": PRIMARY_WITHIN_SECRETED_CUTOFF,
                "rolling_cutoffs": ROLLING_CUTOFFS,
                "learning_rates": LEARNING_RATES,
                "xgboost": {
                    "n_estimators": 200,
                    "max_depth": 4,
                    "learning_rates": LEARNING_RATES,
                    "positive_class_weight": "negative_to_positive_ratio_within_each_training_fold",
                    "random_state": RANDOM_SEED,
                },
                "linear_models": {
                    "l2_logistic_C": 0.1,
                    "elastic_net_C": 0.1,
                    "elastic_net_l1_ratio": 0.5,
                },
                "eval_filter": "Genes in the 2,843-gene scored universe with at least one pre-cutoff temporal record across the seven ABOR layers; static priors alone do not qualify a gene for the external-fold eval set.",
                "within_secreted_note": within_primary["subset_summary"]["note"],
            },
            [
                BASE_DIR / "nm_corpus_v2.csv",
                NM_BENCHMARK_TEX_PATH,
                BASE_DIR / "degs.csv",
                BASE_DIR / "snp.csv",
                BASE_DIR / "cpg.csv",
                BASE_DIR / "hubgene.csv",
                BASE_DIR / "miRNA.csv",
                BASE_DIR / "signature.csv",
                BASE_DIR / "target.csv",
                BASE_DIR / "external_data" / "annotations" / "uniprot_reviewed_human_raw.tsv",
                BASE_DIR / "external_data" / "annotations" / "kegg_hsa05010_link.tsv",
                BASE_DIR / "external_data" / "annotations" / "kegg_hsa_list.tsv",
            ],
        ),
        "annotation_bundle": annotation_bundle,
        "nm_corpus": {
            "n_rows": int(len(corpus_df)),
            "genes": corpus_df["gene"].tolist(),
            "fold_counts": {
                "fold_c": int(corpus_df["fold_c"].sum()),
                "fold_d": int(corpus_df["fold_d"].sum()),
                "shared": int((corpus_df["fold_c"] & corpus_df["fold_d"]).sum()),
            },
        },
        "universe_definition": {
            "scored_universe_2843": {
                "n_genes": int(len(scored_universe)),
                "definition": "build_gene_universe() over the six feature-bearing layers",
            },
            "seven_layer_union_2885": {
                "n_genes": int(len(seven_layer_universe)),
                "definition": "Union across snp/cpg/deg/hub/mirna/signature/target",
            },
            "manuscript_claim_2894": {
                "n_genes": 2894,
                "definition": "Published manuscript claim",
            },
            "adopted_scored_universe": {
                "n_genes": int(len(scored_universe)),
                "definition": "The specified-model candidate space uses the executable 2,843-gene scored universe. target.csv contributes temporal evidence counts and modality counts only for genes already in that universe.",
            },
            "target_only_genes_excluded_from_scored_universe": {
                "n_genes": int(len(target_only_genes)),
                "genes": target_only_genes,
            },
        },
        "feature_construction": {
            "feature_names": MANUSCRIPT_TEN_FEATURES,
            "feature_availability_primary_kw_definition": feature_availability,
            "cpg_year_summary": cpg_year_summary,
            "pmid_year_map_size": int(len(pmid_year_map)),
            "pmid_year_conflicts": {str(k): int(v) for k, v in pmid_conflicts.items()},
            "base_matrix_cutoff_summaries": {
                cutoff: {
                    key: value
                    for key, value in summary.items()
                    if key != "feature_df"
                }
                for cutoff, summary in base_matrices.items()
            },
        },
        "primary_secreted_definition": {
            "definition": "UniProt keyword KW-0964 (Secreted)",
            "external_suite": external_primary,
            "within_secreted": within_primary,
        },
        "go_secreted_sensitivity": {
            "definition": "UniProt KW-0964 OR GO:0005576 / GO:0005615",
            "external_suite": external_go,
            "within_secreted": within_go,
        },
        "legacy_curated_secreted_comparison": {
            "definition": "Hand-curated 50-gene SECRETED_GENES literal from nm_exp_p1_prior_control.py",
            "within_secreted": within_legacy,
            "difference_vs_primary_kw_lr_0_10": {
                "subset_size_delta": int(
                    within_legacy["subset_summary"]["subset_size"] - within_primary["subset_summary"]["subset_size"]
                ),
                "positive_count_delta": int(
                    within_legacy["subset_summary"]["n_positive"] - within_primary["subset_summary"]["n_positive"]
                ),
                "auroc_delta": float(
                    within_legacy["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]["cv_auroc_oof"]
                    - within_primary["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]["cv_auroc_oof"]
                ),
                "p_at_10_delta": float(
                    within_legacy["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]["rank_metrics"]["p_at_10"]
                    - within_primary["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]["rank_metrics"]["p_at_10"]
                ),
            },
        },
        "fold_c_structured_ablation": fold_c_ablation,
        "fold_c_shap": fold_c_shap,
        "fold_c_permutation_null": fold_c_permutation,
        "retrospective_pre_2019_recovery": retrospective,
        "sensitivity_comparison_kw_vs_go": compare_conclusions(external_primary, within_primary, external_go, within_go),
        "deliverables": {
            "feature_matrix_csv": str(OUTPUT_MATRIX_CSV),
        },
        "runtime_seconds": float(time.time() - start),
    }

    replacement_rows = build_replacement_rows(payload)
    payload["replacement_map_rows"] = replacement_rows
    write_json(OUTPUT_JSON, sanitize_for_json(payload))
    write_replacement_markdown(replacement_rows, OUTPUT_MD)

    fold_c_lr01 = external_primary["folds"]["C"]["xgb_results_by_learning_rate"]["lr_0.10"]
    fold_d_lr01 = external_primary["folds"]["D"]["xgb_results_by_learning_rate"]["lr_0.10"]
    within_kw_lr01 = within_primary["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    within_go_lr01 = within_go["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    within_legacy_lr01 = within_legacy["xgb_results_by_learning_rate"]["lr_0.10"]["full_model"]
    print(
        f"External Fold C/D specified-model AUROC (lr=0.10): "
        f"{fold_c_lr01['cv_auroc_mean']:.3f}±{fold_c_lr01['cv_auroc_std']:.3f} / "
        f"{fold_d_lr01['cv_auroc_mean']:.3f}±{fold_d_lr01['cv_auroc_std']:.3f}"
    )
    print(
        f"Within-secreted AUROC (lr=0.10): KW-0964 n={within_primary['subset_summary']['subset_size']} "
        f"{within_kw_lr01['cv_auroc_oof']:.3f}; GO-broader n={within_go['subset_summary']['subset_size']} "
        f"{within_go_lr01['cv_auroc_oof']:.3f}; legacy curated n={within_legacy['subset_summary']['subset_size']} "
        f"{within_legacy_lr01['cv_auroc_oof']:.3f}"
    )
    print(
        f"CpG earliest resolved year = {cpg_year_summary['earliest_resolved_year']}; "
        f"structural-zero claim at T<=2020 supported = {cpg_year_summary['all_zero_at_or_before_2020_claim_supported']}"
    )
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MATRIX_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
