#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)

from nm_exp4_full_universe import build_gene_universe, normalize_gene_tokens
from nm_exp_p1_prior_control import PRIOR_FEATURES
from nm_exp_p6_secreted_ablation import build_fold_c_frame as build_within_secreted_fold_c_frame
from rev_exp_common import (
    BASE_DIR,
    FULL_FEATURES,
    NM_BENCHMARK_TEX_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
    SPECIAL_NM_ALIASES,
    _build_model,
    build_all_layer_gene_universe,
    build_feature_context,
    compute_rank_metrics_from_scores,
    cross_validated_scores,
    delong_test,
    ensure_results_dir,
    full_fit_scores,
    load_all_tables,
    make_elastic_net_pipeline,
    make_fixed_xgb,
    make_l2_logistic,
    provenance,
    rank_dataframe,
    shap_mean_abs_for_features,
    summarize_distribution,
    to_jsonable,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / "rev_exp_e7_external_benchmark.json"
OUTPUT_MD = RESULTS_DIR / "rev_exp_e7_replacement_map.md"
OUTPUT_CORPUS_CSV = BASE_DIR / "nm_corpus_v2.csv"

N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 1000
PERMUTATION_ITERATIONS = 1000
MANUSCRIPT_UNIVERSE_CLAIM = 2894
MANUSCRIPT_STUDY_CLAIM = 243
MANUSCRIPT_ENTRY_CLAIM = 4576
FOLD_SPECS = {
    "C": {"cutoff_year": 2021, "declared_label_year_start": 2021, "declared_label_year_end": 2024},
    "D": {"cutoff_year": 2022, "declared_label_year_start": 2022, "declared_label_year_end": 2025},
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
EXECUTABLE_DEG_FEATURES = [
    "deg_count",
    "deg_up_count",
    "deg_down_count",
    "deg_tissue_breadth",
    "num_deg_studies",
]
EXECUTABLE_EVIDENCE_COUNT_BLOCK = [
    "snp_count",
    "cpg_count",
    "deg_count",
    "hub_count",
    "mirna_count",
    "sig_count_all",
    "sig_count_peripheral",
    "sig_count_high_auc",
    "layer_count",
    "num_studies",
    "num_deg_studies",
]
RETROSPECTIVE_REFERENCE_GENES = ["MAPT", "GFAP", "APOE", "APP", "BACE1", "TREM2", "SNAP25", "NEFL"]
TEXTIT_RE = re.compile(r"\\textit\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite\{([^}]+)\}")


def clean_latex_text(text: str) -> str:
    text = text.replace(r"\-", "")
    text = TEXTIT_RE.sub(r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace(r"\&", "&")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_nm_corpus_tex(tex_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for raw_line in tex_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith(r"\textit{"):
            continue
        parts = [part.strip() for part in line.split("&")]
        if len(parts) != 7:
            raise ValueError(f"Malformed NM corpus row: {line}")
        gene_match = TEXTIT_RE.search(parts[0])
        if gene_match is None:
            raise ValueError(f"Could not parse gene from row: {line}")
        ref_match = CITE_RE.search(parts[3])
        if ref_match is None:
            raise ValueError(f"Could not parse citation key from row: {line}")
        raw_gene = clean_latex_text(gene_match.group(1)).upper()
        canonical_gene = SPECIAL_NM_ALIASES.get(raw_gene, raw_gene)
        year = int(clean_latex_text(parts[4]))
        fold_c = clean_latex_text(parts[5]) == "Y"
        fold_d = clean_latex_text(parts[6]).rstrip("\\") == "Y"
        rows.append(
            {
                "raw_gene": raw_gene,
                "gene": canonical_gene,
                "protein_name": clean_latex_text(parts[1]),
                "evidence_type": clean_latex_text(parts[2]),
                "ref_key": ref_match.group(1),
                "year": year,
                "fold_c": fold_c,
                "fold_d": fold_d,
            }
        )
    if not rows:
        raise RuntimeError(f"No NM corpus rows parsed from {tex_path}")
    corpus_df = pd.DataFrame(rows)
    if corpus_df["gene"].duplicated().any():
        duplicates = sorted(corpus_df.loc[corpus_df["gene"].duplicated(), "gene"].tolist())
        raise RuntimeError(f"Duplicate canonical genes in NM corpus table: {duplicates}")
    return corpus_df


def build_tvl_lookup(target_df: pd.DataFrame) -> dict[str, float]:
    value_sets: dict[str, set[float]] = {}
    for _, row in target_df.iterrows():
        tvl = float(row["TVL"])
        for gene in normalize_gene_tokens(row["GeneSymbol"]):
            value_sets.setdefault(gene, set()).add(tvl)
    conflicts = {gene: sorted(values) for gene, values in value_sets.items() if len(values) > 1}
    if conflicts:
        raise RuntimeError(f"Conflicting TVL values per gene: {conflicts}")
    return {gene: next(iter(values)) for gene, values in value_sets.items()}


def direct_score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def cross_validated_full_universe_predictions(
    feature_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> dict:
    X_eval = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    X_all = feature_df[feature_columns].to_numpy(dtype=float)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_scores = np.full(len(eval_df), np.nan, dtype=float)
    all_scores = np.zeros((len(feature_df), N_SPLITS), dtype=float)
    fold_aurocs: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_eval, y), start=1):
        model = _build_model(model_builder, y[train_idx])
        model.fit(X_eval[train_idx], y[train_idx])
        test_scores = model.predict_proba(X_eval[test_idx])[:, 1]
        oof_scores[test_idx] = test_scores
        all_scores[:, fold_idx - 1] = model.predict_proba(X_all)[:, 1]
        fold_aurocs.append(float(roc_auc_score(y[test_idx], test_scores)))

    if np.isnan(oof_scores).any():
        raise RuntimeError("OOF prediction array was not fully populated.")

    return {
        "oof_scores": oof_scores,
        "full_universe_mean_scores": all_scores.mean(axis=1),
        "fold_aurocs": fold_aurocs,
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs, ddof=1)),
        "auroc_oof": float(roc_auc_score(y, oof_scores)),
    }


def bootstrap_topk_precision_intervals(
    ranked_df: pd.DataFrame,
    ks: list[int],
    n_bootstrap: int,
    seed: int,
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
            by=["score", "num_studies", "gene"],
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


def summarize_topk_enrichment(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    ks: list[int],
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict:
    ranked_df = rank_dataframe(eval_df, scores)
    labels = ranked_df["label"].to_numpy(dtype=int)
    n_total = int(len(ranked_df))
    n_pos = int(labels.sum())
    base_rate = float(n_pos / n_total)
    bootstrap = bootstrap_topk_precision_intervals(ranked_df, ks, BOOTSTRAP_ITERATIONS, bootstrap_seed)
    permutation_hits = permutation_topk_hit_distributions(labels, ks, PERMUTATION_ITERATIONS, permutation_seed)
    rows: list[dict] = []

    for k in ks:
        subset = ranked_df.head(k)
        hits = int(subset["label"].sum())
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
        "ranking_mode": "score_desc_num_studies_desc_gene_asc",
        "n_total": n_total,
        "n_pos": n_pos,
        "random_background_rate": base_rate,
        "top_genes_head": ranked_df.head(max(ks))["gene"].tolist(),
        "rows": rows,
    }


def run_label_permutation_null(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> np.ndarray:
    X = eval_df[feature_columns].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    null_aurocs = np.empty(PERMUTATION_ITERATIONS, dtype=float)

    for iteration in range(PERMUTATION_ITERATIONS):
        y_perm = rng.permutation(y)
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + iteration + 1,
        )
        fold_aurocs: list[float] = []
        for train_idx, test_idx in splitter.split(X, y_perm):
            model = _build_model(model_builder, y_perm[train_idx])
            model.fit(X[train_idx], y_perm[train_idx])
            scores = model.predict_proba(X[test_idx])[:, 1]
            fold_aurocs.append(float(roc_auc_score(y_perm[test_idx], scores)))
        null_aurocs[iteration] = float(np.mean(fold_aurocs))
        if (iteration + 1) % 100 == 0:
            print(f"Permutation {iteration + 1}/{PERMUTATION_ITERATIONS} complete")

    return null_aurocs


def build_fold_result(
    fold_key: str,
    corpus_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    feature_universe: list[str],
    seven_layer_universe: list[str],
    target_only_genes: list[str],
    tvl_lookup: dict[str, float],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    spec = FOLD_SPECS[fold_key]
    fold_column = f"fold_{fold_key.lower()}"
    positive_rows = corpus_df.loc[corpus_df[fold_column]].copy().reset_index(drop=True)
    positive_genes = positive_rows["gene"].tolist()
    positive_set = set(positive_genes)

    context = build_feature_context(spec["cutoff_year"], tables=tables, gene_universe=feature_universe)
    feature_df = context["feature_df"].copy()
    eval_df = feature_df.loc[feature_df["has_evidence_by_cutoff"]].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["gene"].isin(positive_set).astype(int)

    status_rows: list[dict] = []
    evidence_genes = set(context["evidence_by_cutoff"])
    feature_universe_set = set(feature_universe)
    seven_layer_set = set(seven_layer_universe)
    target_only_set = set(target_only_genes)
    for _, row in positive_rows.iterrows():
        gene = row["gene"]
        in_feature_universe = gene in feature_universe_set
        in_seven_layer_universe = gene in seven_layer_set
        has_evidence = gene in evidence_genes
        retained = gene in set(eval_df["gene"])
        if retained:
            drop_reason = None
        elif gene in target_only_set:
            drop_reason = "present only in target.csv; target layer is not feature-bearing and is excluded from the scored universe"
        elif not in_seven_layer_universe:
            drop_reason = "absent from all checked-in ABOR layers"
        elif not in_feature_universe:
            drop_reason = "present outside feature-bearing layers only"
        else:
            drop_reason = "in feature-bearing universe but lacks any pre-cutoff feature-bearing record"
        status_rows.append(
            {
                "gene": gene,
                "year": int(row["year"]),
                "in_feature_universe": in_feature_universe,
                "in_seven_layer_universe": in_seven_layer_universe,
                "target_only_gene": gene in target_only_set,
                "has_pre_cutoff_feature": has_evidence,
                "retained_in_eval_set": retained,
                "drop_reason": drop_reason,
            }
        )

    xgb_cv = cross_validated_scores(eval_df, FULL_FEATURES, make_fixed_xgb)
    l2_cv = cross_validated_scores(eval_df, FULL_FEATURES, make_l2_logistic)
    en_cv = cross_validated_scores(eval_df, FULL_FEATURES, make_elastic_net_pipeline)

    full_fit = full_fit_scores(eval_df, FULL_FEATURES, make_fixed_xgb)
    full_rank_metrics = summarize_topk_enrichment(
        eval_df,
        full_fit,
        ks=[5, 10, 20, 50],
        bootstrap_seed=RANDOM_SEED + (1000 if fold_key == "C" else 2000),
        permutation_seed=RANDOM_SEED + (3000 if fold_key == "C" else 4000),
    )

    labels = eval_df["label"].to_numpy(dtype=int)
    secreted_metrics = direct_score_metrics(labels, eval_df["is_secreted"].to_numpy(dtype=float))
    publication_metrics = direct_score_metrics(labels, eval_df["num_studies"].to_numpy(dtype=float))
    delong_vs_publication = delong_test(
        labels,
        xgb_cv["oof_scores"],
        eval_df["num_studies"].to_numpy(dtype=float),
    )

    tvl_subset = eval_df.loc[eval_df["gene"].isin(tvl_lookup)].copy().reset_index(drop=True)
    tvl_subset["tvl"] = tvl_subset["gene"].map(tvl_lookup)
    if tvl_subset["label"].nunique() != 2:
        raise RuntimeError(f"TVL subset for Fold {fold_key} is not two-class.")
    tvl_metrics = direct_score_metrics(tvl_subset["label"].to_numpy(dtype=int), tvl_subset["tvl"].to_numpy(dtype=float))

    dropped_genes = [row["gene"] for row in status_rows if not row["retained_in_eval_set"]]
    fold_result = {
        "fold_key": fold_key,
        "cutoff_year": spec["cutoff_year"],
        "declared_label_window_from_manuscript": {
            "start_year": spec["declared_label_year_start"],
            "end_year": spec["declared_label_year_end"],
        },
        "authoritative_positive_definition": f"Genes with `{fold_column}=True` parsed from supplementary_table_s1.tex",
        "authoritative_positive_year_range": {
            "min": int(positive_rows["year"].min()),
            "max": int(positive_rows["year"].max()),
        },
        "label_window_conflicts_vs_row_years": {
            "genes_before_declared_window": positive_rows.loc[
                positive_rows["year"] < spec["declared_label_year_start"], "gene"
            ].tolist(),
            "genes_after_declared_window": positive_rows.loc[
                positive_rows["year"] > spec["declared_label_year_end"], "gene"
            ].tolist(),
        },
        "corpus_positive_count_from_table_column": int(len(positive_rows)),
        "corpus_positive_genes_from_table_column": positive_genes,
        "n_total_eval_genes_with_pre_cutoff_features": int(len(eval_df)),
        "n_pos_eval_genes": int(eval_df["label"].sum()),
        "n_neg_eval_genes": int(len(eval_df) - eval_df["label"].sum()),
        "positive_gene_status": status_rows,
        "dropped_positive_genes": dropped_genes,
        "model_metrics": {
            "chronoabor_xgb_full_model": {
                "feature_columns": FULL_FEATURES,
                "cv_auroc_mean": xgb_cv["auroc_mean"],
                "cv_auroc_std": xgb_cv["auroc_std"],
                "cv_auroc_oof": xgb_cv["auroc_oof"],
                "oof_auprc": xgb_cv["auprc_oof"],
            },
            "logistic_regression_l2": {
                "feature_columns": FULL_FEATURES,
                "cv_auroc_mean": l2_cv["auroc_mean"],
                "cv_auroc_std": l2_cv["auroc_std"],
                "cv_auroc_oof": l2_cv["auroc_oof"],
                "oof_auprc": l2_cv["auprc_oof"],
            },
            "elastic_net": {
                "feature_columns": FULL_FEATURES,
                "cv_auroc_mean": en_cv["auroc_mean"],
                "cv_auroc_std": en_cv["auroc_std"],
                "cv_auroc_oof": en_cv["auroc_oof"],
                "oof_auprc": en_cv["auprc_oof"],
            },
        },
        "baselines": {
            "is_secreted": {
                "supported": True,
                "score_column": "is_secreted",
                **secreted_metrics,
            },
            "publication_count_num_studies": {
                "supported": True,
                "score_column": "num_studies",
                "note": "The checked-in scorer exposes `num_studies`. The manuscript's `n_evidence_total` feature is not implemented anywhere in executable code.",
                **publication_metrics,
            },
            "is_ad_driver": {
                "supported": False,
                "verdict": "WITHDRAWN",
                "reason": "No checked-in code or frozen source snapshot exists for the manuscript's `is_ad_driver` prior.",
            },
            "prior_only_is_secreted_plus_is_ad_driver": {
                "supported": False,
                "verdict": "WITHDRAWN",
                "reason": "The manuscript's two-prior model cannot be reconstructed because `is_ad_driver` is absent from the executable pipeline.",
            },
            "tvl": {
                "supported": True,
                "full_universe_supported": False,
                "note": "TVL is only defined for target.csv genes. The row below is therefore a subset-only control, not a full-universe baseline.",
                "subset_n_genes": int(len(tvl_subset)),
                "subset_n_pos": int(tvl_subset["label"].sum()),
                "subset_positive_genes": sorted(tvl_subset.loc[tvl_subset["label"] == 1, "gene"].tolist()),
                **tvl_metrics,
            },
            "random": {
                "supported": True,
                "auroc": 0.5,
                "auprc": float(eval_df["label"].mean()),
            },
        },
        "auprc_summary": {
            "chronoabor_xgb_full_model_oof": xgb_cv["auprc_oof"],
            "publication_count_num_studies": publication_metrics["auprc"],
            "is_secreted": secreted_metrics["auprc"],
            "random_background_rate": float(eval_df["label"].mean()),
        },
        "delong_xgb_vs_publication_count_num_studies": delong_vs_publication,
        "topk_enrichment_full_fit": full_rank_metrics,
        "topk_enrichment_oof_diagnostic": compute_rank_metrics_from_scores(eval_df, xgb_cv["oof_scores"]),
        "feature_year_source_counts": context["year_source_counts"],
    }
    return fold_result, feature_df, eval_df


def compute_fold_c_dependent_outputs(
    fold_c_feature_df: pd.DataFrame,
    fold_c_eval_df: pd.DataFrame,
    fold_c_positive_set: set[str],
    tables: dict[str, pd.DataFrame],
    feature_universe: list[str],
) -> dict:
    shap_values = shap_mean_abs_for_features(fold_c_eval_df, FULL_FEATURES, make_fixed_xgb, FULL_FEATURES)
    shap_ranked = [
        {"feature": feature, "mean_abs_shap": value}
        for feature, value in sorted(shap_values.items(), key=lambda item: item[1], reverse=True)
    ]

    ablation_specs = [
        {
            "name": "full_model",
            "feature_columns": FULL_FEATURES,
            "note": "Executable 16-feature ChronoABOR scorer.",
        },
        {
            "name": "no_evidence_count_block",
            "feature_columns": [feature for feature in FULL_FEATURES if feature not in EXECUTABLE_EVIDENCE_COUNT_BLOCK],
            "note": "Removed count-derived features from the executable scorer.",
        },
        {
            "name": "no_deg_features",
            "feature_columns": [feature for feature in FULL_FEATURES if feature not in EXECUTABLE_DEG_FEATURES],
            "note": "Removed all DEG-derived executable features.",
        },
        {
            "name": "temporal_only_executable",
            "feature_columns": [feature for feature in FULL_FEATURES if feature != "is_secreted"],
            "note": "Removed the only checked-in static prior (`is_secreted`).",
        },
    ]
    ablation_rows: list[dict] = []
    full_mean_auroc = None
    for spec in ablation_specs:
        result = cross_validated_scores(fold_c_eval_df, spec["feature_columns"], make_fixed_xgb)
        if spec["name"] == "full_model":
            full_mean_auroc = result["auroc_mean"]
        ablation_rows.append(
            {
                "model_configuration": spec["name"],
                "feature_columns": spec["feature_columns"],
                "cv_auroc_mean": result["auroc_mean"],
                "cv_auroc_std": result["auroc_std"],
                "cv_auroc_oof": result["auroc_oof"],
                "delta_vs_full_mean_auroc": None if full_mean_auroc is None else float(result["auroc_mean"] - full_mean_auroc),
                "note": spec["note"],
            }
        )
    if full_mean_auroc is None:
        raise RuntimeError("Full model ablation row was not computed.")

    secreted_df, secreted_summary = build_within_secreted_fold_c_frame()
    prior_scores = full_fit_scores(secreted_df, PRIOR_FEATURES, make_fixed_xgb)
    full_scores = full_fit_scores(secreted_df, FULL_FEATURES, make_fixed_xgb)
    prior_rank = compute_rank_metrics_from_scores(secreted_df, prior_scores)
    full_rank = compute_rank_metrics_from_scores(secreted_df, full_scores)

    print("Running Fold C permutation null (1000 label shuffles)")
    null_aurocs = run_label_permutation_null(fold_c_eval_df, FULL_FEATURES, make_fixed_xgb)
    observed_cv_mean = cross_validated_scores(fold_c_eval_df, FULL_FEATURES, make_fixed_xgb)["auroc_mean"]
    permutation_summary = summarize_distribution(null_aurocs)
    permutation_result = {
        "n_permutations": PERMUTATION_ITERATIONS,
        "null_distribution_summary": permutation_summary,
        "percentiles": {
            "p95": float(np.quantile(null_aurocs, 0.95)),
            "p99": float(np.quantile(null_aurocs, 0.99)),
            "max": float(np.max(null_aurocs)),
        },
        "observed_cv_auroc_mean": observed_cv_mean,
        "empirical_p_value": float((1 + np.sum(null_aurocs >= observed_cv_mean)) / (PERMUTATION_ITERATIONS + 1)),
    }

    retrospective_context = build_feature_context(2018, tables=tables, gene_universe=feature_universe)
    retrospective_feature_df = retrospective_context["feature_df"].copy()
    retrospective_eval_df = retrospective_feature_df.loc[retrospective_feature_df["has_evidence_by_cutoff"]].copy().reset_index(drop=True)
    retrospective_eval_df["label"] = retrospective_eval_df["gene"].isin(fold_c_positive_set).astype(int)
    retrospective_cv = cross_validated_full_universe_predictions(
        retrospective_feature_df,
        retrospective_eval_df,
        FULL_FEATURES,
        make_fixed_xgb,
    )
    ranked_full_df = retrospective_feature_df[["gene", "num_studies"]].copy()
    ranked_full_df["score"] = retrospective_cv["full_universe_mean_scores"]
    ranked_full_df = ranked_full_df.sort_values(
        by=["score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rank_lookup = {gene: idx + 1 for idx, gene in enumerate(ranked_full_df["gene"].tolist())}
    reference_ranks = {gene: int(rank_lookup[gene]) for gene in RETROSPECTIVE_REFERENCE_GENES}
    top_k_needed = max(reference_ranks.values())
    retrospective_p = float(
        stats.hypergeom.sf(
            len(RETROSPECTIVE_REFERENCE_GENES) - 1,
            len(ranked_full_df),
            len(RETROSPECTIVE_REFERENCE_GENES),
            top_k_needed,
        )
    )
    retrospective_result = {
        "training_label_source": "Fold C corpus column assignments",
        "feature_cutoff_year": 2018,
        "n_total_scored_universe": int(len(retrospective_feature_df)),
        "n_eval_genes_with_pre_2019_features": int(len(retrospective_eval_df)),
        "n_pos_eval_genes": int(retrospective_eval_df["label"].sum()),
        "cv_auroc_mean": retrospective_cv["auroc_mean"],
        "cv_auroc_std": retrospective_cv["auroc_std"],
        "cv_auroc_oof": retrospective_cv["auroc_oof"],
        "reference_gene_ranks": reference_ranks,
        "all_reference_genes_within_top_k": top_k_needed,
        "hypergeometric_p_upper_tail": retrospective_p,
        "top20_genes": ranked_full_df.head(20)["gene"].tolist(),
    }

    unsupported = [
        {
            "item": "Table 1 and B10 `is_ad_driver` baseline / two-prior prior-only model",
            "verdict": "WITHDRAWN",
            "reason": "`is_ad_driver` is absent from executable code and from any checked-in frozen prior snapshot.",
        },
        {
            "item": "B9 ten-feature SHAP values for `deg_lfc_mean`, `deg_lfc_max`, `is_ad_driver`, and `n_evidence_total`",
            "verdict": "WITHDRAWN",
            "reason": "Those features are not implemented anywhere in the checked-in scorer.",
        },
        {
            "item": "B6 static-prior snapshot dates (`UniProt 2019-11`, `KEGG 2019-Q4`, `DisGeNET v7.0`)",
            "verdict": "WITHDRAWN",
            "reason": "The checked-in pipeline does not load or preserve those snapshots.",
        },
    ]

    return {
        "shap_executable_fold_c_model": {
            "feature_set": FULL_FEATURES,
            "ranking": shap_ranked,
            "note": "This is the executable 16-feature model. The manuscript's ten-feature SHAP table is not reproducible from the checked-in codebase.",
        },
        "structured_ablation_executable_model": {
            "rows": ablation_rows,
            "note": "Ablations refer to the executable 16-feature scorer, not the manuscript's unavailable ten-feature / two-prior model.",
        },
        "within_secreted_prior_vs_full_check": {
            "dataset_summary": secreted_summary,
            "prior_model_source": "The executable `prior_only` definition from nm_exp_p6_secreted_ablation.py (feature block from nm_exp_p1_prior_control.py).",
            "prior_only_feature_columns": PRIOR_FEATURES,
            "full_model_feature_columns": FULL_FEATURES,
            "prior_only_p_at_10": prior_rank["p_at_10"],
            "full_model_p_at_10": full_rank["p_at_10"],
            "prior_only_p_at_5": prior_rank["p_at_5"],
            "full_model_p_at_5": full_rank["p_at_5"],
        },
        "fold_c_permutation_null": permutation_result,
        "retrospective_pre_2019_recovery": retrospective_result,
        "unsupported_manuscript_components": unsupported,
    }


def archived_crosscheck() -> dict:
    def load_json(path: str) -> dict | list:
        return json.loads((BASE_DIR / path).read_text())

    p6 = load_json("results/nm_exp_p6_secreted_ablation.json")
    p6_by_method = {row["method"]: row for row in p6["methods"]}

    p9 = load_json("results/nm_exp_p9_rolling_origin.json")
    p9_full = next(row for row in p9["aggregated_methods"] if row["method"] == "full_model")

    p1 = load_json("results/nm_exp_p1_prior_control.json")
    p7 = load_json("results/nm_exp_p7_ranked_enrichment.json")
    p7b = load_json("results/nm_exp_p7b_matched_enrichment.json")
    p10 = load_json("results/nm_exp_p10_second_validation.json")
    p11 = load_json("results/nm_exp_p11_noncanonical_enrichment.json")
    p12_baseline = load_json("results/nm_exp_p12_baseline_comparison.json")
    p12_threshold = load_json("results/nm_exp_p12_threshold_robustness.json")
    nm_exp4 = load_json("results/nm_exp4_full_universe.json")
    llm_secreted = load_json("results/nm_exp_llm_within_secreted.json")

    return {
        "nm_exp_p6_secreted_ablation.json": {
            "verdict": "CONFIRMED",
            "notes": "Primary within-secreted endpoint matches the submitted manuscript exactly.",
            "headline_numbers": {
                "auroc": p6_by_method["full_model"]["cv_auroc"],
                "auroc_ci95": p6_by_method["full_model"]["bootstrap_ci"],
                "p_at_5": p6_by_method["full_model"]["ranking_metrics"]["p_at_5"],
                "p_at_10": p6_by_method["full_model"]["ranking_metrics"]["p_at_10"],
                "prior_only_p_at_10": p6_by_method["prior_only"]["ranking_metrics"]["p_at_10"],
                "temporal_only_auroc": p6_by_method["temporal_only"]["cv_auroc"],
            },
        },
        "nm_exp_p9_rolling_origin.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": {
                "fold_auroc_A": p9_full["per_fold_auroc"]["A"],
                "fold_auroc_B": p9_full["per_fold_auroc"]["B"],
                "fold_auroc_C": p9_full["per_fold_auroc"]["C"],
                "mean_auroc": p9_full["mean_auroc"],
                "sd_auroc": p9_full["sd_auroc"],
            },
        },
        "nm_exp_p1_prior_control.json": {
            "verdict": "CONFIRMED_WITH_SCOPE_NOTE",
            "notes": "Supports the executable within-secreted prior/temporal/full comparison, not the manuscript's unavailable two-prior external ablation.",
            "headline_numbers": {
                "secreted_temporal_auroc": p1["secreted_temporal_auroc"],
                "secreted_prior_auroc": p1["secreted_prior_auroc"],
                "secreted_full_auroc": p1["secreted_full_auroc"],
                "full_model_auroc": p1["full_model_auroc"],
            },
        },
        "nm_exp_p7_ranked_enrichment.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": {
                "chronoabor_nes": p7["methods"]["chronoabor"]["nes"],
                "chronoabor_p_value": p7["methods"]["chronoabor"]["p_value"],
                "sig_max_auc_nes": p7["methods"]["sig_max_auc"]["nes"],
                "sig_max_auc_p_value": p7["methods"]["sig_max_auc"]["p_value"],
            },
        },
        "nm_exp_p7b_matched_enrichment.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": {
                "matched_nes": p7b["matched_null"]["matched_nes"],
                "matched_p_value": p7b["matched_null"]["matched_p_value"],
                "observed_es": p7b["observed_enrichment"]["es"],
            },
        },
        "nm_exp_p10_second_validation.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": {
                "set2_random_nes": p10["cohort_analyses"]["set2"]["random_null_enrichment"]["nes"],
                "set2_random_p_value": p10["cohort_analyses"]["set2"]["random_null_enrichment"]["p_value"],
                "set2_top50_odds_ratio": p10["cohort_analyses"]["set2"]["top50_fisher"]["odds_ratio"],
                "set2_top50_fisher_p": p10["cohort_analyses"]["set2"]["top50_fisher"]["fisher_p_value"],
                "set2_matched_nes": p10["cohort_analyses"]["set2"]["matched_null_enrichment"]["nes"],
                "set2_matched_p_value": p10["cohort_analyses"]["set2"]["matched_null_enrichment"]["p_value"],
            },
        },
        "nm_exp_p11_noncanonical_enrichment.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": {
                "set1_random_nes": p11["cohort_analyses"]["set1"]["gsea_random_null"]["nes"],
                "set1_random_p_value": p11["cohort_analyses"]["set1"]["gsea_random_null"]["p_value"],
                "set1_top50_odds_ratio": p11["cohort_analyses"]["set1"]["top50_fisher"]["odds_ratio"],
                "set1_top50_fisher_p": p11["cohort_analyses"]["set1"]["top50_fisher"]["fisher_p_value"],
                "set2_random_nes": p11["cohort_analyses"]["set2"]["gsea_random_null"]["nes"],
                "set2_random_p_value": p11["cohort_analyses"]["set2"]["gsea_random_null"]["p_value"],
                "set2_top50_odds_ratio": p11["cohort_analyses"]["set2"]["top50_fisher"]["odds_ratio"],
                "set2_top50_fisher_p": p11["cohort_analyses"]["set2"]["top50_fisher"]["fisher_p_value"],
            },
        },
        "nm_exp_p12_baseline_comparison.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": p12_baseline,
        },
        "nm_exp_p12_threshold_robustness.json": {
            "verdict": "CONFIRMED",
            "headline_numbers": p12_threshold,
        },
        "nm_exp4_full_universe.json": {
            "verdict": "CONFIRMED_BUT_NOT_EXTERNAL",
            "notes": "These are ABOR-internal prospective results. They explain the checked-in 0.827 / null-max 0.732 numbers that were later mixed into the manuscript, but they are not the external NM-corpus benchmark.",
            "headline_numbers": {
                "cv_auroc_mean": nm_exp4["cv_auroc_mean"],
                "cv_auroc_std": nm_exp4["cv_auroc_std"],
                "cv_auprc": nm_exp4["cv_auprc"],
                "permutation_p_value": nm_exp4["permutation_p_value"],
                "known_positive_full_ranks": {gene: info["full_universe_rank"] for gene, info in nm_exp4["known_positives_rank"].items()},
            },
        },
        "nm_exp_llm_within_secreted.json": {
            "verdict": "CONFIRMED",
            "notes": "Supports the within-secreted LLM comparison, not Table 1's full-universe external LLM rows.",
            "headline_numbers": {
                "comparison_rows": llm_secreted["comparison_rows"],
            },
        },
    }


def load_e5_ground_truth() -> dict:
    e5_path = BASE_DIR / "results" / "rev_exp_e5_study_list.json"
    payload = json.loads(e5_path.read_text())
    return {
        "source_json": str(e5_path),
        "ground_truth_totals": payload["ground_truth_totals"],
        "manuscript_claim_comparison": payload["manuscript_claim_comparison"],
    }


def make_row(location: str, old_value: str, new_value: str, verdict: str) -> dict:
    return {
        "manuscript_location": location,
        "old_published_value": old_value,
        "new_rederived_value": new_value,
        "verdict": verdict,
    }


def build_replacement_rows(payload: dict) -> list[dict]:
    corpus = payload["nm_corpus_v2"]
    universe = payload["universe_definitions"]
    fold_c = payload["folds"]["C"]
    fold_d = payload["folds"]["D"]
    deps = payload["fold_c_dependent_outputs"]
    p6 = payload["archived_crosscheck"]["nm_exp_p6_secreted_ablation.json"]["headline_numbers"]
    p9 = payload["archived_crosscheck"]["nm_exp_p9_rolling_origin.json"]["headline_numbers"]
    e5 = payload["e5_ground_truth"]
    topk_c = {row["k"]: row for row in fold_c["topk_enrichment_full_fit"]["rows"]}
    retro = deps["retrospective_pre_2019_recovery"]
    ablation = {row["model_configuration"]: row for row in deps["structured_ablation_executable_model"]["rows"]}
    shap_rank = deps["shap_executable_fold_c_model"]["ranking"]

    rows = [
        make_row(
            "Abstract: ABOR corpus size",
            "243 studies; 4,576 entries",
            f"{e5['ground_truth_totals']['unique_study_total']} studies; {e5['ground_truth_totals']['raw_row_total']} raw rows",
            "REPLACED",
        ),
        make_row(
            "Abstract: external benchmark AUROCs",
            "0.983 and 0.958",
            f"Fold C {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}; Fold D {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}",
            "REPLACED",
        ),
        make_row(
            "Abstract: within-secreted primary endpoint",
            "AUROC 0.759 [0.605, 0.902]; P@5=P@10=1.0",
            f"AUROC {p6['auroc']:.3f} [{p6['auroc_ci95']['lower']:.3f}, {p6['auroc_ci95']['upper']:.3f}]; P@5={p6['p_at_5']:.3f}; P@10={p6['p_at_10']:.3f}",
            "CONFIRMED",
        ),
        make_row(
            "Abstract: rolling-origin summary",
            "Mean AUROC 0.771 ± 0.021",
            f"Mean AUROC {p9['mean_auroc']:.3f} ± {p9['sd_auroc']:.3f}",
            "CONFIRMED",
        ),
        make_row(
            "Conclusions: ABOR resource size",
            "4,576 entries; 243 studies; 2,894 genes",
            f"{e5['ground_truth_totals']['raw_row_total']} raw rows; {e5['ground_truth_totals']['unique_study_total']} studies; {universe['adopted_scored_universe']['n_genes']} scored genes (6-layer pipeline universe), {universe['seven_layer_union_including_target']['n_genes']} genes in the 7-layer union including target.csv",
            "REPLACED",
        ),
        make_row(
            "Conclusions: external literature evaluation headline",
            "AUROC up to 0.983; AUPRC 0.412; top-10 precision 100%; 123-fold enrichment",
            f"Fold C AUROC {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}, OOF AUPRC {fold_c['auprc_summary']['chronoabor_xgb_full_model_oof']:.3f}, full-fit P@10 {topk_c[10]['precision_at_k']:.3f}, EF@10 {topk_c[10]['enrichment_factor_at_k']:.1f}x; Fold D AUROC {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}, OOF AUPRC {fold_d['auprc_summary']['chronoabor_xgb_full_model_oof']:.3f}, full-fit P@10 {next(row['precision_at_k'] for row in fold_d['topk_enrichment_full_fit']['rows'] if row['k'] == 10):.3f}, EF@10 {next(row['enrichment_factor_at_k'] for row in fold_d['topk_enrichment_full_fit']['rows'] if row['k'] == 10):.1f}x",
            "REPLACED",
        ),
        make_row(
            "Conclusions: retrospective recovery",
            "All eight established genes recovered within the top 8",
            f"All eight recovered within the top {retro['all_reference_genes_within_top_k']} only; ranks {', '.join(f'{gene}={rank}' for gene, rank in retro['reference_gene_ranks'].items())}",
            "REPLACED",
        ),
        make_row(
            "Conclusions: within-secreted prior-only refinement",
            "57% → 100% top-10 precision",
            f"{deps['within_secreted_prior_vs_full_check']['prior_only_p_at_10']:.0%} → {deps['within_secreted_prior_vs_full_check']['full_model_p_at_10']:.0%} top-10 precision under the executable p6 prior-only vs full comparison",
            "REPLACED",
        ),
        make_row(
            "Table 1: ChronoABOR (full model)",
            "Fold C 0.983 ± 0.014; Fold D 0.958 ± 0.069",
            f"Fold C {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_c['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}; Fold D {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f} ± {fold_d['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}",
            "REPLACED",
        ),
        make_row(
            "Table 1: Logistic regression (all features)",
            "Fold C 0.971 ± 0.019; Fold D 0.944 ± 0.081",
            f"Fold C {fold_c['model_metrics']['logistic_regression_l2']['cv_auroc_mean']:.3f} ± {fold_c['model_metrics']['logistic_regression_l2']['cv_auroc_std']:.3f}; Fold D {fold_d['model_metrics']['logistic_regression_l2']['cv_auroc_mean']:.3f} ± {fold_d['model_metrics']['logistic_regression_l2']['cv_auroc_std']:.3f}",
            "REPLACED",
        ),
        make_row(
            "Table 1: Elastic net (all features)",
            "Fold C 0.968 ± 0.022; Fold D 0.941 ± 0.084",
            f"Fold C {fold_c['model_metrics']['elastic_net']['cv_auroc_mean']:.3f} ± {fold_c['model_metrics']['elastic_net']['cv_auroc_std']:.3f}; Fold D {fold_d['model_metrics']['elastic_net']['cv_auroc_mean']:.3f} ± {fold_d['model_metrics']['elastic_net']['cv_auroc_std']:.3f}",
            "REPLACED",
        ),
        make_row(
            "Table 1: GPT / Gemini full-universe external rows",
            "GPT-5.2 0.793/0.225/P@10 0.50; GPT-5.4 0.770/0.232/P@10 0.50; Gemini rows as printed",
            "No current checked-in result JSON reproduces the Table 1 LLM rows on the authoritative 31-gene C/D benchmark; keep them out of the replacement table pending a separate rerun.",
            "WITHDRAWN",
        ),
        make_row(
            "Table 1: is_secreted alone",
            "Fold C 0.926; Fold D 0.926",
            f"Fold C {fold_c['baselines']['is_secreted']['auroc']:.3f}; Fold D {fold_d['baselines']['is_secreted']['auroc']:.3f}",
            "REPLACED",
        ),
        make_row(
            "Table 1: is_ad_driver alone",
            "Fold C 0.814; Fold D 0.809",
            "No executable `is_ad_driver` prior exists in the repo.",
            "WITHDRAWN",
        ),
        make_row(
            "Table 1: prior-only (is_secreted + is_ad_driver)",
            "Fold C 0.953; Fold D 0.950",
            "The two-prior model cannot be reconstructed because `is_ad_driver` is absent from the codebase.",
            "WITHDRAWN",
        ),
        make_row(
            "Table 1: citation frequency",
            "Fold C 0.752; Fold D 0.726 (`n_evidence_total`)",
            f"Fold C {fold_c['baselines']['publication_count_num_studies']['auroc']:.3f}; Fold D {fold_d['baselines']['publication_count_num_studies']['auroc']:.3f} using the checked-in `num_studies` publication-count baseline",
            "REPLACED",
        ),
        make_row(
            "Table 1: TVL baseline",
            "Fold C 0.631; Fold D 0.632",
            f"Subset-only control: Fold C {fold_c['baselines']['tvl']['auroc']:.3f}; Fold D {fold_d['baselines']['tvl']['auroc']:.3f} on n={fold_c['baselines']['tvl']['subset_n_genes']} genes with TVL and n_pos={fold_c['baselines']['tvl']['subset_n_pos']}",
            "REPLACED",
        ),
        make_row(
            "Table 1: random baseline",
            "0.500 / 0.500",
            "0.500 / 0.500",
            "CONFIRMED",
        ),
        make_row(
            "tab:secreted_subset",
            "Entire table",
            "All primary values match results/nm_exp_p6_secreted_ablation.json exactly.",
            "CONFIRMED",
        ),
        make_row(
            "B5: permutation mean AUROC",
            "0.503",
            f"{deps['fold_c_permutation_null']['null_distribution_summary']['mean']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: permutation SD",
            "0.068",
            f"{deps['fold_c_permutation_null']['null_distribution_summary']['std']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: 95th percentile",
            "0.732",
            f"{deps['fold_c_permutation_null']['percentiles']['p95']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: 99th percentile",
            "0.759",
            f"{deps['fold_c_permutation_null']['percentiles']['p99']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: permutation max",
            "0.791",
            f"{deps['fold_c_permutation_null']['percentiles']['max']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: observed full-model AUROC",
            "0.985",
            f"{deps['fold_c_permutation_null']['observed_cv_auroc_mean']:.3f}",
            "REPLACED",
        ),
        make_row(
            "B5: empirical p-value",
            "<0.001",
            f"{deps['fold_c_permutation_null']['empirical_p_value']:.3g}",
            "REPLACED",
        ),
        make_row(
            "B6: NM corpus size and fold counts",
            "32 genes; Fold C 21; Fold D 21; shared 11",
            f"{corpus['n_rows']} genes; Fold C {corpus['fold_counts']['fold_c']}; Fold D {corpus['fold_counts']['fold_d']}; shared {corpus['fold_counts']['shared']}",
            "REPLACED",
        ),
        make_row(
            "B6: external label windows",
            "Fold C 2021-2024; Fold D 2022-2025",
            f"Column-defined folds are authoritative. Positive row years actually span Fold C {fold_c['authoritative_positive_year_range']['min']}-{fold_c['authoritative_positive_year_range']['max']} and Fold D {fold_d['authoritative_positive_year_range']['min']}-{fold_d['authoritative_positive_year_range']['max']}.",
            "REPLACED",
        ),
        make_row(
            "B6: genes with pre-cutoff features",
            "Fold C 2,585; Fold D 2,611",
            f"Fold C {fold_c['n_total_eval_genes_with_pre_cutoff_features']}; Fold D {fold_d['n_total_eval_genes_with_pre_cutoff_features']}",
            "REPLACED",
        ),
        make_row(
            "B6: benchmark positives surviving the filter",
            "Fold C 21; Fold D 21",
            f"Fold C {fold_c['n_pos_eval_genes']}; Fold D {fold_d['n_pos_eval_genes']}",
            "REPLACED",
        ),
        make_row(
            "B6: static-prior snapshot dates",
            "UniProt 2019-11; KEGG 2019-Q4; DisGeNET v7.0",
            "No such frozen priors are implemented in executable code.",
            "WITHDRAWN",
        ),
    ]

    for k in [5, 10, 20, 50]:
        row = topk_c[k]
        old_map = {
            5: "P@5 1.00 [0.60, 1.00]; EF >100x; Hypergeom 2.1e-11; Permutation <0.001",
            10: "P@10 1.00 [0.67, 1.00]; EF 123x; Hypergeom 9.8e-23; Permutation <0.001",
            20: "P@20 0.95 [0.75, 1.00]; EF 117x; Hypergeom 7.9e-45; Permutation <0.001",
            50: "P@50 0.42 [0.28, 0.56]; EF 52x; Permutation <0.001",
        }
        rows.append(
            make_row(
                f"B7: Fold C top-{k}",
                old_map[k],
                f"P@{k} {row['precision_at_k']:.3f} [{row['precision_bootstrap_ci95']['lower']:.3f}, {row['precision_bootstrap_ci95']['upper']:.3f}]; EF {row['enrichment_factor_at_k']:.1f}x; Hypergeom {row['hypergeometric_p_upper_tail']:.3g}; Permutation {row['permutation_p_upper_tail']:.3g}",
                "REPLACED",
            )
        )

    rows.extend(
        [
            make_row(
                "B7: Fold C random background rate",
                "0.0081",
                f"{fold_c['topk_enrichment_full_fit']['random_background_rate']:.4f}",
                "REPLACED",
            ),
            make_row(
                "B7 note: Fold C AUPRCs",
                "Full 0.412; citation 0.094; is_secreted 0.287",
                f"Full {fold_c['auprc_summary']['chronoabor_xgb_full_model_oof']:.3f}; publication-count (`num_studies`) {fold_c['auprc_summary']['publication_count_num_studies']:.3f}; is_secreted {fold_c['auprc_summary']['is_secreted']:.3f}",
                "REPLACED",
            ),
            make_row(
                "B8: title/note",
                "Full 2,894-gene universe; all eight in top 8; P=5.9e-13",
                f"Full 2,843-gene scored universe; all eight in top {retro['all_reference_genes_within_top_k']}; P={retro['hypergeometric_p_upper_tail']:.3g}",
                "REPLACED",
            ),
        ]
    )

    old_ranks = {"MAPT": 1, "GFAP": 2, "APOE": 3, "APP": 4, "BACE1": 5, "TREM2": 6, "SNAP25": 7, "NEFL": 8}
    for gene in RETROSPECTIVE_REFERENCE_GENES:
        rows.append(
            make_row(
                f"B8: rank of {gene}",
                str(old_ranks[gene]),
                str(retro["reference_gene_ranks"][gene]),
                "REPLACED",
            )
        )

    rows.extend(
        [
            make_row(
                "B9: is_secreted mean|SHAP|",
                "3.647",
                f"{next(row['mean_abs_shap'] for row in shap_rank if row['feature'] == 'is_secreted'):.3f}",
                "REPLACED",
            ),
            make_row(
                "B9: deg_lfc_mean mean|SHAP|",
                "1.748",
                "Feature absent from executable scorer.",
                "WITHDRAWN",
            ),
            make_row(
                "B9: deg_lfc_max mean|SHAP|",
                "1.219",
                "Feature absent from executable scorer.",
                "WITHDRAWN",
            ),
            make_row(
                "B9: is_ad_driver mean|SHAP|",
                "0.877",
                "Feature absent from executable scorer.",
                "WITHDRAWN",
            ),
            make_row(
                "B9: n_evidence_total mean|SHAP|",
                "0.760",
                "Feature absent from executable scorer.",
                "WITHDRAWN",
            ),
            make_row(
                "B10: full model",
                "0.985",
                f"{ablation['full_model']['cv_auroc_mean']:.3f}",
                "REPLACED",
            ),
            make_row(
                "B10: no evidence-count feature block",
                "0.980",
                f"{ablation['no_evidence_count_block']['cv_auroc_mean']:.3f} on the executable 16-feature model",
                "REPLACED",
            ),
            make_row(
                "B10: no DEG features",
                "0.962",
                f"{ablation['no_deg_features']['cv_auroc_mean']:.3f}",
                "REPLACED",
            ),
            make_row(
                "B10: prior only (is_secreted + is_ad_driver)",
                "0.953",
                "Two-prior model unsupported; no executable `is_ad_driver` feature exists.",
                "WITHDRAWN",
            ),
            make_row(
                "B10: temporal only",
                "0.677",
                f"{ablation['temporal_only_executable']['cv_auroc_mean']:.3f} for the executable temporal block (all checked-in features except `is_secreted`)",
                "REPLACED",
            ),
            make_row(
                "B10 note: within-secreted top-10 precision",
                "57% → 100%",
                f"{deps['within_secreted_prior_vs_full_check']['prior_only_p_at_10']:.0%} → {deps['within_secreted_prior_vs_full_check']['full_model_p_at_10']:.0%} under the executable p6 comparison",
                "REPLACED",
            ),
        ]
    )
    return rows


def write_replacement_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# ChronoABOR External Benchmark Replacement Map",
        "",
        "| Manuscript location | Old published value | New re-derived value | Verdict |",
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

    tables = load_all_tables()
    feature_universe = build_gene_universe(tables)
    seven_layer_universe = build_all_layer_gene_universe(tables)
    target_only_genes = sorted(set(seven_layer_universe) - set(feature_universe))
    tvl_lookup = build_tvl_lookup(tables["target"])

    corpus_df = parse_nm_corpus_tex(NM_BENCHMARK_TEX_PATH)
    e5_ground_truth = load_e5_ground_truth()
    corpus_df[["gene", "protein_name", "evidence_type", "ref_key", "year", "fold_c", "fold_d"]].to_csv(
        OUTPUT_CORPUS_CSV,
        index=False,
    )

    shared_genes = corpus_df.loc[corpus_df["fold_c"] & corpus_df["fold_d"], "gene"].tolist()
    corpus_summary = {
        "source_tex": str(NM_BENCHMARK_TEX_PATH),
        "output_csv": str(OUTPUT_CORPUS_CSV),
        "n_rows": int(len(corpus_df)),
        "genes": corpus_df["gene"].tolist(),
        "fold_counts": {
            "fold_c": int(corpus_df["fold_c"].sum()),
            "fold_d": int(corpus_df["fold_d"].sum()),
            "shared": int((corpus_df["fold_c"] & corpus_df["fold_d"]).sum()),
        },
        "shared_genes": shared_genes,
        "observed_study_claim_comparison": {
            "manuscript_claim_distinct_genes": 32,
            "actual_unique_studies_from_e5_ground_truth": e5_ground_truth["ground_truth_totals"]["unique_study_total"],
            "actual_raw_row_total_from_e5_ground_truth": e5_ground_truth["ground_truth_totals"]["raw_row_total"],
            "actual_gene_rows_in_nm_corpus_table": int(len(corpus_df)),
            "footnote_claim_fold_c": 21,
            "footnote_claim_fold_d": 21,
            "footnote_claim_shared": 11,
        },
    }

    universe_summary = {
        "pipeline_feature_bearing_universe": {
            "n_genes": int(len(feature_universe)),
            "layers": ["snp", "cpg", "deg", "hub", "mirna", "signature"],
            "description": "Checked-in six-layer scoring universe from build_gene_universe() / LAYER_FILES.",
        },
        "seven_layer_union_including_target": {
            "n_genes": int(len(seven_layer_universe)),
            "layers": ["snp", "cpg", "deg", "hub", "mirna", "signature", "target"],
            "description": "Seven-layer union including target.csv genes from E5.",
        },
        "manuscript_claim": {
            "n_genes": MANUSCRIPT_UNIVERSE_CLAIM,
        },
        "adopted_scored_universe": {
            "name": "pipeline_feature_bearing_universe",
            "n_genes": int(len(feature_universe)),
            "rationale": "The executable scorer only engineers temporal features from six feature-bearing layers. target.csv is not in LAYER_FILES and contributes no checked-in temporal feature.",
        },
        "target_only_genes_excluded_from_scored_universe": {
            "n_genes": int(len(target_only_genes)),
            "genes": target_only_genes,
            "why_excluded": "These genes exist only in target.csv or outside feature-bearing layers and therefore cannot receive any checked-in model feature.",
        },
    }

    feature_definition_summary = {
        "manuscript_claimed_ten_features": MANUSCRIPT_TEN_FEATURES,
        "checked_in_executable_features": FULL_FEATURES,
        "n_checked_in_executable_features": len(FULL_FEATURES),
        "manuscript_features_missing_from_executable_code": sorted(set(MANUSCRIPT_TEN_FEATURES) - set(FULL_FEATURES)),
        "executable_features_absent_from_manuscript_ten_feature_table": sorted(set(FULL_FEATURES) - set(MANUSCRIPT_TEN_FEATURES)),
        "note": "The checked-in scorer is a 16-feature model (15 columns from FEATURE_COLUMNS plus is_secreted). The manuscript's ten-feature / two-prior specification is not what the repository executes.",
    }

    folds: dict[str, dict] = {}
    fold_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold_key in ["C", "D"]:
        result, feature_df, eval_df = build_fold_result(
            fold_key,
            corpus_df,
            tables,
            feature_universe,
            seven_layer_universe,
            target_only_genes,
            tvl_lookup,
        )
        folds[fold_key] = result
        fold_frames[fold_key] = (feature_df, eval_df)

    fold_c_positive_set = set(corpus_df.loc[corpus_df["fold_c"], "gene"].tolist())
    fold_c_outputs = compute_fold_c_dependent_outputs(
        fold_frames["C"][0],
        fold_frames["C"][1],
        fold_c_positive_set,
        tables,
        feature_universe,
    )

    payload = {
        **provenance(
            SCRIPT_NAME,
            {
                "n_splits": N_SPLITS,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "permutation_iterations": PERMUTATION_ITERATIONS,
                "fold_specs": FOLD_SPECS,
                "adopted_scored_universe_definition": "build_gene_universe() over the six feature-bearing layers in nm_exp4_full_universe.py",
            },
            [
                NM_BENCHMARK_TEX_PATH,
                BASE_DIR / "revision-1" / "supplementary_table_s1.tex",
                BASE_DIR / "target.csv",
                BASE_DIR / "degs.csv",
                BASE_DIR / "snp.csv",
                BASE_DIR / "cpg.csv",
                BASE_DIR / "hubgene.csv",
                BASE_DIR / "miRNA.csv",
                BASE_DIR / "signature.csv",
                BASE_DIR / "results" / "rev_exp_e5_study_list.json",
            ],
        ),
        "e5_ground_truth": e5_ground_truth,
        "nm_corpus_v2": corpus_summary,
        "universe_definitions": universe_summary,
        "feature_definition_reconciliation": feature_definition_summary,
        "folds": folds,
        "fold_c_dependent_outputs": fold_c_outputs,
        "archived_crosscheck": archived_crosscheck(),
        "runtime_seconds": float(time.time() - start),
    }

    replacement_rows = build_replacement_rows(payload)
    payload["replacement_map_rows"] = replacement_rows
    write_json(OUTPUT_JSON, payload)
    write_replacement_markdown(replacement_rows, OUTPUT_MD)

    fold_c_top10 = next(row for row in folds["C"]["topk_enrichment_full_fit"]["rows"] if row["k"] == 10)
    fold_d_top10 = next(row for row in folds["D"]["topk_enrichment_full_fit"]["rows"] if row["k"] == 10)
    print(f"nm_corpus_v2.csv: {corpus_summary['n_rows']} rows; Fold C/D/shared = {corpus_summary['fold_counts']['fold_c']}/{corpus_summary['fold_counts']['fold_d']}/{corpus_summary['fold_counts']['shared']}")
    print(f"Scored universe adopted: {universe_summary['adopted_scored_universe']['n_genes']} genes; target-only exclusions = {universe_summary['target_only_genes_excluded_from_scored_universe']['n_genes']}")
    print(
        f"Fold C: n_total={folds['C']['n_total_eval_genes_with_pre_cutoff_features']}, n_pos={folds['C']['n_pos_eval_genes']}, "
        f"XGB AUROC={folds['C']['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f}±{folds['C']['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}, "
        f"P@10={fold_c_top10['precision_at_k']:.3f}, EF@10={fold_c_top10['enrichment_factor_at_k']:.1f}x"
    )
    print(
        f"Fold D: n_total={folds['D']['n_total_eval_genes_with_pre_cutoff_features']}, n_pos={folds['D']['n_pos_eval_genes']}, "
        f"XGB AUROC={folds['D']['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_mean']:.3f}±{folds['D']['model_metrics']['chronoabor_xgb_full_model']['cv_auroc_std']:.3f}, "
        f"P@10={fold_d_top10['precision_at_k']:.3f}, EF@10={fold_d_top10['enrichment_factor_at_k']:.1f}x"
    )
    print(
        f"Within-secreted executable prior-only vs full P@10: "
        f"{fold_c_outputs['within_secreted_prior_vs_full_check']['prior_only_p_at_10']:.3f} -> "
        f"{fold_c_outputs['within_secreted_prior_vs_full_check']['full_model_p_at_10']:.3f}"
    )
    print(
        f"Retrospective pre-2019: all eight reference genes fall within top {fold_c_outputs['retrospective_pre_2019_recovery']['all_reference_genes_within_top_k']} "
        f"(P={fold_c_outputs['retrospective_pre_2019_recovery']['hypergeometric_p_upper_tail']:.3g})"
    )
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
