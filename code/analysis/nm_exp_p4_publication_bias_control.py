#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nm_exp4_full_universe import (
    FEATURE_COLUMNS,
    FEATURE_CUTOFF_YEAR,
    LABEL_END_YEAR,
    LABEL_START_YEAR,
    LAYER_FILES,
    RANDOM_SEED,
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    extract_year_from_doi_like,
    load_tables,
    make_xgb,
    normalize_gene_tokens,
    normalize_pmid,
    to_jsonable,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p4_bias_control.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp_p4_bias_plots.png"
TARGET_CSV_PATH = BASE_DIR / "target.csv"
N_SPLITS = 5
FIRST_YEAR_DEFAULT = 2019

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

LAYER_YEAR_SPECS = {
    "snp": ("Gene_symbol", "PMID", False),
    "cpg": ("Gene_symbol", "PMID", False),
    "deg": ("Gene_symbol", "PMID_DOI", False),
    "hub": ("Gene_symbl", "PMID", False),
    "mirna": ("Gene_symbol", "PMID_DOI", False),
    "signature": ("Gene_symbol", "PMID", True),
}


def resolve_year_with_default(
    raw_value: object,
    pmid_year_map: dict[str, int],
    default_year: int,
    direct_year: int | None = None,
) -> int:
    if direct_year is not None:
        return int(direct_year)

    pmid = normalize_pmid(raw_value)
    if pmid is not None and pmid in pmid_year_map:
        return int(pmid_year_map[pmid])

    extracted = extract_year_from_doi_like(raw_value)
    if extracted is not None:
        return int(extracted)

    return int(default_year)


def build_first_year_in_abor(
    tables: dict[str, pd.DataFrame],
    pmid_year_map: dict[str, int],
) -> dict[str, int]:
    first_year: dict[str, int] = {}
    for layer, (gene_column, year_column, is_signature) in LAYER_YEAR_SPECS.items():
        table = tables[layer]
        for _, row in table.iterrows():
            direct_year = None
            if is_signature and pd.notna(row.get("Year_int")):
                direct_year = int(row["Year_int"])
            year = resolve_year_with_default(
                raw_value=row.get(year_column),
                pmid_year_map=pmid_year_map,
                default_year=FIRST_YEAR_DEFAULT,
                direct_year=direct_year,
            )
            for gene in normalize_gene_tokens(row.get(gene_column)):
                first_year[gene] = min(first_year.get(gene, year), year)
    return first_year


def build_target_csv_diagnostic(target_path: Path) -> dict:
    target_df = pd.read_csv(target_path)
    normalized_genes: set[str] = set()
    for raw_gene in target_df["GeneSymbol"]:
        normalized_genes.update(normalize_gene_tokens(raw_gene))

    nm_corpus_set = set(NM_CORPUS_GENES)
    overlap = sorted(nm_corpus_set & normalized_genes)
    return {
        "rows": int(len(target_df)),
        "normalized_unique_genes": int(len(normalized_genes)),
        "nm_corpus_overlap_count": int(len(overlap)),
        "nm_corpus_overlap_genes": overlap,
        "decision": (
            "Do not derive the external 32-gene benchmark from target.csv: the table lacks a clean year field, "
            "overlaps only a small subset of the NM corpus, and mixes historical as well as preclinical targets."
        ),
    }


def assign_external_labels(
    gene_universe: list[str],
    evidence_by_cutoff: set[str],
) -> tuple[pd.DataFrame, dict]:
    universe_set = set(gene_universe)
    positive_requested = sorted(set(NM_CORPUS_GENES))
    positive_in_universe = sorted(set(positive_requested) & universe_set)
    positive_with_cutoff_evidence = sorted(set(positive_in_universe) & evidence_by_cutoff)
    positive_without_cutoff_evidence = sorted(set(positive_in_universe) - evidence_by_cutoff)
    negative_genes = {gene for gene in evidence_by_cutoff if gene not in set(positive_in_universe)}

    rows = []
    for gene in gene_universe:
        if gene in positive_in_universe:
            label = 1
            label_status = "positive" if gene in evidence_by_cutoff else "positive_no_cutoff_evidence"
        elif gene in negative_genes:
            label = 0
            label_status = "negative"
        else:
            label = np.nan
            label_status = "unlabeled"
        rows.append(
            {
                "gene": gene,
                "label": label,
                "label_status": label_status,
                "is_nm_corpus_gene_requested": bool(gene in set(positive_requested)),
            }
        )

    diagnostics = {
        "nm_corpus_genes_requested": positive_requested,
        "n_nm_corpus_genes_requested": int(len(positive_requested)),
        "nm_corpus_genes_in_abor_universe": positive_in_universe,
        "n_nm_corpus_genes_in_abor_universe": int(len(positive_in_universe)),
        "nm_corpus_genes_with_cutoff_evidence": positive_with_cutoff_evidence,
        "n_nm_corpus_genes_with_cutoff_evidence": int(len(positive_with_cutoff_evidence)),
        "nm_corpus_genes_without_cutoff_evidence": positive_without_cutoff_evidence,
    }
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True), diagnostics


def evaluate_with_cv(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    allow_loo_when_small_positive: bool = False,
) -> dict:
    y = eval_df["label"].to_numpy(dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if n_pos < 2 or n_neg < 2:
        return {
            "auroc": None,
            "cv_method": None,
            "n_splits": 0,
            "n_samples": int(len(eval_df)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "fold_aurocs": [],
            "reason": "insufficient positive or negative examples (<2)",
        }

    X = eval_df[feature_columns].to_numpy(dtype=float)
    oof_probs = np.full(len(eval_df), np.nan, dtype=float)
    fold_aurocs: list[float] = []

    if allow_loo_when_small_positive and n_pos < 10:
        splitter = LeaveOneOut()
        split_iterator = splitter.split(X, y)
        cv_method = "leave_one_out"
        n_splits = int(len(eval_df))
    else:
        if min(n_pos, n_neg) < N_SPLITS:
            return {
                "auroc": None,
                "cv_method": None,
                "n_splits": 0,
                "n_samples": int(len(eval_df)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "fold_aurocs": [],
                "reason": f"insufficient class count for {N_SPLITS}-fold stratified CV",
            }
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        split_iterator = splitter.split(X, y)
        cv_method = f"stratified_{N_SPLITS}fold"
        n_splits = N_SPLITS

    for train_idx, test_idx in split_iterator:
        model = make_xgb()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        oof_probs[test_idx] = probs
        if len(np.unique(y[test_idx])) == 2:
            fold_aurocs.append(float(roc_auc_score(y[test_idx], probs)))

    if np.isnan(oof_probs).any():
        raise RuntimeError("Cross-validation failed to assign all held-out probabilities.")

    return {
        "auroc": float(roc_auc_score(y, oof_probs)),
        "cv_method": cv_method,
        "n_splits": int(n_splits),
        "n_samples": int(len(eval_df)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fold_aurocs": fold_aurocs,
        "reason": None,
    }


def build_matched_dataset(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    positives = eval_df.loc[eval_df["label"] == 1].copy().reset_index(drop=True)
    negatives = eval_df.loc[eval_df["label"] == 0].copy().reset_index(drop=True)

    matched_negative_genes: set[str] = set()
    positive_without_match: list[str] = []
    per_positive_match_counts: dict[str, int] = {}

    for _, row in positives.iterrows():
        studies = int(row["num_studies"])
        if studies <= 3:
            lower_bound = studies - 1
            upper_bound = studies + 1
        else:
            lower_bound = studies * 0.70
            upper_bound = studies * 1.30
        mask = negatives["num_studies"].between(lower_bound, upper_bound, inclusive="both")
        matched_genes = negatives.loc[mask, "gene"].tolist()
        if not matched_genes:
            positive_without_match.append(str(row["gene"]))
        matched_negative_genes.update(matched_genes)
        per_positive_match_counts[str(row["gene"])] = int(len(matched_genes))

    matched_df = eval_df.loc[
        (eval_df["label"] == 1) | (eval_df["gene"].isin(matched_negative_genes))
    ].copy().reset_index(drop=True)

    diagnostics = {
        "n_positives": int(len(positives)),
        "n_negatives_total": int(len(negatives)),
        "n_matched_negatives": int(len(matched_negative_genes)),
        "n_positives_without_match": int(len(positive_without_match)),
        "positives_without_match": sorted(positive_without_match),
        "per_positive_match_counts": per_positive_match_counts,
    }
    return matched_df, diagnostics


def residualize_features(eval_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    residual_df = eval_df.copy()
    study_counts = eval_df[["num_studies"]].to_numpy(dtype=float)
    regression_summary: dict[str, dict] = {}

    for feature in feature_columns:
        target = eval_df[feature].to_numpy(dtype=float)
        if np.allclose(target, target[0]):
            predicted = np.full_like(target, fill_value=target[0], dtype=float)
            slope = 0.0
            intercept = float(target[0])
        else:
            model = LinearRegression()
            model.fit(study_counts, target)
            predicted = model.predict(study_counts)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
        residual_df[feature] = target - predicted
        regression_summary[feature] = {
            "slope_vs_num_studies": slope,
            "intercept": intercept,
        }

    return residual_df, regression_summary


def assign_year_cohort(year: int) -> str | None:
    if year <= 2015:
        return "Early"
    if 2016 <= year <= 2019:
        return "Mid"
    if 2020 <= year <= 2021:
        return "Recent"
    return None


def plot_bar_values(ax: plt.Axes, labels: list[str], values: list[float | None], title: str) -> None:
    y_values = [np.nan if value is None else float(value) for value in values]
    x = np.arange(len(labels))
    bars = ax.bar(x, np.nan_to_num(y_values, nan=0.0), color="#3f6c8f", alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, color="#777777")
    ax.axhline(0.7, linestyle="--", linewidth=1.2, color="#c87000")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, y_values):
        if np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.03, "NA", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.03, 0.97),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def save_plots(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_bar_values(
        axes[0, 0],
        ["Baseline", "Matched"],
        [results["baseline_auroc"], results["matched_auroc"]],
        "Matched vs Unmatched AUROC",
    )
    plot_bar_values(
        axes[0, 1],
        ["Secreted", "Non-secreted"],
        [
            results["is_secreted_analysis"]["secreted"]["auroc"],
            results["is_secreted_analysis"]["non_secreted"]["auroc"],
        ],
        "AUROC by is_secreted subgroup",
    )
    plot_bar_values(
        axes[1, 0],
        ["Baseline", "Residualized"],
        [results["baseline_auroc"], results["residualized_auroc"]],
        "Baseline vs Residualized AUROC",
    )
    plot_bar_values(
        axes[1, 1],
        ["Early", "Mid", "Recent"],
        [
            results["year_stratified_analysis"]["Early"]["auroc"],
            results["year_stratified_analysis"]["Mid"]["auroc"],
            results["year_stratified_analysis"]["Recent"]["auroc"],
        ],
        "AUROC by first-year cohort",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def print_summary(results: dict) -> None:
    print("NM Experiment P4: Publication bias control")
    print(f"Feature cutoff year: {FEATURE_CUTOFF_YEAR}")
    print(f"Label window: {LABEL_START_YEAR}-{LABEL_END_YEAR}")
    print(
        "NM corpus requested / in ABOR / with pre-cutoff evidence: "
        f"{results['label_diagnostics']['n_nm_corpus_genes_requested']} / "
        f"{results['label_diagnostics']['n_nm_corpus_genes_in_abor_universe']} / "
        f"{results['label_diagnostics']['n_nm_corpus_genes_with_cutoff_evidence']}"
    )
    print(f"Eval set: {results['n_eval_genes']} genes ({results['n_positives']} positives, {results['n_negatives']} negatives)")
    print(
        "target.csv diagnostic: "
        f"{results['target_csv_diagnostic']['normalized_unique_genes']} normalized genes, "
        f"{results['target_csv_diagnostic']['nm_corpus_overlap_count']} overlap with NM corpus"
    )
    print()
    print(f"Baseline AUROC: {results['baseline_auroc']:.4f}" if results["baseline_auroc"] is not None else "Baseline AUROC: NA")
    print(f"Matched AUROC: {results['matched_auroc']:.4f}" if results["matched_auroc"] is not None else "Matched AUROC: NA")
    print(f"Bias delta: {results['bias_delta']:.4f}" if results["bias_delta"] is not None else "Bias delta: NA")
    print(
        f"Residualized AUROC: {results['residualized_auroc']:.4f}"
        if results["residualized_auroc"] is not None
        else "Residualized AUROC: NA"
    )
    print(
        f"Residual delta: {results['residual_delta']:.4f}"
        if results["residual_delta"] is not None
        else "Residual delta: NA"
    )
    print()
    print("is_secreted subgroups:")
    for key, label in [("secreted", "  is_secreted=1"), ("non_secreted", "  is_secreted=0")]:
        group = results["is_secreted_analysis"][key]
        if group["auroc"] is None:
            print(f"{label}: AUROC=NA ({group['reason']}) [n_pos={group['n_pos']}, n_neg={group['n_neg']}]")
        else:
            print(f"{label}: AUROC={group['auroc']:.4f} [n_pos={group['n_pos']}, n_neg={group['n_neg']}]")
    print()
    print("first_year_in_abor cohorts:")
    for cohort in ["Early", "Mid", "Recent"]:
        group = results["year_stratified_analysis"][cohort]
        if group["auroc"] is None:
            print(f"  {cohort}: AUROC=NA ({group['reason']}) [n_pos={group['n_pos']}, n_neg={group['n_neg']}]")
        else:
            print(f"  {cohort}: AUROC={group['auroc']:.4f} [n_pos={group['n_pos']}, n_neg={group['n_neg']}]")
    print()
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )
    first_year_in_abor = build_first_year_in_abor(tables, pmid_year_map)
    label_df, label_diagnostics = assign_external_labels(gene_universe, evidence_by_cutoff)

    full_df = feature_df.merge(label_df, on="gene", how="left")
    full_df["is_secreted"] = (full_df["sig_count_peripheral"] > 0).astype(int)
    full_df["first_year_in_abor"] = full_df["gene"].map(first_year_in_abor).fillna(FIRST_YEAR_DEFAULT).astype(int)
    full_df["first_year_cohort"] = full_df["first_year_in_abor"].apply(assign_year_cohort)

    eval_df = full_df.loc[full_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["label"].astype(int)

    baseline_metrics = evaluate_with_cv(eval_df, FEATURE_COLUMNS, allow_loo_when_small_positive=False)

    matched_df, matched_diagnostics = build_matched_dataset(eval_df)
    matched_metrics = evaluate_with_cv(matched_df, FEATURE_COLUMNS, allow_loo_when_small_positive=False)

    residual_df, residualization_summary = residualize_features(eval_df, FEATURE_COLUMNS)
    residual_metrics = evaluate_with_cv(residual_df, FEATURE_COLUMNS, allow_loo_when_small_positive=False)

    is_secreted_results: dict[str, dict] = {}
    for subgroup_value, key in [(1, "secreted"), (0, "non_secreted")]:
        subgroup_df = eval_df.loc[eval_df["is_secreted"] == subgroup_value].copy().reset_index(drop=True)
        subgroup_metrics = evaluate_with_cv(
            subgroup_df,
            FEATURE_COLUMNS,
            allow_loo_when_small_positive=True,
        )
        is_secreted_results[key] = subgroup_metrics

    year_stratified_results: dict[str, dict] = {}
    for cohort in ["Early", "Mid", "Recent"]:
        cohort_df = eval_df.loc[eval_df["first_year_cohort"] == cohort].copy().reset_index(drop=True)
        n_pos = int(cohort_df["label"].sum()) if not cohort_df.empty else 0
        n_neg = int(len(cohort_df) - n_pos)
        if n_pos < 5:
            year_stratified_results[cohort] = {
                "auroc": None,
                "cv_method": None,
                "n_splits": 0,
                "n_samples": int(len(cohort_df)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "fold_aurocs": [],
                "reason": "insufficient positives for 5-fold CV (<5)",
            }
            continue
        year_stratified_results[cohort] = evaluate_with_cv(
            cohort_df,
            FEATURE_COLUMNS,
            allow_loo_when_small_positive=False,
        )

    baseline_auroc = baseline_metrics["auroc"]
    matched_auroc = matched_metrics["auroc"]
    residualized_auroc = residual_metrics["auroc"]
    bias_delta = None if baseline_auroc is None or matched_auroc is None else float(baseline_auroc - matched_auroc)
    residual_delta = (
        None if baseline_auroc is None or residualized_auroc is None else float(baseline_auroc - residualized_auroc)
    )

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "label_window": f"{LABEL_START_YEAR}-{LABEL_END_YEAR}",
        "feature_columns": FEATURE_COLUMNS,
        "n_genes_universe": int(len(gene_universe)),
        "n_genes_with_evidence_by_cutoff": int(len(evidence_by_cutoff)),
        "n_eval_genes": int(len(eval_df)),
        "n_positives": int(eval_df["label"].sum()),
        "n_negatives": int(len(eval_df) - int(eval_df["label"].sum())),
        "target_csv_diagnostic": build_target_csv_diagnostic(TARGET_CSV_PATH),
        "label_diagnostics": label_diagnostics,
        "nm_positive_source": "fixed_32_gene_external_nm_corpus_reused_from_repo_benchmark_definition",
        "baseline_auroc": baseline_auroc,
        "matched_auroc": matched_auroc,
        "bias_delta": bias_delta,
        "residualized_auroc": residualized_auroc,
        "residual_delta": residual_delta,
        "baseline_analysis": baseline_metrics,
        "matched_analysis": {
            **matched_metrics,
            **matched_diagnostics,
        },
        "is_secreted_analysis": {
            "secreted": is_secreted_results["secreted"],
            "non_secreted": is_secreted_results["non_secreted"],
            "interpretation_rule": "If is_secreted=0 AUROC > 0.6, the model retains predictive power beyond a secreted-gene prior.",
        },
        "residualization_analysis": {
            **residual_metrics,
            "linear_regression_summary": residualization_summary,
            "interpretation_rule": "If residual_delta < 0.05, performance is not strongly dependent on publication count.",
        },
        "year_stratified_analysis": year_stratified_results,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
        "signature_pmid_lookup_size": int(len(pmid_year_map)),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "first_year_in_abor_summary": {
            "default_year_when_unknown": FIRST_YEAR_DEFAULT,
            "cohort_counts_all_genes": {
                "Early": int((full_df["first_year_cohort"] == "Early").sum()),
                "Mid": int((full_df["first_year_cohort"] == "Mid").sum()),
                "Recent": int((full_df["first_year_cohort"] == "Recent").sum()),
                "Excluded_after_2021": int(full_df["first_year_cohort"].isna().sum()),
            },
        },
        "runtime_seconds": float(time.time() - start_time),
    }

    save_plots(results)
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))
    print_summary(results)


if __name__ == "__main__":
    main()
