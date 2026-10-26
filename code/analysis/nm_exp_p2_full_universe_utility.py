#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from nm_exp4_full_universe import (
    FEATURE_COLUMNS as ABOR_FEATURE_COLUMNS,
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    is_peripheral_tissue,
    load_tables,
    normalize_gene_tokens,
    resolve_year,
    to_jsonable,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p2_full_universe_utility.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp_p2_utility_curves.png"
TARGET_CSV = BASE_DIR / "target.csv"

RANDOM_SEED = 42
FEATURE_CUTOFF_YEAR = 2021
TOP_K_VALUES = [5, 10, 20, 50, 100]
NM_CORPUS = {
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
}
SYMPTOMATIC_DRUG_TERMS = {
    "donepezil",
    "memantine",
    "rivastigmine",
    "galantamine",
}
SPECIAL_TARGET_GENE_SPLITS = {
    "ACHE/GRIN2B": ["ACHE", "GRIN2B"],
    "HCRTR1/HCRTR2": ["HCRTR1", "HCRTR2"],
    "C1QA/B/C": ["C1QA", "C1QB", "C1QC"],
    "PRKAA1/2": ["PRKAA1", "PRKAA2"],
    "DNMT1/3A/3B": ["DNMT1", "DNMT3A", "DNMT3B"],
    "NR1H2/3": ["NR1H2", "NR1H3"],
    "IL12B/IL12RB2": ["IL12B", "IL12RB2"],
    "OLIG2/MBP/SOX10": ["OLIG2", "MBP", "SOX10"],
    "VEGFC/FLT4": ["VEGFC", "FLT4"],
    "GABBR1/GABRA1": ["GABBR1", "GABRA1"],
}
FEATURE_COLUMNS = ABOR_FEATURE_COLUMNS + ["is_secreted"]


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        matched = lower_to_original.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def coerce_binary_label(raw_value: object) -> int | None:
    if pd.isna(raw_value):
        return None
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        numeric = float(raw_value)
        if numeric in (0.0, 1.0):
            return int(numeric)
        return None

    text = str(raw_value).strip().lower()
    if not text:
        return None
    if text in {"1", "1.0", "true", "yes", "positive", "pos"}:
        return 1
    if text in {"0", "0.0", "false", "no", "negative", "neg"}:
        return 0
    return None


def normalize_target_gene_tokens(raw_value: object, known_genes: set[str]) -> list[str]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip().upper()
    if not text or text == "MULTIPLE":
        return []
    if text in SPECIAL_TARGET_GENE_SPLITS:
        return [gene for gene in SPECIAL_TARGET_GENE_SPLITS[text] if gene in known_genes]
    return [gene for gene in normalize_gene_tokens(text) if gene in known_genes]


def derive_row_label_from_target(row: pd.Series, status_column: str, drug_name_column: str) -> int | None:
    drug_status = str(row.get(status_column, "")).strip().lower()
    drug_name = str(row.get(drug_name_column, "")).strip().lower()
    symptomatic = any(term in drug_name for term in SYMPTOMATIC_DRUG_TERMS)

    is_positive = (
        ("fda-approved" in drug_status) or (drug_status in {"phase 3", "phase 2/3"})
    ) and not symptomatic
    is_negative = (
        ("preclinical" in drug_status) or (drug_status == "phase 1") or ("discontinued" in drug_status)
    ) and not is_positive

    if is_positive:
        return 1
    if is_negative:
        return 0
    return None


def load_internal_labels(target_path: Path, known_genes: set[str]) -> tuple[pd.DataFrame, dict]:
    target_df = pd.read_csv(target_path)
    gene_column = find_column(target_df, ["Gene_symbol", "GeneSymbol", "gene", "gene_symbol", "symbol"])
    if gene_column is None:
        raise RuntimeError(f"Could not find gene column in {target_path.name}. Columns: {list(target_df.columns)}")

    explicit_label_column = None
    for column in target_df.columns:
        if str(column).lower() == "label":
            explicit_label_column = str(column)
            break

    status_column = None
    drug_name_column = None
    label_source = "explicit_label_column"
    if explicit_label_column is None:
        status_column = find_column(target_df, ["DrugStatus", "drug_status"])
        drug_name_column = find_column(target_df, ["DrugName", "drug_name"])
        if status_column is None or drug_name_column is None:
            raise RuntimeError(
                f"No explicit label column found in {target_path.name}, and fallback DrugStatus/DrugName columns are missing."
            )
        label_source = "derived_from_drugstatus"

    gene_to_labels: dict[str, set[int]] = {}
    mapped_rows = 0
    unlabeled_rows = 0
    rows_without_mapped_genes = 0
    for _, row in target_df.iterrows():
        if explicit_label_column is not None:
            label = coerce_binary_label(row.get(explicit_label_column))
        else:
            label = derive_row_label_from_target(row, status_column=status_column, drug_name_column=drug_name_column)

        if label is None:
            unlabeled_rows += 1
            continue

        genes = normalize_target_gene_tokens(row.get(gene_column), known_genes)
        if not genes:
            rows_without_mapped_genes += 1
            continue

        mapped_rows += 1
        for gene in genes:
            gene_to_labels.setdefault(gene, set()).add(label)

    rows = []
    n_conflicts = 0
    for gene in sorted(gene_to_labels):
        labels = gene_to_labels[gene]
        if 1 in labels and 0 in labels:
            n_conflicts += 1
        resolved_label = 1 if 1 in labels else 0
        rows.append(
            {
                "gene": gene,
                "label": resolved_label,
                "label_values_seen": sorted(labels),
            }
        )

    labels_df = pd.DataFrame(rows, columns=["gene", "label", "label_values_seen"])
    summary = {
        "source_file": str(target_path),
        "target_columns": list(target_df.columns),
        "gene_column": gene_column,
        "label_column": explicit_label_column,
        "label_source": label_source,
        "rows_in_target_csv": int(len(target_df)),
        "rows_with_mapped_labeled_genes": int(mapped_rows),
        "rows_without_explicit_or_derived_label": int(unlabeled_rows),
        "rows_with_label_but_no_genes_in_universe": int(rows_without_mapped_genes),
        "n_labeled_genes_in_universe": int(len(labels_df)),
        "n_positive_genes_in_universe": int((labels_df["label"] == 1).sum()) if not labels_df.empty else 0,
        "n_negative_genes_in_universe": int((labels_df["label"] == 0).sum()) if not labels_df.empty else 0,
        "n_genes_with_conflicting_labels_positive_overrides": int(n_conflicts),
    }
    return labels_df, summary


def build_is_secreted_set(signature_df: pd.DataFrame, pmid_year_map: dict[str, int], cutoff_year: int) -> set[str]:
    secreted_genes: set[str] = set()
    for _, row in signature_df.iterrows():
        direct_year = int(row["Year_int"]) if pd.notna(row.get("Year_int")) else None
        year, _ = resolve_year(row.get("PMID"), pmid_year_map, direct_year=direct_year)
        if year > cutoff_year or not is_peripheral_tissue(row.get("Tissue")):
            continue
        secreted_genes.update(normalize_gene_tokens(row.get("Gene_symbol")))
    return secreted_genes


def rank_by_score(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    sort_columns = [score_column]
    ascending = [False]
    if score_column != "num_studies":
        sort_columns.append("num_studies")
        ascending.append(False)
    sort_columns.append("gene")
    ascending.append(True)
    return df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)


def compute_utility_metrics(
    universe_df: pd.DataFrame,
    score_column: str,
    ef_base_rate: float,
) -> tuple[dict, pd.DataFrame]:
    ranked_df = rank_by_score(universe_df, score_column)
    labels = ranked_df["nm_label"].to_numpy(dtype=int)
    scores = ranked_df[score_column].to_numpy(dtype=float)

    metrics = {
        "score_column": score_column,
        "auprc": float(average_precision_score(labels, scores)),
        "p_at_k": {},
        "ef_at_k": {},
        "lift_at_k": {},
        "tp_at_k": {},
        "top_10_genes": ranked_df.head(10)["gene"].tolist(),
    }

    for k in TOP_K_VALUES:
        subset = ranked_df.head(k)
        true_positives = int(subset["nm_label"].sum())
        precision = float(true_positives / k)
        enrichment = float(precision / ef_base_rate) if ef_base_rate > 0 else None
        metrics["p_at_k"][str(k)] = precision
        metrics["ef_at_k"][str(k)] = enrichment
        metrics["lift_at_k"][str(k)] = enrichment
        metrics["tp_at_k"][str(k)] = true_positives

    return metrics, ranked_df


def plot_precision_curves(results_by_method: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    method_styles = [
        ("chronoabor", "ChronoABOR", "#1565c0", "o"),
        ("baseline_num_studies", "Baseline: num_studies", "#ef6c00", "s"),
        ("baseline_sig_max_auc", "Baseline: sig_max_auc", "#2e7d32", "^"),
        ("baseline_layer_count", "Baseline: layer_count", "#6a1b9a", "D"),
    ]

    for key, label, color, marker in method_styles:
        precisions = [results_by_method[key]["p_at_k"][str(k)] for k in TOP_K_VALUES]
        ax.plot(TOP_K_VALUES, precisions, label=label, color=color, marker=marker, linewidth=2)

    ax.set_title("P@k over the full ABOR universe")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision")
    ax.set_xticks(TOP_K_VALUES)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    known_genes = set(gene_universe)
    pmid_year_map, _ = build_signature_pmid_year_map(tables["signature"])

    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )
    is_secreted_genes = build_is_secreted_set(tables["signature"], pmid_year_map, FEATURE_CUTOFF_YEAR)
    feature_df["is_secreted"] = feature_df["gene"].isin(is_secreted_genes).astype(int)
    feature_df[FEATURE_COLUMNS] = feature_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    internal_labels_df, label_summary = load_internal_labels(TARGET_CSV, known_genes)
    full_df = feature_df.merge(internal_labels_df[["gene", "label"]], on="gene", how="left")

    train_df = full_df.loc[full_df["label"].isin([0, 1])].copy().reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("No internally labeled genes overlapped the ABOR universe.")
    train_df["label"] = train_df["label"].astype(int)
    if train_df["label"].nunique() < 2:
        raise RuntimeError("Need at least one positive and one negative internal label to train XGBoost.")

    model = make_xgb()
    X_train = train_df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    model.fit(X_train, y_train)

    X_all = full_df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    full_df["chronoabor_score"] = model.predict_proba(X_all)[:, 1]
    full_df["baseline_num_studies"] = full_df["num_studies"].fillna(0.0).astype(float)
    full_df["baseline_sig_max_auc"] = full_df["sig_max_auc"].fillna(0.0).astype(float)
    full_df["baseline_layer_count"] = full_df["layer_count"].fillna(0.0).astype(float)
    full_df["nm_label"] = full_df["gene"].isin(NM_CORPUS).astype(int)

    nm_present = sorted(known_genes & NM_CORPUS)
    nm_missing = sorted(NM_CORPUS - known_genes)
    requested_ef_base_rate = float(len(NM_CORPUS) / len(full_df))
    evaluable_nm_base_rate = float(full_df["nm_label"].mean())

    chrono_metrics, chrono_ranked_df = compute_utility_metrics(full_df, "chronoabor_score", requested_ef_base_rate)
    baseline_num_studies_metrics, _ = compute_utility_metrics(full_df, "baseline_num_studies", requested_ef_base_rate)
    baseline_sig_max_auc_metrics, _ = compute_utility_metrics(full_df, "baseline_sig_max_auc", requested_ef_base_rate)
    baseline_layer_count_metrics, _ = compute_utility_metrics(full_df, "baseline_layer_count", requested_ef_base_rate)

    results_by_method = {
        "chronoabor": chrono_metrics,
        "baseline_num_studies": baseline_num_studies_metrics,
        "baseline_sig_max_auc": baseline_sig_max_auc_metrics,
        "baseline_layer_count": baseline_layer_count_metrics,
    }
    plot_precision_curves(results_by_method)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "n_genes_universe": int(len(gene_universe)),
        "n_genes_with_any_abor_evidence_by_cutoff": int(len(evidence_by_cutoff)),
        "feature_columns": FEATURE_COLUMNS,
        "internal_label_summary": label_summary,
        "internal_training_summary": {
            "n_labeled_genes_used": int(len(train_df)),
            "n_positive": int(train_df["label"].sum()),
            "n_negative": int((1 - train_df["label"]).sum()),
        },
        "nm_corpus_total_genes": int(len(NM_CORPUS)),
        "nm_corpus_genes_present_in_universe": nm_present,
        "nm_corpus_genes_missing_from_universe": nm_missing,
        "nm_corpus_present_count": int(len(nm_present)),
        "nm_corpus_requested_base_rate_for_ef": requested_ef_base_rate,
        "nm_corpus_evaluable_base_rate_from_labels": evaluable_nm_base_rate,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
        "methods": results_by_method,
        "chronoabor_top_20": chrono_ranked_df.head(20)[
            ["gene", "chronoabor_score", "nm_label", "num_studies", "sig_max_auc", "layer_count", "is_secreted"]
        ].to_dict(orient="records"),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    chrono_p10 = chrono_metrics["p_at_k"]["10"]
    chrono_auprc = chrono_metrics["auprc"]
    chrono_ef10 = chrono_metrics["ef_at_k"]["10"]

    print(f"Universe size: {len(gene_universe)} genes")
    print(f"Internal labeled training genes: {len(train_df)} ({int(train_df['label'].sum())} positive, {int((1 - train_df['label']).sum())} negative)")
    print(f"NM_CORPUS genes present in universe: {len(nm_present)}/{len(NM_CORPUS)}")
    print(f"ChronoABOR P@10: {chrono_p10:.4f}")
    print(f"ChronoABOR AUPRC: {chrono_auprc:.4f}")
    print(f"ChronoABOR EF@10: {chrono_ef10:.4f}")
    print(f"Baseline num_studies P@10: {baseline_num_studies_metrics['p_at_k']['10']:.4f}")
    print(f"Baseline sig_max_auc P@10: {baseline_sig_max_auc_metrics['p_at_k']['10']:.4f}")
    print(f"Baseline layer_count P@10: {baseline_layer_count_metrics['p_at_k']['10']:.4f}")
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Utility plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
