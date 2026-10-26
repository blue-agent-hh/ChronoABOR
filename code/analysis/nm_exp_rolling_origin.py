#!/usr/bin/env python3

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SIGNATURE_PATH = BASE_DIR / "signature.csv"
OUTPUT_JSON = RESULTS_DIR / "nm_exp2_rolling_origin.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp2_plot.png"
RANDOM_SEED = 42
VALID_GENE_RE = re.compile(r"^[A-Z0-9-]+$")
INVALID_GENE_TOKENS = {"", "NA", "N/A", "NAN", "NONE", "NULL"}
MAIN_CUTOFF_YEARS = [2019, 2020, 2021, 2022]
POSITIVE_LABEL_YEARS = list(range(2020, 2026))
PERMUTATION_COUNT = 1000


def normalize_gene_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    genes = []
    for part in re.split(r"[,;/]", str(raw_value).upper()):
        gene = part.strip()
        if gene in INVALID_GENE_TOKENS:
            continue
        if not VALID_GENE_RE.fullmatch(gene):
            continue
        genes.append(gene)
    return sorted(set(genes))


def load_exploded_signature(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"ABOR_ID", "Year", "Tissue", "AUC", "Gene_symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"signature.csv is missing required columns: {sorted(missing)}")

    df["Year_int"] = pd.to_numeric(df["Year"], errors="coerce")
    df["AUC_num"] = pd.to_numeric(df["AUC"], errors="coerce")
    df["genes"] = df["Gene_symbol"].apply(normalize_gene_tokens)
    df = df.loc[df["Year_int"].notna()].copy()
    df["Year_int"] = df["Year_int"].astype(int)

    exploded_rows = []
    for _, row in df.iterrows():
        for gene in row["genes"]:
            exploded_rows.append(
                {
                    "ABOR_ID": row["ABOR_ID"],
                    "gene": gene,
                    "Year_int": int(row["Year_int"]),
                    "Tissue": str(row["Tissue"]),
                    "AUC_num": float(row["AUC_num"]) if pd.notna(row["AUC_num"]) else np.nan,
                }
            )
    exploded = pd.DataFrame(exploded_rows)
    return df, exploded


def peripheral_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.contains("plasma|blood|serum", case=False, regex=True)


def brain_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.contains("brain", case=False, regex=False)


def build_first_peripheral_year_map(exploded: pd.DataFrame) -> dict[str, int]:
    mask = peripheral_mask(exploded["Tissue"]) & exploded["AUC_num"].ge(0.88)
    filtered = exploded.loc[mask].copy()
    first_year = filtered.groupby("gene")["Year_int"].min().astype(int).to_dict()
    return {gene: int(year) for gene, year in first_year.items()}


def build_feature_table(exploded: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    training = exploded.loc[exploded["Year_int"].le(cutoff_year)].copy()
    records = []
    for gene, gene_rows in training.groupby("gene"):
        periph_count = gene_rows.loc[
            peripheral_mask(gene_rows["Tissue"]) & gene_rows["AUC_num"].ge(0.80)
        ].shape[0]
        brain_count = gene_rows.loc[brain_mask(gene_rows["Tissue"])].shape[0]
        max_auc = gene_rows["AUC_num"].max()
        min_year = int(gene_rows["Year_int"].min())
        records.append(
            {
                "gene": gene,
                "count_peripheral_sigs": int(periph_count),
                "count_brain_sigs": int(brain_count),
                "max_auc": float(0.0 if pd.isna(max_auc) else max_auc),
                "n_years_active": int(cutoff_year - min_year),
            }
        )
    return pd.DataFrame(records).sort_values("gene").reset_index(drop=True)


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_SEED) -> tuple[np.ndarray, float, str]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives < 2 or negatives < 2:
        raise ValueError("At least two positives and two negatives are required for model evaluation.")

    if positives < 20:
        splitter = LeaveOneOut()
        split_iter = splitter.split(X)
        cv_name = "LOOCV"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
        cv_name = "StratifiedKFold(5)"

    preds = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    for train_idx, test_idx in split_iter:
        y_train = y[train_idx]
        if np.unique(y_train).size < 2:
            raise ValueError("A training fold contains only one class; AUROC is undefined.")
        model = make_xgb()
        model.fit(X[train_idx], y_train)
        preds[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    if np.isnan(preds).any():
        raise ValueError("Failed to produce predictions for every held-out example.")

    auroc = float(roc_auc_score(y, preds))
    return preds, auroc, cv_name


def build_standard_labels(genes: list[str], first_peripheral_year: dict[str, int], cutoff_year: int) -> tuple[set[str], set[str]]:
    positives = {gene for gene in genes if first_peripheral_year.get(gene) == cutoff_year + 1}
    negatives = {
        gene
        for gene in genes
        if gene not in first_peripheral_year or int(first_peripheral_year.get(gene, 9999)) > 2025
    }
    return positives, negatives


def build_exclude_2024_labels(genes: list[str], first_peripheral_year: dict[str, int]) -> tuple[set[str], set[str], int]:
    excluded_2024 = {gene for gene in genes if first_peripheral_year.get(gene) == 2024}
    positives = {gene for gene in genes if first_peripheral_year.get(gene) == 2025}
    negatives = {
        gene
        for gene in genes
        if gene not in first_peripheral_year or int(first_peripheral_year.get(gene, 9999)) > 2025
    }
    return positives, negatives, len(excluded_2024)


def evaluate_split(
    feature_table: pd.DataFrame,
    positives: set[str],
    negatives: set[str],
    split_name: str,
    cutoff_year: int,
    strict_skip: bool,
) -> dict:
    eligible = feature_table.loc[feature_table["gene"].isin(positives | negatives)].copy()
    if eligible.empty:
        return {
            "split_name": split_name,
            "forecast_cutoff_year": cutoff_year,
            "forecast_year": cutoff_year + 1,
            "n_training_genes": int(feature_table.shape[0]),
            "n_eval_genes": 0,
            "n_pos": 0,
            "n_neg": 0,
            "cv_method": None,
            "auroc": None,
            "skipped": True,
            "skip_reason": "No eligible genes after label assignment.",
        }

    eligible["label"] = eligible["gene"].map(lambda gene: 1 if gene in positives else 0).astype(int)
    X = eligible[["count_peripheral_sigs", "count_brain_sigs", "max_auc", "n_years_active"]].to_numpy(dtype=float)
    y = eligible["label"].to_numpy(dtype=int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())

    result = {
        "split_name": split_name,
        "forecast_cutoff_year": cutoff_year,
        "forecast_year": cutoff_year + 1,
        "n_training_genes": int(feature_table.shape[0]),
        "n_eval_genes": int(eligible.shape[0]),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "cv_method": None,
        "auroc": None,
        "skipped": False,
        "skip_reason": None,
        "feature_columns": [
            "count_peripheral_sigs",
            "count_brain_sigs",
            "max_auc",
            "n_years_active",
        ],
        "feature_rows": eligible[
            ["gene", "count_peripheral_sigs", "count_brain_sigs", "max_auc", "n_years_active", "label"]
        ].to_dict(orient="records"),
    }

    if strict_skip and n_pos < 5:
        result["skipped"] = True
        result["skip_reason"] = "Fewer than 5 positives in this split."
        return result

    try:
        preds, auroc, cv_name = cross_validated_auroc(X, y)
        result["cv_method"] = cv_name
        result["auroc"] = auroc
        result["predictions"] = [
            {
                "gene": str(gene),
                "label": int(label),
                "predicted_probability": float(prob),
            }
            for gene, label, prob in zip(eligible["gene"], y, preds, strict=True)
        ]
    except Exception as exc:
        result["skipped"] = True
        result["skip_reason"] = str(exc)

    return result


def run_permutation_test(split_result: dict, n_permutations: int = PERMUTATION_COUNT) -> dict:
    if split_result.get("skipped"):
        return {
            "selected_split": split_result.get("split_name"),
            "forecast_cutoff_year": split_result.get("forecast_cutoff_year"),
            "forecast_year": split_result.get("forecast_year"),
            "n_permutations": n_permutations,
            "observed_auroc": None,
            "null_mean_auroc": None,
            "null_std_auroc": None,
            "p_value": None,
            "skipped": True,
            "skip_reason": split_result.get("skip_reason"),
        }

    feature_rows = pd.DataFrame(split_result["feature_rows"])
    y = feature_rows["label"].to_numpy(dtype=int)
    X = feature_rows[["count_peripheral_sigs", "count_brain_sigs", "max_auc", "n_years_active"]].to_numpy(
        dtype=float
    )
    observed_auroc = float(split_result["auroc"])
    null_aurocs = []
    rng = np.random.default_rng(RANDOM_SEED)

    for perm_idx in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            if int(y_perm.sum()) < 2 or int((1 - y_perm).sum()) < 2:
                continue
            _, perm_auroc, _ = cross_validated_auroc(X, y_perm, random_state=RANDOM_SEED + perm_idx + 1)
            null_aurocs.append(float(perm_auroc))
        except Exception:
            continue

    if not null_aurocs:
        return {
            "selected_split": split_result.get("split_name"),
            "forecast_cutoff_year": split_result.get("forecast_cutoff_year"),
            "forecast_year": split_result.get("forecast_year"),
            "n_permutations": n_permutations,
            "observed_auroc": observed_auroc,
            "null_mean_auroc": None,
            "null_std_auroc": None,
            "p_value": None,
            "skipped": True,
            "skip_reason": "No valid permutations completed.",
        }

    p_value = (1.0 + sum(auc >= observed_auroc for auc in null_aurocs)) / (1.0 + len(null_aurocs))
    return {
        "selected_split": split_result.get("split_name"),
        "forecast_cutoff_year": split_result.get("forecast_cutoff_year"),
        "forecast_year": split_result.get("forecast_year"),
        "n_permutations": n_permutations,
        "n_valid_permutations": len(null_aurocs),
        "observed_auroc": observed_auroc,
        "null_mean_auroc": float(np.mean(null_aurocs)),
        "null_std_auroc": float(np.std(null_aurocs)),
        "p_value": float(p_value),
        "skipped": False,
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def make_plot(main_results: list[dict], auxiliary_2023: dict, permutation_result: dict) -> None:
    years = [result["forecast_year"] for result in main_results]
    aurocs = [np.nan if result["auroc"] is None else float(result["auroc"]) for result in main_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, aurocs, marker="o", linewidth=2, color="#2c7fb8", label="Rolling-origin AUROC")

    if auxiliary_2023.get("auroc") is not None:
        ax.scatter(
            [auxiliary_2023["forecast_year"]],
            [auxiliary_2023["auroc"]],
            marker="^",
            s=90,
            color="#d95f0e",
            label="Auxiliary 2023->2024 split",
        )

    if not permutation_result.get("skipped"):
        mean_auc = float(permutation_result["null_mean_auroc"])
        std_auc = float(permutation_result["null_std_auroc"])
        xmin = min(years)
        xmax = max(years)
        ax.fill_between(
            [xmin, xmax],
            [mean_auc - std_auc, mean_auc - std_auc],
            [mean_auc + std_auc, mean_auc + std_auc],
            color="#bdbdbd",
            alpha=0.35,
            label="Permutation null mean +/- std",
        )
        ax.axhline(mean_auc, color="#737373", linestyle="--", linewidth=1)

    for result in main_results:
        if result.get("skipped"):
            ax.text(
                result["forecast_year"],
                0.03,
                f"n+={result['n_pos']}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4d4d4d",
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel("AUROC")
    ax.set_title("Rolling-origin temporal stress test")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df, exploded = load_exploded_signature(SIGNATURE_PATH)
    if exploded.empty:
        raise ValueError("No exploded signature rows were generated.")

    first_peripheral_year = build_first_peripheral_year_map(exploded)
    main_results = []

    for cutoff_year in MAIN_CUTOFF_YEARS:
        feature_table = build_feature_table(exploded, cutoff_year)
        genes = feature_table["gene"].tolist()
        positives, negatives = build_standard_labels(genes, first_peripheral_year, cutoff_year)
        result = evaluate_split(
            feature_table=feature_table,
            positives=positives,
            negatives=negatives,
            split_name=f"cutoff_{cutoff_year}",
            cutoff_year=cutoff_year,
            strict_skip=True,
        )
        main_results.append(result)

    feature_table_2023 = build_feature_table(exploded, 2023)
    genes_2023 = feature_table_2023["gene"].tolist()
    positives_2023, negatives_2023 = build_standard_labels(genes_2023, first_peripheral_year, 2023)
    auxiliary_2023 = evaluate_split(
        feature_table=feature_table_2023,
        positives=positives_2023,
        negatives=negatives_2023,
        split_name="cutoff_2023_standard",
        cutoff_year=2023,
        strict_skip=False,
    )

    exclude_2024_positives, exclude_2024_negatives, excluded_2024_gene_count = build_exclude_2024_labels(
        genes_2023, first_peripheral_year
    )
    exclude_2024_result = evaluate_split(
        feature_table=feature_table_2023,
        positives=exclude_2024_positives,
        negatives=exclude_2024_negatives,
        split_name="exclude_2024_test",
        cutoff_year=2023,
        strict_skip=False,
    )
    exclude_2024_result["excluded_2024_gene_count"] = excluded_2024_gene_count
    exclude_2024_result[
        "definition"
    ] = "Training uses Year<=2023. Positives are genes first maturing in 2025 after excluding 2024 maturers."

    permutation_candidates = main_results + [auxiliary_2023]
    eligible_perm_candidates = [result for result in permutation_candidates if result["n_pos"] >= 2 and result["n_neg"] >= 2]
    if eligible_perm_candidates:
        selected = max(eligible_perm_candidates, key=lambda item: (item["n_pos"], item["n_eval_genes"]))
        permutation_result = run_permutation_test(selected)
    else:
        permutation_result = {
            "selected_split": None,
            "forecast_cutoff_year": None,
            "forecast_year": None,
            "n_permutations": PERMUTATION_COUNT,
            "observed_auroc": None,
            "null_mean_auroc": None,
            "null_std_auroc": None,
            "p_value": None,
            "skipped": True,
            "skip_reason": "No split had at least two positives and two negatives for permutation testing.",
        }

    make_plot(main_results, auxiliary_2023, permutation_result)

    peripheral_rows = raw_df.loc[
        peripheral_mask(raw_df["Tissue"]) & raw_df["AUC_num"].ge(0.88) & raw_df["Year_int"].notna()
    ].copy()

    results = {
        "random_seed": RANDOM_SEED,
        "n_signature_rows": int(raw_df.shape[0]),
        "n_exploded_rows": int(exploded.shape[0]),
        "n_filtered_high_auc_peripheral_rows": int(peripheral_rows.shape[0]),
        "n_genes_with_first_peripheral_year": int(len(first_peripheral_year)),
        "positive_label_years": POSITIVE_LABEL_YEARS,
        "forecast_results": main_results,
        "auxiliary_cutoff_2023_standard": auxiliary_2023,
        "exclude_2024_test": exclude_2024_result,
        "permutation_test": permutation_result,
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("Experiment 2: Rolling-origin temporal stress test")
    for result in main_results:
        auroc_text = f"{result['auroc']:.3f}" if result["auroc"] is not None else "NA"
        status = "skipped" if result["skipped"] else "ok"
        print(
            f"Cutoff {result['forecast_cutoff_year']} -> {result['forecast_year']}: "
            f"n_pos={result['n_pos']}, n_neg={result['n_neg']}, auroc={auroc_text}, status={status}"
        )
    aux_text = f"{auxiliary_2023['auroc']:.3f}" if auxiliary_2023["auroc"] is not None else "NA"
    excl_text = f"{exclude_2024_result['auroc']:.3f}" if exclude_2024_result["auroc"] is not None else "NA"
    print(f"Auxiliary 2023->2024 split AUROC: {aux_text}")
    print(f"Exclude-2024 test AUROC: {excl_text}")
    if permutation_result.get("skipped"):
        print(f"Permutation test skipped: {permutation_result['skip_reason']}")
    else:
        print(
            f"Permutation split={permutation_result['selected_split']}, "
            f"observed_auroc={permutation_result['observed_auroc']:.3f}, "
            f"p_value={permutation_result['p_value']:.6g}"
        )
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
