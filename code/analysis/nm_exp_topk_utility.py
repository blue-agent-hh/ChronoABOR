#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_JSON = RESULTS_DIR / "experiment_results.json"
OUTPUT_JSON = RESULTS_DIR / "nm_exp3_topk_utility.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp3_plots.png"
RANDOM_SEED = 42
TOP_K_VALUES = [5, 10, 15, 20, 25, 50]


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


def load_labeled_targets(path: Path) -> tuple[pd.DataFrame, list[str]]:
    payload = json.loads(path.read_text())
    feature_columns = []
    for layer in ["snp", "cpg", "deg", "hub", "mirna", "signature"]:
        feature_columns.extend(payload["feature_columns"][layer])

    df = pd.DataFrame(payload["targets"])
    df = df.loc[df["label"].isin([0, 1, 0.0, 1.0])].copy()
    df["label"] = df["label"].astype(int)
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["publication_score"] = df["deg_up_count"].fillna(0.0) + df["deg_down_count"].fillna(0.0)
    return df.reset_index(drop=True), feature_columns


def cross_validated_probabilities(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives < 2 or negatives < 2:
        raise ValueError("Need at least two positives and two negatives for cross-validated predictions.")

    if len(y) > 100:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X, y)
        cv_name = "StratifiedKFold(5)"
    else:
        splitter = LeaveOneOut()
        split_iter = splitter.split(X)
        cv_name = "LOOCV"

    preds = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    for train_idx, test_idx in split_iter:
        y_train = y[train_idx]
        if np.unique(y_train).size < 2:
            raise ValueError("A training fold contains a single class.")
        model = make_xgb()
        model.fit(X[train_idx], y_train)
        preds[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    if np.isnan(preds).any():
        raise ValueError("Not all held-out predictions were populated.")
    return preds, cv_name


def compute_topk_metrics(df: pd.DataFrame, ks: list[int], base_rate: float) -> list[dict]:
    rows = []
    for k in ks:
        if k > len(df):
            continue
        subset = df.head(k)
        tp = int(subset["label"].sum())
        precision = float(tp / k)
        recall = float(tp / int(df["label"].sum()))
        lift = float(precision / base_rate) if base_rate > 0 else None
        nnt = float(k / tp) if tp > 0 else None
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


def compute_reliability_bins(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(probs, bins[1:-1], right=True)
    records = []
    for idx in range(n_bins):
        mask = assignments == idx
        if mask.any():
            mean_pred = float(np.mean(probs[mask]))
            frac_pos = float(np.mean(y[mask]))
            count = int(mask.sum())
        else:
            mean_pred = None
            frac_pos = None
            count = 0
        records.append(
            {
                "bin_index": idx,
                "bin_lower": float(bins[idx]),
                "bin_upper": float(bins[idx + 1]),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "fraction_positive": frac_pos,
            }
        )
    return records


def decision_curve(y: np.ndarray, probs: np.ndarray) -> list[dict]:
    thresholds = np.linspace(0.01, 0.50, 50)
    n = len(y)
    prevalence = float(np.mean(y))
    rows = []
    for threshold in thresholds:
        predicted_positive = probs >= threshold
        tp = int(np.sum(predicted_positive & (y == 1)))
        fp = int(np.sum(predicted_positive & (y == 0)))
        weight = float(threshold / (1.0 - threshold))
        net_benefit = float((tp / n) - (fp / n) * weight)
        treat_all = float(prevalence - (1.0 - prevalence) * weight)
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit_model": net_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    return rows


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


def make_plot(
    y: np.ndarray,
    probs: np.ndarray,
    ranked_df: pd.DataFrame,
    topk_rows: list[dict],
    dca_rows: list[dict],
) -> None:
    precision_curve, recall_curve, _ = precision_recall_curve(y, probs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(recall_curve, precision_curve, color="#1b9e77", linewidth=2, label="ChronoABOR")
    axes[0].axhline(float(np.mean(y)), color="#666666", linestyle="--", linewidth=1, label="Random baseline")
    for row in topk_rows:
        k = row["k"]
        subset = ranked_df.head(k)
        recall = float(subset["label"].sum() / y.sum())
        precision = float(subset["label"].sum() / k)
        axes[0].scatter(recall, precision, s=45, color="#d95f02")
        axes[0].text(recall, precision, f"k={k}", fontsize=8, ha="left", va="bottom")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-recall with top-k markers")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower left")

    thresholds = [row["threshold"] for row in dca_rows]
    model_nb = [row["net_benefit_model"] for row in dca_rows]
    all_nb = [row["net_benefit_treat_all"] for row in dca_rows]
    none_nb = [row["net_benefit_treat_none"] for row in dca_rows]
    axes[1].plot(thresholds, model_nb, color="#7570b3", linewidth=2, label="ChronoABOR")
    axes[1].plot(thresholds, all_nb, color="#e7298a", linestyle="--", linewidth=1.5, label="Treat all")
    axes[1].plot(thresholds, none_nb, color="#555555", linestyle=":", linewidth=1.5, label="Treat none")
    axes[1].set_xlabel("Threshold probability")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_title("Decision curve analysis")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_df, feature_columns = load_labeled_targets(INPUT_JSON)
    X = labeled_df[feature_columns].to_numpy(dtype=float)
    y = labeled_df["label"].to_numpy(dtype=int)
    probs, cv_name = cross_validated_probabilities(X, y)
    labeled_df["predicted_probability"] = probs

    auroc = float(roc_auc_score(y, probs))
    average_precision = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))
    base_rate = float(np.mean(y))

    ranked_df = labeled_df.sort_values(
        by=["predicted_probability", "publication_score", "GeneSymbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    published_df = labeled_df.sort_values(
        by=["publication_score", "GeneSymbol"],
        ascending=[False, True],
    ).reset_index(drop=True)

    topk_chronoabor = compute_topk_metrics(ranked_df, TOP_K_VALUES, base_rate)
    topk_published = compute_topk_metrics(published_df, TOP_K_VALUES, base_rate)
    topk_table = []
    published_lookup = {row["k"]: row for row in topk_published}
    for row in topk_chronoabor:
        baseline = published_lookup[row["k"]]
        topk_table.append(
            {
                "k": row["k"],
                "chronoabor_precision_at_k": row["precision_at_k"],
                "chronoabor_recall_at_k": row["recall_at_k"],
                "chronoabor_lift_at_k": row["lift_at_k"],
                "chronoabor_nnt_at_k": row["nnt_at_k"],
                "random_expected_precision_at_k": base_rate,
                "most_published_precision_at_k": baseline["precision_at_k"],
                "most_published_recall_at_k": baseline["recall_at_k"],
                "most_published_lift_at_k": baseline["lift_at_k"],
                "most_published_nnt_at_k": baseline["nnt_at_k"],
            }
        )

    reliability_bins = compute_reliability_bins(y, probs, n_bins=10)
    dca_rows = decision_curve(y, probs)
    make_plot(y, probs, ranked_df, topk_chronoabor, dca_rows)

    ranking_records = ranked_df[
        ["GeneSymbol", "label", "predicted_probability", "publication_score"] + feature_columns
    ].to_dict(orient="records")

    results = {
        "random_state": RANDOM_SEED,
        "cv_method": cv_name,
        "n_labeled_genes": int(len(labeled_df)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "base_rate": base_rate,
        "feature_columns": feature_columns,
        "auroc": auroc,
        "average_precision": average_precision,
        "brier_score": brier,
        "precision_at_k_table": topk_table,
        "reliability_bins": reliability_bins,
        "decision_curve": dca_rows,
        "ranking": ranking_records,
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("Experiment 3: Prospective top-k utility analysis")
    print(f"Cross-validation: {cv_name}")
    print(f"AUROC: {auroc:.3f}")
    print(f"Average precision: {average_precision:.3f}")
    print(f"Brier score: {brier:.3f}")
    for row in topk_table:
        print(
            f"k={row['k']}: precision={row['chronoabor_precision_at_k']:.3f}, "
            f"recall={row['chronoabor_recall_at_k']:.3f}, "
            f"lift={row['chronoabor_lift_at_k']:.3f}"
        )
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
