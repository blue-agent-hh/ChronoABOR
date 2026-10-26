import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
RESULTS_DIR = "results"
LEAKAGE_FIELDS = [
    "ClinicalEfficacyEvidence",
    "RegulatoryApproval",
    "TVL",
    "GeneticEvidence",
    "MolecularEvidence",
    "AnimalModelEvidence",
    "HumanBiomarkerEvidence",
]
SYMPTOMATIC_DRUG_TERMS = {
    "donepezil",
    "memantine",
    "rivastigmine",
    "galantamine",
}
LAYER_NAMES = ["snp", "cpg", "deg", "hub", "mirna", "signature"]
LAYER_FEATURE_COLUMNS = {
    "snp": ["in_snp", "snp_count"],
    "cpg": ["in_cpg", "cpg_count"],
    "deg": ["in_deg", "deg_up_count", "deg_down_count", "deg_tissue_breadth"],
    "hub": ["in_hub", "hub_count"],
    "mirna": ["in_mirna", "mirna_count", "mirna_up_count", "mirna_down_count"],
    "signature": ["in_sig", "sig_peripheral", "sig_brain_only", "sig_max_auc"],
}
PERIPHERAL_TISSUE_PATTERN = re.compile(r"\b(plasma|blood|serum)\b", flags=re.IGNORECASE)
BRAIN_TISSUE_PATTERN = re.compile(
    r"\b(brain|cortex|cortical|hippocampus|hippocampal|temporal|frontal|entorhinal|gyrus)\b",
    flags=re.IGNORECASE,
)
SPECIAL_GENE_SPLITS = {
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


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_gene_symbol(value) -> str:
    return normalize_text(value).upper()


def parse_numeric(value, default=0.0) -> float:
    if pd.isna(value):
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return float(default)
    return float(match.group(0))


def deduplicate_keep_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def expand_shorthand_symbol(base_symbol: str, suffix: str, known_symbols) -> str:
    if suffix in known_symbols:
        return suffix
    for cut in range(1, len(base_symbol)):
        candidate = base_symbol[:-cut] + suffix
        if candidate in known_symbols:
            return candidate
    return suffix


def split_gene_symbol(raw_symbol, known_symbols) -> list:
    text = normalize_gene_symbol(raw_symbol)
    if not text or text == "MULTIPLE":
        return []
    if text in SPECIAL_GENE_SPLITS:
        return SPECIAL_GENE_SPLITS[text]

    tokens = re.split(r"[;,]", text)
    genes = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "/" not in token:
            genes.append(token)
            continue

        parts = [part.strip() for part in token.split("/") if part.strip()]
        if not parts:
            continue
        genes.append(parts[0])
        base_symbol = parts[0]
        for part in parts[1:]:
            genes.append(expand_shorthand_symbol(base_symbol, part, known_symbols))

    return deduplicate_keep_order(genes)


def build_known_gene_universe(datasets):
    universe = set()
    direct_columns = [
        ("target", "GeneSymbol"),
        ("snp", "Gene_symbol"),
        ("cpg", "Gene_symbol"),
        ("degs", "Gene_symbol"),
        ("hubgene", "Gene_symbl"),
        ("miRNA", "Gene_symbol"),
    ]
    for dataset_name, column in direct_columns:
        if column not in datasets[dataset_name].columns:
            continue
        for value in datasets[dataset_name][column].dropna():
            text = normalize_gene_symbol(value)
            if text:
                universe.add(text)

    if "Gene_symbol" in datasets["signature"].columns:
        for raw in datasets["signature"]["Gene_symbol"].dropna():
            for token in re.split(r"[,;/]", normalize_gene_symbol(raw)):
                token = token.strip()
                if token:
                    universe.add(token)
    return universe


def load_data():
    filenames = {
        "target": "target.csv",
        "snp": "snp.csv",
        "cpg": "cpg.csv",
        "degs": "degs.csv",
        "hubgene": "hubgene.csv",
        "miRNA": "miRNA.csv",
        "signature": "signature.csv",
    }
    return {name: pd.read_csv(path) for name, path in filenames.items()}


def filter_ad_datasets(datasets):
    degs = datasets["degs"].copy()
    degs_before = len(degs)
    degs = degs[degs["Disease"].fillna("").str.contains("Alzheimer", case=False, na=False)].copy()

    hub = datasets["hubgene"].copy()
    hub_before = len(hub)
    combined_text = (
        hub["Alteration"].fillna("").astype(str)
        + " "
        + hub["Tissue"].fillna("").astype(str)
        + " "
        + hub["Sample"].fillna("").astype(str)
    )
    hub_has_ad_context = combined_text.str.contains(r"\b(?:AD|Alzheimer)\b", case=False, regex=True, na=False)
    hub_exclusion = combined_text.str.contains(
        r"\b(?:MDD|depression|T2DM|diabetes|vascular dementia|VaD|UC|ulcerative colitis|blood|plasma|serum|peripheral|pancreatic|colon)\b",
        case=False,
        regex=True,
        na=False,
    )
    hub = hub[hub_has_ad_context & ~hub_exclusion].copy()

    filtered = datasets.copy()
    filtered["target"] = datasets["target"].drop(columns=LEAKAGE_FIELDS, errors="ignore").copy()
    filtered["degs"] = degs
    filtered["hubgene"] = hub

    print_header("1. Data Loading and AD-Specific Filtering")
    print(f"Loaded target.csv: {len(datasets['target'])} rows")
    print(f"Loaded snp.csv: {len(datasets['snp'])} rows")
    print(f"Loaded cpg.csv: {len(datasets['cpg'])} rows")
    print(f"Loaded degs.csv: {degs_before} rows -> {len(degs)} AD-only rows")
    print(f"Loaded hubgene.csv: {hub_before} rows -> {len(hub)} AD-focused hub rows")
    print(f"Loaded miRNA.csv: {len(datasets['miRNA'])} rows")
    print(f"Loaded signature.csv: {len(datasets['signature'])} rows")
    print("\nLeakage fields removed from target.csv:")
    for column in LEAKAGE_FIELDS:
        print(f"- {column}")

    return filtered


def construct_labels(target_df):
    target = target_df.copy()
    target["GeneSymbol"] = target["GeneSymbol"].fillna("").astype(str)
    target["DrugStatus_norm"] = target["DrugStatus"].fillna("").astype(str).str.lower().str.strip()
    target["DrugName_norm"] = target["DrugName"].fillna("").astype(str).str.lower()

    symptomatic_mask = target["DrugName_norm"].apply(
        lambda text: any(term in text for term in SYMPTOMATIC_DRUG_TERMS)
    )
    positive_mask = (
        target["DrugStatus_norm"].str.contains("fda-approved", na=False)
        | target["DrugStatus_norm"].isin(["phase 3", "phase 2/3"])
    ) & ~symptomatic_mask
    negative_mask = (
        target["DrugStatus_norm"].str.contains("preclinical", na=False)
        | target["DrugStatus_norm"].eq("phase 1")
        | target["DrugStatus_norm"].str.contains("discontinued", na=False)
    ) & ~positive_mask

    target["is_positive"] = positive_mask
    target["is_negative"] = negative_mask

    grouped_rows = []
    for gene_symbol, group in target.groupby("GeneSymbol", dropna=False):
        gene_symbol = normalize_text(gene_symbol)
        if not gene_symbol:
            continue
        label = math.nan
        if group["is_positive"].any():
            label = 1
        elif group["is_negative"].any():
            label = 0
        grouped_rows.append(
            {
                "GeneSymbol": gene_symbol,
                "label": label,
                "n_target_rows": int(len(group)),
                "drugs": sorted(group["DrugName"].fillna("").astype(str).unique().tolist()),
                "statuses": sorted(group["DrugStatus"].fillna("").astype(str).unique().tolist()),
            }
        )

    labels_df = pd.DataFrame(grouped_rows).sort_values("GeneSymbol").reset_index(drop=True)
    labeled_df = labels_df[labels_df["label"].isin([0, 1])].copy()
    labeled_df["label"] = labeled_df["label"].astype(int)

    print_header("2. Leakage-Free Label Construction")
    print(f"Unique target genes in target.csv: {labels_df['GeneSymbol'].nunique()}")
    print(f"Labeled target genes used in experiments: {len(labeled_df)}")
    print("Label distribution:")
    counts = labeled_df["label"].value_counts().sort_index()
    print(f"- Negative (y=0): {int(counts.get(0, 0))}")
    print(f"- Positive (y=1): {int(counts.get(1, 0))}")
    print(f"- Unlabeled/held out: {int(len(labels_df) - len(labeled_df))}")

    return labels_df


def build_gene_stats(datasets, known_symbols):
    stats = {
        "snp": defaultdict(lambda: {"count": 0}),
        "cpg": defaultdict(lambda: {"count": 0}),
        "deg": defaultdict(lambda: {"up_count": 0, "down_count": 0, "tissues": set()}),
        "hub": defaultdict(lambda: {"count": 0}),
        "mirna": defaultdict(lambda: {"count": 0, "up_count": 0, "down_count": 0}),
        "signature": defaultdict(lambda: {"tissues": [], "aucs": []}),
    }

    for gene in datasets["snp"]["Gene_symbol"].dropna():
        gene = normalize_gene_symbol(gene)
        if gene:
            stats["snp"][gene]["count"] += 1

    for gene in datasets["cpg"]["Gene_symbol"].dropna():
        gene = normalize_gene_symbol(gene)
        if gene:
            stats["cpg"][gene]["count"] += 1

    for _, row in datasets["degs"].iterrows():
        gene = normalize_gene_symbol(row.get("Gene_symbol"))
        if not gene:
            continue
        alteration = normalize_text(row.get("Alteration")).lower()
        if "up" in alteration:
            stats["deg"][gene]["up_count"] += 1
        if "down" in alteration:
            stats["deg"][gene]["down_count"] += 1
        tissue = normalize_text(row.get("Tissue"))
        if tissue:
            stats["deg"][gene]["tissues"].add(tissue)

    for gene in datasets["hubgene"]["Gene_symbl"].dropna():
        gene = normalize_gene_symbol(gene)
        if gene:
            stats["hub"][gene]["count"] += 1

    for _, row in datasets["miRNA"].iterrows():
        gene = normalize_gene_symbol(row.get("Gene_symbol"))
        if not gene:
            continue
        alteration = normalize_text(row.get("Alteration")).lower()
        stats["mirna"][gene]["count"] += 1
        if "up" in alteration:
            stats["mirna"][gene]["up_count"] += 1
        if "down" in alteration:
            stats["mirna"][gene]["down_count"] += 1

    for _, row in datasets["signature"].iterrows():
        genes = split_gene_symbol(row.get("Gene_symbol"), known_symbols)
        tissue = normalize_text(row.get("Tissue"))
        auc = parse_numeric(row.get("AUC"), default=0.0)
        for gene in genes:
            stats["signature"][gene]["tissues"].append(tissue)
            stats["signature"][gene]["aucs"].append(auc)

    return stats


def aggregate_target_features(target_gene_symbol, gene_stats, known_symbols):
    component_genes = split_gene_symbol(target_gene_symbol, known_symbols)

    snp_count = sum(gene_stats["snp"][gene]["count"] for gene in component_genes)
    cpg_count = sum(gene_stats["cpg"][gene]["count"] for gene in component_genes)
    deg_up = sum(gene_stats["deg"][gene]["up_count"] for gene in component_genes)
    deg_down = sum(gene_stats["deg"][gene]["down_count"] for gene in component_genes)

    deg_tissues = set()
    for gene in component_genes:
        deg_tissues.update(gene_stats["deg"][gene]["tissues"])

    hub_count = sum(gene_stats["hub"][gene]["count"] for gene in component_genes)
    mirna_count = sum(gene_stats["mirna"][gene]["count"] for gene in component_genes)
    mirna_up = sum(gene_stats["mirna"][gene]["up_count"] for gene in component_genes)
    mirna_down = sum(gene_stats["mirna"][gene]["down_count"] for gene in component_genes)

    sig_tissues = []
    sig_aucs = []
    for gene in component_genes:
        sig_tissues.extend(gene_stats["signature"][gene]["tissues"])
        sig_aucs.extend(gene_stats["signature"][gene]["aucs"])
    sig_peripheral = int(any(PERIPHERAL_TISSUE_PATTERN.search(tissue) for tissue in sig_tissues))
    sig_brain_only = int(
        bool(sig_tissues) and not sig_peripheral and all(BRAIN_TISSUE_PATTERN.search(tissue) for tissue in sig_tissues)
    )
    sig_max_auc = max(sig_aucs) if sig_aucs else 0.0

    features = {
        "target_components": component_genes,
        "in_snp": int(snp_count > 0),
        "snp_count": int(snp_count),
        "in_cpg": int(cpg_count > 0),
        "cpg_count": int(cpg_count),
        "in_deg": int((deg_up + deg_down) > 0),
        "deg_up_count": int(deg_up),
        "deg_down_count": int(deg_down),
        "deg_tissue_breadth": int(len(deg_tissues)),
        "in_hub": int(hub_count > 0),
        "hub_count": int(hub_count),
        "in_mirna": int(mirna_count > 0),
        "mirna_count": int(mirna_count),
        "mirna_up_count": int(mirna_up),
        "mirna_down_count": int(mirna_down),
        "in_sig": int(len(sig_tissues) > 0),
        "sig_peripheral": int(sig_peripheral),
        "sig_brain_only": int(sig_brain_only),
        "sig_max_auc": float(sig_max_auc),
    }
    features["layer_presence_vector"] = [
        features["in_snp"],
        features["in_cpg"],
        features["in_deg"],
        features["in_hub"],
        features["in_mirna"],
        features["in_sig"],
    ]
    return features


def build_feature_table(labels_df, datasets, known_symbols):
    gene_stats = build_gene_stats(datasets, known_symbols)
    records = []
    for _, row in labels_df.iterrows():
        features = aggregate_target_features(row["GeneSymbol"], gene_stats, known_symbols)
        records.append({"GeneSymbol": row["GeneSymbol"], "label": row["label"], **features})

    feature_df = pd.DataFrame(records)

    print_header("3. Per-Target 6-Layer Feature Construction")
    print(f"Feature table shape: {feature_df.shape}")
    print("Layer presence counts:")
    for layer_name, presence_col in zip(LAYER_NAMES, ["in_snp", "in_cpg", "in_deg", "in_hub", "in_mirna", "in_sig"]):
        print(f"- {layer_name}: {int(feature_df[presence_col].sum())} / {len(feature_df)} targets")

    return feature_df


def get_layer_matrices(feature_df):
    matrices = {}
    for layer_name, columns in LAYER_FEATURE_COLUMNS.items():
        matrices[layer_name] = feature_df[columns].astype(float).to_numpy()
    return matrices


def build_feature_matrix(layer_matrices, subset, n_samples):
    if not subset:
        return np.zeros((n_samples, 0), dtype=float)
    blocks = [layer_matrices[LAYER_NAMES[idx]] for idx in subset]
    return np.hstack(blocks).astype(float)


def fit_predict_single_split(X_train, y_train, X_test):
    if X_train.shape[1] == 0 or len(np.unique(y_train)) < 2:
        return float(np.mean(y_train))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                    max_iter=1000,
                ),
            ),
        ]
    )
    try:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[0, 1]
    except Exception:
        prob = float(np.mean(y_train))
    return float(np.clip(prob, 1e-6, 1 - 1e-6))


def loo_predict_lr(X, y):
    loo = LeaveOneOut()
    probabilities = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        probabilities[test_idx[0]] = fit_predict_single_split(X[train_idx], y[train_idx], X[test_idx])
    return probabilities


def evaluate_predictions(y_true, probabilities):
    metrics = {
        "accuracy": float(accuracy_score(y_true, (probabilities >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def evaluate_subset(layer_matrices, subset, y):
    X = build_feature_matrix(layer_matrices, subset, n_samples=len(y))
    probabilities = loo_predict_lr(X, y)
    result = {
        "subset": list(subset),
        "subset_name": "empty" if not subset else "+".join(LAYER_NAMES[idx] for idx in subset),
        "budget": len(subset),
        "n_features": int(X.shape[1]),
    }
    result.update(evaluate_predictions(y, probabilities))
    return result


def evaluate_random_baseline(y):
    rng = np.random.RandomState(RANDOM_STATE)
    probabilities = rng.uniform(low=0.0, high=1.0, size=len(y))
    result = {"subset": [], "subset_name": "random", "budget": 0, "n_features": 0}
    result.update(evaluate_predictions(y, probabilities))
    return result


def enumerate_all_subsets():
    subsets = []
    for subset_size in range(len(LAYER_NAMES) + 1):
        for subset in combinations(range(len(LAYER_NAMES)), subset_size):
            subsets.append(subset)
    return subsets


def plot_subset_auroc_summary(oracle_df, best_subset_row, output_path):
    mean_by_budget = oracle_df.groupby("budget", as_index=False)["auroc"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(mean_by_budget["budget"], mean_by_budget["auroc"], marker="o", linewidth=2, color="#1f77b4")
    plt.scatter(
        best_subset_row["budget"],
        best_subset_row["auroc"],
        color="#d62728",
        s=100,
        label=f"Best subset: {best_subset_row['subset_name']}",
        zorder=5,
    )
    plt.xticks(range(0, len(LAYER_NAMES) + 1))
    plt.xlabel("Number of layers")
    plt.ylabel("LOO AUROC")
    plt.title("Oracle Mean AUROC by Layer Budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_subsets(oracle_df, output_path, top_k=10):
    top_df = oracle_df.sort_values(["auroc", "accuracy"], ascending=[False, False]).head(top_k)
    plt.figure(figsize=(11, 6))
    plt.barh(top_df["subset_name"][::-1], top_df["auroc"][::-1], color="#2ca02c")
    plt.xlabel("LOO AUROC")
    plt.ylabel("Layer subset")
    plt.title(f"Top-{top_k} Layer Subsets by AUROC")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def greedy_acquisition(oracle_lookup):
    current_subset = tuple()
    remaining = set(range(len(LAYER_NAMES)))
    path = [
        {
            "budget": 0,
            "subset": [],
            "subset_name": "empty",
            "auroc": float(oracle_lookup[current_subset]["auroc"]),
        }
    ]
    order = []
    current_auroc = float(oracle_lookup[current_subset]["auroc"])

    while remaining:
        best_layer = None
        best_candidate_result = None
        best_improvement = -math.inf
        for layer_idx in sorted(remaining):
            candidate_subset = tuple(sorted(current_subset + (layer_idx,)))
            candidate_result = oracle_lookup[candidate_subset]
            candidate_auroc = float(candidate_result["auroc"])
            improvement = candidate_auroc - current_auroc
            if (
                best_candidate_result is None
                or improvement > best_improvement
                or (
                    math.isclose(improvement, best_improvement)
                    and candidate_auroc > float(best_candidate_result["auroc"])
                )
            ):
                best_layer = layer_idx
                best_candidate_result = candidate_result
                best_improvement = improvement

        current_subset = tuple(sorted(current_subset + (best_layer,)))
        current_auroc = float(best_candidate_result["auroc"])
        remaining.remove(best_layer)
        order.append(LAYER_NAMES[best_layer])
        path.append(
            {
                "budget": len(current_subset),
                "subset": list(current_subset),
                "subset_name": best_candidate_result["subset_name"],
                "added_layer": LAYER_NAMES[best_layer],
                "auroc": current_auroc,
                "improvement": float(best_improvement),
            }
        )

    return {"order": order, "path": path}


def plot_budget_curves(best_by_budget, greedy_path, output_path):
    budgets = sorted(best_by_budget)
    oracle_aurocs = [best_by_budget[budget]["auroc"] for budget in budgets]
    greedy_lookup = {step["budget"]: step["auroc"] for step in greedy_path if step["budget"] > 0}
    greedy_aurocs = [greedy_lookup.get(budget, np.nan) for budget in budgets]

    plt.figure(figsize=(8, 5))
    plt.plot(budgets, oracle_aurocs, marker="o", linewidth=2, color="#1f77b4", label="Oracle best subset")
    plt.plot(budgets, greedy_aurocs, marker="s", linewidth=2, color="#ff7f0e", label="Greedy acquisition")
    plt.xticks(budgets)
    plt.xlabel("Budget (number of layers)")
    plt.ylabel("LOO AUROC")
    plt.title("Utility vs Budget")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def layer_frequency_in_top_k(oracle_df, top_k=10):
    non_empty = oracle_df[oracle_df["budget"] > 0].copy()
    top_df = non_empty.sort_values(["auroc", "accuracy"], ascending=[False, False]).head(top_k)
    counts = Counter()
    for subset in top_df["subset"]:
        for layer_idx in subset:
            counts[LAYER_NAMES[layer_idx]] += 1
    return {layer: int(counts.get(layer, 0)) for layer in LAYER_NAMES}


def plot_layer_frequency(layer_frequency, top_k, output_path):
    items = sorted(layer_frequency.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="#9467bd")
    plt.xlabel("Layer")
    plt.ylabel("Frequency")
    plt.title(f"Layer Frequency in Top-{top_k} Oracle Subsets")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, tuple):
        return [json_ready(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw_datasets = load_data()
    datasets = filter_ad_datasets(raw_datasets)
    known_symbols = build_known_gene_universe(raw_datasets)

    target_labels = construct_labels(datasets["target"])
    if target_labels.empty:
        raise RuntimeError("No target genes found after label construction.")

    feature_df = build_feature_table(target_labels, datasets, known_symbols)
    eval_df = feature_df[feature_df["label"].isin([0, 1])].copy().sort_values("GeneSymbol").reset_index(drop=True)
    eval_df["label"] = eval_df["label"].astype(int)
    y = eval_df["label"].to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Need at least one positive and one negative labeled target gene to run the experiment.")

    print_header("4. Static Baselines with Leave-One-Out CV")
    layer_matrices = get_layer_matrices(eval_df)
    baseline_results = {}
    baseline_results["all_layers"] = evaluate_subset(layer_matrices, tuple(range(len(LAYER_NAMES))), y)
    print(
        f"All layers LR: AUROC={baseline_results['all_layers']['auroc']:.4f}, "
        f"Brier={baseline_results['all_layers']['brier']:.4f}, "
        f"Accuracy={baseline_results['all_layers']['accuracy']:.4f}"
    )
    for layer_idx, layer_name in enumerate(LAYER_NAMES):
        result = evaluate_subset(layer_matrices, (layer_idx,), y)
        baseline_results[f"{layer_name}_only"] = result
        print(
            f"{layer_name} only LR: AUROC={result['auroc']:.4f}, "
            f"Brier={result['brier']:.4f}, Accuracy={result['accuracy']:.4f}"
        )
    baseline_results["random"] = evaluate_random_baseline(y)
    print(
        f"Random baseline: AUROC={baseline_results['random']['auroc']:.4f}, "
        f"Brier={baseline_results['random']['brier']:.4f}, "
        f"Accuracy={baseline_results['random']['accuracy']:.4f}"
    )

    print_header("5. Exhaustive Subset Oracle")
    subset_results = [evaluate_subset(layer_matrices, subset, y) for subset in enumerate_all_subsets()]
    oracle_df = pd.DataFrame(subset_results)
    oracle_df = oracle_df.sort_values(["auroc", "accuracy", "budget"], ascending=[False, False, True]).reset_index(drop=True)
    best_subset_row = oracle_df.iloc[0]
    print(
        f"Best subset: {best_subset_row['subset_name']} "
        f"(budget={int(best_subset_row['budget'])}, AUROC={best_subset_row['auroc']:.4f}, "
        f"Brier={best_subset_row['brier']:.4f}, Accuracy={best_subset_row['accuracy']:.4f})"
    )
    print("Mean AUROC by number of layers:")
    mean_auroc_by_budget = oracle_df.groupby("budget")["auroc"].mean()
    for budget, value in mean_auroc_by_budget.items():
        print(f"- {int(budget)} layers: {value:.4f}")
    print("Top-10 subsets by AUROC:")
    for _, row in oracle_df.head(10).iterrows():
        print(
            f"- {row['subset_name']}: AUROC={row['auroc']:.4f}, "
            f"Brier={row['brier']:.4f}, Accuracy={row['accuracy']:.4f}"
        )

    plot_subset_auroc_summary(
        oracle_df,
        best_subset_row,
        os.path.join(RESULTS_DIR, "oracle_mean_auroc_by_budget.png"),
    )
    plot_top_subsets(
        oracle_df,
        os.path.join(RESULTS_DIR, "oracle_top10_subsets.png"),
        top_k=10,
    )

    print_header("6. Greedy Information Gain Acquisition")
    oracle_lookup = {tuple(result["subset"]): result for result in subset_results}
    greedy_results = greedy_acquisition(oracle_lookup)
    print("Acquisition order:")
    print(f"- {' -> '.join(greedy_results['order'])}")
    print("AUROC path:")
    for step in greedy_results["path"]:
        if step["budget"] == 0:
            print(f"- Budget 0 ({step['subset_name']}): AUROC={step['auroc']:.4f}")
        else:
            print(
                f"- Budget {step['budget']} ({step['subset_name']}): AUROC={step['auroc']:.4f} "
                f"[added {step['added_layer']}, delta={step['improvement']:.4f}]"
            )

    print_header("7. Utility-Budget Curve")
    best_by_budget = {}
    for budget in range(1, len(LAYER_NAMES) + 1):
        budget_df = oracle_df[oracle_df["budget"] == budget].copy()
        best_row = budget_df.sort_values(["auroc", "accuracy"], ascending=[False, False]).iloc[0]
        best_by_budget[budget] = {
            "subset": best_row["subset"],
            "subset_name": best_row["subset_name"],
            "auroc": float(best_row["auroc"]),
            "brier": float(best_row["brier"]),
            "accuracy": float(best_row["accuracy"]),
        }
        greedy_row = next(step for step in greedy_results["path"] if step["budget"] == budget)
        print(
            f"- Budget {budget}: oracle={best_row['subset_name']} ({best_row['auroc']:.4f}) | "
            f"greedy={greedy_row['subset_name']} ({greedy_row['auroc']:.4f})"
        )

    plot_budget_curves(
        best_by_budget,
        greedy_results["path"],
        os.path.join(RESULTS_DIR, "utility_vs_budget.png"),
    )

    print_header("8. Layer Acquisition Frequency Analysis")
    top_k = 10
    layer_frequency = layer_frequency_in_top_k(oracle_df, top_k=top_k)
    for layer, count in sorted(layer_frequency.items(), key=lambda item: item[1], reverse=True):
        print(f"- {layer}: {count} appearances in top-{top_k} subsets")

    plot_layer_frequency(
        layer_frequency,
        top_k=top_k,
        output_path=os.path.join(RESULTS_DIR, "layer_frequency_top10.png"),
    )

    results_payload = {
        "random_state": RANDOM_STATE,
        "n_targets": int(len(feature_df)),
        "label_distribution": {
            "negative": int((eval_df["label"] == 0).sum()),
            "positive": int((eval_df["label"] == 1).sum()),
            "unlabeled": int(feature_df["label"].isna().sum()),
        },
        "feature_columns": LAYER_FEATURE_COLUMNS,
        "targets": feature_df[
            [
                "GeneSymbol",
                "label",
                "target_components",
                "layer_presence_vector",
                "in_snp",
                "snp_count",
                "in_cpg",
                "cpg_count",
                "in_deg",
                "deg_up_count",
                "deg_down_count",
                "deg_tissue_breadth",
                "in_hub",
                "hub_count",
                "in_mirna",
                "mirna_count",
                "mirna_up_count",
                "mirna_down_count",
                "in_sig",
                "sig_peripheral",
                "sig_brain_only",
                "sig_max_auc",
            ]
        ].to_dict(orient="records"),
        "baselines": baseline_results,
        "oracle_results": subset_results,
        "best_subset": {
            "subset": best_subset_row["subset"],
            "subset_name": best_subset_row["subset_name"],
            "budget": int(best_subset_row["budget"]),
            "auroc": float(best_subset_row["auroc"]),
            "brier": float(best_subset_row["brier"]),
            "accuracy": float(best_subset_row["accuracy"]),
        },
        "mean_auroc_by_budget": {str(int(k)): float(v) for k, v in mean_auroc_by_budget.items()},
        "greedy_acquisition": greedy_results,
        "best_by_budget": best_by_budget,
        "layer_frequency_top10": layer_frequency,
        "plots": {
            "oracle_mean_auroc_by_budget": os.path.join(RESULTS_DIR, "oracle_mean_auroc_by_budget.png"),
            "oracle_top10_subsets": os.path.join(RESULTS_DIR, "oracle_top10_subsets.png"),
            "utility_vs_budget": os.path.join(RESULTS_DIR, "utility_vs_budget.png"),
            "layer_frequency_top10": os.path.join(RESULTS_DIR, "layer_frequency_top10.png"),
        },
    }

    with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready(results_payload), f, indent=2)

    print_header("9. Output Files")
    print(f"Saved plots and JSON results under ./{RESULTS_DIR}/")
    for filename in sorted(os.listdir(RESULTS_DIR)):
        print(f"- {os.path.join(RESULTS_DIR, filename)}")


if __name__ == "__main__":
    main()
