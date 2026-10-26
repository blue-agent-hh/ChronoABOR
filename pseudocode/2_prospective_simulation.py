from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


FOLDS = {
    "C": {"cutoff": 2021, "label_window": (2021, 2024)},
    "D": {"cutoff": 2022, "label_window": (2022, 2025)},
}


def make_model(n_pos, n_neg):
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )


def assign_labels(gene_universe, nm_corpus, fold_def):
    labels = {}
    start, end = fold_def["label_window"]
    for gene in gene_universe:
        if gene in nm_corpus and start <= nm_corpus[gene]["first_year"] <= end:
            labels[gene] = 1
        else:
            labels[gene] = 0
    return labels


def run_prospective_fold(fold_name, fold_def, abor_tables, gene_universe,
                         nm_corpus, static_priors):
    T = fold_def["cutoff"]

    X = build_feature_matrix(gene_universe, abor_tables, cutoff_year=T,
                             static_priors=static_priors)

    y = assign_labels(gene_universe, nm_corpus, fold_def)

    has_evidence = [g for g in gene_universe
                    if X.loc[g, "n_evidence_total"] > 0]
    X_eval = X.loc[has_evidence]
    y_eval = [y[g] for g in has_evidence]

    n_pos = sum(y_eval)
    n_neg = len(y_eval) - n_pos

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    all_proba = {}

    for train_idx, test_idx in cv.split(X_eval, y_eval):
        model = make_model(n_pos, n_neg)
        model.fit(X_eval.iloc[train_idx], [y_eval[i] for i in train_idx])
        proba = model.predict_proba(X_eval.iloc[test_idx])[:, 1]

        for i, idx in enumerate(test_idx):
            gene = has_evidence[idx]
            all_proba[gene] = proba[i]

        aurocs.append(roc_auc_score(
            [y_eval[i] for i in test_idx], proba
        ))

    ranking = sorted(all_proba.items(), key=lambda x: -x[1])

    auroc_mean = mean(aurocs)
    auroc_std  = std(aurocs)
    p_at_10    = precision_at_k(ranking, y, k=10)
    ef_at_10   = p_at_10 / (n_pos / len(has_evidence))

    return {
        "fold": fold_name,
        "n_total": len(has_evidence),
        "n_pos": n_pos,
        "auroc": f"{auroc_mean:.3f} ± {auroc_std:.3f}",
        "P@10": p_at_10,
        "EF@10": ef_at_10,
        "ranking": ranking,
    }


def precision_at_k(ranking, labels, k):
    top_k_genes = [gene for gene, _ in ranking[:k]]
    return sum(labels[g] for g in top_k_genes) / k


def main():
    abor_tables = load_abor_tables()
    gene_universe = build_gene_universe(abor_tables)
    nm_corpus = load_nm_corpus()
    static_priors = load_static_priors(gene_universe)

    for fold_name, fold_def in FOLDS.items():
        result = run_prospective_fold(
            fold_name, fold_def, abor_tables,
            gene_universe, nm_corpus, static_priors
        )
        print(f"Fold {fold_name}: AUROC={result['auroc']}, "
              f"P@10={result['P@10']:.3f}, EF@10={result['EF@10']:.1f}x")
