def within_secreted_evaluation(full_ranking, labels, secreted_genes, n_bootstrap=1000):
    secreted_ranking = [(g, s) for g, s in full_ranking if g in secreted_genes]
    secreted_labels  = {g: labels[g] for g, _ in secreted_ranking}

    n_secreted = len(secreted_ranking)
    n_pos      = sum(secreted_labels.values())

    y_true = [secreted_labels[g] for g, _ in secreted_ranking]
    y_score = [s for _, s in secreted_ranking]

    auroc  = roc_auc_score(y_true, y_score)
    p_at_5  = precision_at_k(secreted_ranking, secreted_labels, k=5)
    p_at_10 = precision_at_k(secreted_ranking, secreted_labels, k=10)

    auroc_ci = bootstrap_auroc(y_true, y_score, n_bootstrap)

    baselines = {
        "pub_count":    rank_by_feature(secreted_genes, "n_evidence_total"),
        "sig_max_auc":  rank_by_feature(secreted_genes, "sig_max_auc"),
        "prior_only":   rank_by_priors(secreted_genes),
        "temporal_only": rank_by_temporal_features(secreted_genes),
    }

    for name, baseline_ranking in baselines.items():
        delta, p_value = paired_bootstrap_test(
            secreted_ranking, baseline_ranking, secreted_labels, n_bootstrap
        )
        print(f"  vs {name}: ΔAUROC={delta:+.3f}, p={p_value:.4f}")

    return {
        "auroc": auroc,
        "auroc_95ci": auroc_ci,
        "P@5": p_at_5,
        "P@10": p_at_10,
    }


def jackknife_leave_one_positive_out(secreted_ranking, secreted_labels):
    positives = [g for g, y in secreted_labels.items() if y == 1]
    jackknife_aurocs = []

    for held_out in positives:
        subset_labels = {g: y for g, y in secreted_labels.items() if g != held_out}
        subset_ranking = [(g, s) for g, s in secreted_ranking if g != held_out]

        y_true = [subset_labels[g] for g, _ in subset_ranking]
        y_score = [s for _, s in subset_ranking]
        jackknife_aurocs.append(roc_auc_score(y_true, y_score))

    return {
        "mean": mean(jackknife_aurocs),
        "std": std(jackknife_aurocs),
        "per_gene": dict(zip(positives, jackknife_aurocs)),
    }


def rolling_origin_evaluation(abor_tables, secreted_genes, nm_corpus,
                               static_priors, cutoffs=[2019, 2020, 2021]):
    results_per_cutoff = []

    for T in cutoffs:
        X = build_feature_matrix(secreted_genes, abor_tables,
                                 cutoff_year=T, static_priors=static_priors)
        model = train_model(X, nm_corpus_labels)
        ranking = rank_genes(model, X)

        auroc = evaluate_auroc(ranking, nm_corpus_labels)
        p_at_10 = precision_at_k(ranking, nm_corpus_labels, k=10)
        results_per_cutoff.append({"cutoff": T, "auroc": auroc, "P@10": p_at_10})

    pooled_delta = pooled_paired_bootstrap(results_per_cutoff, n_resamples=1000)

    return results_per_cutoff, pooled_delta
