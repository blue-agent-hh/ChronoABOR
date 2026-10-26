def compute_enrichment_score(ranked_genes, hit_set, weight_exponent=1.0):
    N = len(ranked_genes)
    hits_in_ranking = hit_set & set(ranked_genes)
    N_H = len(hits_in_ranking)
    N_miss = N - N_H

    if N_H == 0:
        return 0.0, []

    hit_weights = {}
    for rank, gene in enumerate(ranked_genes):
        if gene in hits_in_ranking:
            hit_weights[gene] = (N - rank) ** weight_exponent

    weight_sum = sum(hit_weights.values())

    running_sum = 0.0
    max_es = 0.0
    peak_idx = 0
    running_sums = []

    for i, gene in enumerate(ranked_genes):
        if gene in hits_in_ranking:
            running_sum += hit_weights[gene] / weight_sum
        else:
            running_sum -= 1.0 / N_miss
        running_sums.append(running_sum)

        if abs(running_sum) > abs(max_es):
            max_es = running_sum
            peak_idx = i

    leading_edge = [g for g in ranked_genes[:peak_idx + 1] if g in hits_in_ranking]

    return max_es, leading_edge


def random_permutation_test(ranked_genes, hit_set, observed_es, n_perm=1000):
    null_scores = []
    all_genes = list(set(ranked_genes))

    for _ in range(n_perm):
        random_hits = random_sample(all_genes, size=len(hit_set))
        es, _ = compute_enrichment_score(ranked_genes, set(random_hits))
        null_scores.append(es)

    mean_null = mean([abs(s) for s in null_scores])
    nes = observed_es / mean_null if mean_null > 0 else 0.0
    p_value = sum(1 for s in null_scores if s >= observed_es) / n_perm

    return nes, p_value


def matched_null_permutation_test(ranked_genes, hit_set, observed_es,
                                   gene_features, n_perm=1000):
    non_hits = [g for g in ranked_genes if g not in hit_set]
    null_scores = []

    for _ in range(n_perm):
        matched_set = set()
        for hit_gene in hit_set:
            candidates = [
                g for g in non_hits
                if g not in matched_set
                and gene_features[g]["is_secreted"] == gene_features[hit_gene]["is_secreted"]
                and (gene_features[g]["sig_count_all"] > 0) == (gene_features[hit_gene]["sig_count_all"] > 0)
            ]
            if candidates:
                best = min(candidates, key=lambda g:
                    abs(gene_features[g]["num_studies"] - gene_features[hit_gene]["num_studies"]))
                matched_set.add(best)

        es, _ = compute_enrichment_score(ranked_genes, matched_set)
        null_scores.append(es)

    mean_null = mean([abs(s) for s in null_scores])
    nes = observed_es / mean_null if mean_null > 0 else 0.0
    p_value = sum(1 for s in null_scores if s >= observed_es) / n_perm

    return nes, p_value


def main():
    ranked_genes = load_chronoabor_ranking(cutoff=2021)
    gene_features = load_gene_features()

    hit_sets = {
        "Indiana/Knight upregulated": load_iadrc_knight_hits(),
        "Frick incident-LOAD":        load_frick_hits(),
    }

    for name, hit_set in hit_sets.items():
        es, leading_edge = compute_enrichment_score(ranked_genes, hit_set)

        nes_rand, p_rand = random_permutation_test(
            ranked_genes, hit_set, es, n_perm=1000
        )

        nes_matched, p_matched = matched_null_permutation_test(
            ranked_genes, hit_set, es, gene_features, n_perm=1000
        )

        print(f"{name}:")
        print(f"  Random null:  NES={nes_rand:.3f}, p={p_rand:.3f}")
        print(f"  Matched null: NES={nes_matched:.3f}, p={p_matched:.3f}")
        print(f"  Leading edge: {leading_edge[:10]}")
