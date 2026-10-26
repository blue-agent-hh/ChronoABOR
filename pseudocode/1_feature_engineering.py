ABOR_LAYERS = {
    "deg":       "degs.csv",
    "snp":       "snp.csv",
    "cpg":       "cpg.csv",
    "hubgene":   "hubgene.csv",
    "mirna":     "miRNA.csv",
    "signature": "signature.csv",
    "target":    "target.csv",
}


def build_gene_universe(abor_tables):
    universe = set()
    for layer_name, table in abor_tables.items():
        for row in table:
            genes = hgnc_normalize(row["Gene_symbol"])
            universe.update(genes)
    return sorted(universe)


def build_feature_vector(gene, abor_tables, cutoff_year, static_priors):
    features = {
        "is_secreted":  static_priors[gene]["is_secreted"],
        "is_ad_driver": static_priors[gene]["is_ad_driver"],
    }

    lfc_values = []
    evidence_count = 0
    deg_study_ids = set()
    layer_set = set()
    has_cpg = False
    has_snp = False
    n_up = 0
    n_down = 0

    for layer_name, table in abor_tables.items():
        for record in table:
            if gene not in hgnc_normalize(record["Gene_symbol"]):
                continue

            pub_year = resolve_publication_year(record)

            if pub_year > cutoff_year:
                continue

            evidence_count += 1
            layer_set.add(layer_name)

            if layer_name == "deg":
                lfc = abs(record.get("log_fold_change", 0))
                if lfc > 0:
                    lfc_values.append(lfc)
                deg_study_ids.add(record["study_id"])
                if "up" in record.get("direction", "").lower():
                    n_up += 1
                if "down" in record.get("direction", "").lower():
                    n_down += 1

            elif layer_name == "cpg":
                has_cpg = True

            elif layer_name == "snp":
                has_snp = True

    features["deg_lfc_mean"] = mean(lfc_values) if lfc_values else 0.0
    features["deg_lfc_max"]  = max(lfc_values)  if lfc_values else 0.0
    features["n_evidence_total"] = evidence_count
    features["num_deg_studies"]  = len(deg_study_ids)
    features["num_modalities"]   = len(layer_set)
    features["has_methylation"]  = int(has_cpg)
    features["has_snp"]          = int(has_snp)

    total_deg = n_up + n_down
    features["directionality_consistency"] = (
        abs(n_up - n_down) / total_deg if total_deg > 0 else 0.0
    )

    return features


def build_feature_matrix(gene_universe, abor_tables, cutoff_year, static_priors):
    rows = []
    for gene in gene_universe:
        fv = build_feature_vector(gene, abor_tables, cutoff_year, static_priors)
        rows.append(fv)
    return DataFrame(rows, index=gene_universe)
