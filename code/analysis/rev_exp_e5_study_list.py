#!/usr/bin/env python3

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd

from nm_exp4_full_universe import normalize_gene_tokens
from rev_exp_common import (
    ALL_LAYER_SPECS,
    BASE_DIR,
    RESULTS_DIR,
    annotate_tables_with_study_metadata,
    ensure_results_dir,
    load_all_tables,
    provenance,
    to_jsonable,
    write_json,
)

SCRIPT_NAME = Path(__file__).name
OUTPUT_JSON = RESULTS_DIR / f"{Path(__file__).stem}.json"
OUTPUT_CSV = RESULTS_DIR / f"{Path(__file__).stem}.csv"
YEAR_SOURCE_PRIORITY = {
    "direct_year": 0,
    "pmid_lookup": 1,
    "doi_extract": 2,
    "default_2018": 3,
}
MANUSCRIPT_CLAIMS = {
    "abstract_expert_annotated_entries": 4576,
    "abstract_peer_reviewed_studies": 243,
    "supplementary_table_b2_deg_records": 1052,
    "supplementary_table_b2_total_after_dedup": 2245,
}


def update_best_year(existing: tuple[int, str] | None, new_year: int, new_source: str) -> tuple[int, str]:
    if existing is None:
        return new_year, new_source
    old_year, old_source = existing
    old_priority = YEAR_SOURCE_PRIORITY.get(old_source, 999)
    new_priority = YEAR_SOURCE_PRIORITY.get(new_source, 999)
    if new_priority < old_priority:
        return new_year, new_source
    if new_priority == old_priority and new_year < old_year:
        return new_year, new_source
    return existing


def main() -> None:
    start_time = time.time()
    ensure_results_dir()

    tables = load_all_tables()
    annotated_tables, _, pmid_conflicts = annotate_tables_with_study_metadata(tables)

    studies: dict[str, dict] = {}
    raw_rows_total = 0
    expanded_gene_records_total = 0
    unique_layer_gene_pairs: set[tuple[str, str]] = set()
    unique_layer_gene_study_records: set[tuple[str, str, str]] = set()
    unique_gene_study_records: set[tuple[str, str]] = set()
    union_gene_set: set[str] = set()
    per_layer_ground_truth: dict[str, dict] = {}

    for layer, spec in ALL_LAYER_SPECS.items():
        df = annotated_tables[layer]
        raw_rows_total += len(df)
        layer_unique_genes: set[str] = set()
        layer_expanded_gene_records = 0
        layer_unique_layer_gene_study_records: set[tuple[str, str, str]] = set()

        for _, row in df.iterrows():
            study_id = row["_study_id"]
            id_type = row["_study_id_type"]
            year = int(row["_resolved_year"])
            year_source = str(row["_year_resolution_source"])
            genes = normalize_gene_tokens(row[spec["gene_column"]])

            layer_expanded_gene_records += len(genes)
            for gene in genes:
                layer_unique_genes.add(gene)
                union_gene_set.add(gene)
                unique_layer_gene_pairs.add((layer, gene))
                unique_layer_gene_study_records.add((layer, gene, study_id))
                layer_unique_layer_gene_study_records.add((layer, gene, study_id))
                unique_gene_study_records.add((gene, study_id))
            study_record = studies.setdefault(
                study_id,
                {
                    "study_id": study_id,
                    "id_type": id_type,
                    "best_year": None,
                    "year_source": None,
                    "all_year_sources": set(),
                    "layers": set(),
                    "n_records": 0,
                    "n_gene_records": 0,
                    "raw_ids": set(),
                },
            )
            if study_record["id_type"] != id_type:
                id_type_preference = {"PMID": 0, "DOI": 1, "URL": 2}
                current_rank = id_type_preference.get(study_record["id_type"], 99)
                new_rank = id_type_preference.get(id_type, 99)
                if new_rank < current_rank:
                    study_record["id_type"] = id_type
            best_year = update_best_year(study_record["best_year"], year, year_source)
            study_record["best_year"] = best_year
            study_record["year_source"] = best_year[1]
            study_record["all_year_sources"].add(year_source)
            study_record["layers"].add(layer)
            study_record["n_records"] += 1
            study_record["n_gene_records"] += len(genes)
            study_record["raw_ids"].add(str(row[spec["study_column"]]))

        per_layer_ground_truth[layer] = {
            "filename": spec["filename"],
            "raw_row_count": int(len(df)),
            "unique_study_count": int(df["_study_id"].nunique()),
            "unique_gene_count": int(len(layer_unique_genes)),
            "expanded_gene_record_count": int(layer_expanded_gene_records),
            "deduplicated_layer_gene_study_count": int(len(layer_unique_layer_gene_study_records)),
            "pmid_rows": int(df["_study_id_type"].eq("PMID").sum()),
            "doi_rows": int(df["_study_id_type"].eq("DOI").sum()),
            "url_rows": int(df["_study_id_type"].eq("URL").sum()),
            "year_resolution_source_counts": df["_year_resolution_source"].value_counts().to_dict(),
        }

    final_study_rows = []
    year_histogram: Counter[int] = Counter()
    unresolved_study_count = 0
    non_pmid_url_study_count = 0
    non_pmid_url_row_count = 0

    for study_id, record in studies.items():
        if record["best_year"] is None:
            raise RuntimeError(f"Study {study_id} never received a resolved year.")
        year, year_source = record["best_year"]
        year_histogram[year] += 1
        if year_source == "default_2018":
            unresolved_study_count += 1
        if record["id_type"] == "URL":
            non_pmid_url_study_count += 1
            non_pmid_url_row_count += record["n_records"]
        final_study_rows.append(
            {
                "study_id": study_id,
                "id_type": record["id_type"],
                "year": int(year),
                "layers": ";".join(sorted(record["layers"])),
                "n_records": int(record["n_records"]),
            }
        )

    study_df = pd.DataFrame(final_study_rows).sort_values(
        by=["year", "id_type", "study_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    study_df.to_csv(OUTPUT_CSV, index=False)

    ground_truth_totals = {
        "raw_row_total": int(raw_rows_total),
        "unique_study_total": int(len(studies)),
        "unique_gene_union_total": int(len(union_gene_set)),
        "expanded_gene_record_total": int(expanded_gene_records_total or sum(item["expanded_gene_record_count"] for item in per_layer_ground_truth.values())),
        "deduplicated_layer_gene_pair_total": int(len(unique_layer_gene_pairs)),
        "deduplicated_layer_gene_study_total": int(len(unique_layer_gene_study_records)),
        "deduplicated_gene_study_total_cross_layer": int(len(unique_gene_study_records)),
    }
    manuscript_delta = {
        key: ground_truth_totals.get(
            "raw_row_total" if key == "abstract_expert_annotated_entries" else "unique_study_total",
            None,
        )
        for key in MANUSCRIPT_CLAIMS
    }
    manuscript_delta = {
        "abstract_expert_annotated_entries": {
            "claimed": MANUSCRIPT_CLAIMS["abstract_expert_annotated_entries"],
            "observed": ground_truth_totals["raw_row_total"],
            "delta_observed_minus_claimed": ground_truth_totals["raw_row_total"]
            - MANUSCRIPT_CLAIMS["abstract_expert_annotated_entries"],
        },
        "abstract_peer_reviewed_studies": {
            "claimed": MANUSCRIPT_CLAIMS["abstract_peer_reviewed_studies"],
            "observed": ground_truth_totals["unique_study_total"],
            "delta_observed_minus_claimed": ground_truth_totals["unique_study_total"]
            - MANUSCRIPT_CLAIMS["abstract_peer_reviewed_studies"],
        },
        "supplementary_table_b2_deg_records": {
            "claimed": MANUSCRIPT_CLAIMS["supplementary_table_b2_deg_records"],
            "observed": per_layer_ground_truth["deg"]["raw_row_count"],
            "delta_observed_minus_claimed": per_layer_ground_truth["deg"]["raw_row_count"]
            - MANUSCRIPT_CLAIMS["supplementary_table_b2_deg_records"],
        },
        "supplementary_table_b2_total_after_dedup": {
            "claimed": MANUSCRIPT_CLAIMS["supplementary_table_b2_total_after_dedup"],
            "observed_primary_definition": ground_truth_totals["deduplicated_layer_gene_study_total"],
            "observed_cross_layer_gene_study_definition": ground_truth_totals["deduplicated_gene_study_total_cross_layer"],
            "delta_primary_minus_claimed": ground_truth_totals["deduplicated_layer_gene_study_total"]
            - MANUSCRIPT_CLAIMS["supplementary_table_b2_total_after_dedup"],
            "delta_cross_layer_minus_claimed": ground_truth_totals["deduplicated_gene_study_total_cross_layer"]
            - MANUSCRIPT_CLAIMS["supplementary_table_b2_total_after_dedup"],
        },
    }

    results = {
        **provenance(
            SCRIPT_NAME,
            parameters={
                "study_layers": list(ALL_LAYER_SPECS),
                "year_resolution_rule": "nm_exp4_full_universe.resolve_year applied after canonical PMID/DOI/URL normalization",
                "url_handling": "target.csv URLs were canonicalized; PubMed URLs were collapsed to PMID, DOI URLs to DOI, all others remained URL identifiers",
                "deduplicated_entry_primary_definition": "unique (layer, gene, study_id) tuples after gene expansion and canonical study-id deduplication",
                "deduplicated_entry_secondary_definition": "unique (gene, study_id) tuples collapsed across layers after gene expansion",
            },
            input_paths=[BASE_DIR / spec["filename"] for spec in ALL_LAYER_SPECS.values()],
        ),
        "outputs": {
            "json": str(OUTPUT_JSON),
            "study_csv": str(OUTPUT_CSV),
        },
        "year_range": {
            "min_year": int(study_df["year"].min()),
            "max_year": int(study_df["year"].max()),
        },
        "year_histogram": {str(year): int(count) for year, count in sorted(year_histogram.items())},
        "unresolved_study_count": int(unresolved_study_count),
        "non_pmid_url_study_count": int(non_pmid_url_study_count),
        "non_pmid_url_row_count": int(non_pmid_url_row_count),
        "unique_studies_per_layer": {
            layer: per_layer_ground_truth[layer]["unique_study_count"] for layer in ALL_LAYER_SPECS
        },
        "ground_truth_totals": ground_truth_totals,
        "per_layer_ground_truth": per_layer_ground_truth,
        "manuscript_claim_comparison": manuscript_delta,
        "signature_pmid_year_conflicts": pmid_conflicts,
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, results)

    print(
        f"E5 study list: {ground_truth_totals['unique_study_total']} unique canonical studies, "
        f"years {study_df['year'].min()}-{study_df['year'].max()}, "
        f"{unresolved_study_count} unresolved-to-default-2018."
    )
    print(
        f"Rows: raw={ground_truth_totals['raw_row_total']}, "
        f"dedup(layer,gene,study)={ground_truth_totals['deduplicated_layer_gene_study_total']}, "
        f"dedup(gene,study)={ground_truth_totals['deduplicated_gene_study_total_cross_layer']}."
    )
    print(
        f"Manuscript deltas: entries {manuscript_delta['abstract_expert_annotated_entries']['delta_observed_minus_claimed']:+d}, "
        f"studies {manuscript_delta['abstract_peer_reviewed_studies']['delta_observed_minus_claimed']:+d}, "
        f"DEG rows {manuscript_delta['supplementary_table_b2_deg_records']['delta_observed_minus_claimed']:+d}."
    )
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
