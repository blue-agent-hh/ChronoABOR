#!/usr/bin/env python3

import json
import random
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SIGNATURE_PATH = BASE_DIR / "signature.csv"
OUTPUT_JSON = RESULTS_DIR / "nm_exp1_clinicaltrials.json"
OUTPUT_PLOT = RESULTS_DIR / "nm_exp1_plot.png"
API_URL = "https://clinicaltrials.gov/api/v2/studies"
RANDOM_SEED = 42
TOP_N = 50
NULL_N = 50
REQUEST_SLEEP_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 10
VALID_GENE_RE = re.compile(r"^[A-Z0-9-]+$")
INVALID_GENE_TOKENS = {"", "NA", "N/A", "NAN", "NONE", "NULL"}


def normalize_gene_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    tokens = []
    for part in re.split(r"[,;/]", str(raw_value).upper()):
        gene = part.strip()
        if gene in INVALID_GENE_TOKENS:
            continue
        if not VALID_GENE_RE.fullmatch(gene):
            continue
        tokens.append(gene)
    return sorted(set(tokens))


def load_signature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ABOR_ID", "AUC", "Tissue", "Gene_symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"signature.csv is missing required columns: {sorted(missing)}")
    df["AUC_num"] = pd.to_numeric(df["AUC"], errors="coerce")
    df["genes"] = df["Gene_symbol"].apply(normalize_gene_tokens)
    return df


def build_gene_scores(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    peripheral_mask = df["Tissue"].fillna("").astype(str).str.contains(
        "plasma|blood|serum", case=False, regex=True
    )
    score_rows = df.loc[peripheral_mask & df["AUC_num"].ge(0.80)].copy()
    gene_counts: dict[str, int] = {}
    for genes in score_rows["genes"]:
        for gene in genes:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    ranked = [
        {"gene": gene, "chronoabor_score_proxy": int(count)}
        for gene, count in sorted(gene_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    gene_universe = sorted({gene for genes in df["genes"] for gene in genes})
    return ranked, gene_universe


def choose_null_genes(top_genes: list[str], gene_universe: list[str]) -> tuple[list[str], bool]:
    rng = random.Random(RANDOM_SEED)
    top_gene_set = set(top_genes)
    candidates = [gene for gene in gene_universe if gene not in top_gene_set]
    if len(candidates) >= NULL_N:
        return sorted(rng.sample(candidates, NULL_N)), False
    sampled = [rng.choice(candidates) for _ in range(NULL_N)] if candidates else []
    return sampled, True


def fetch_trial_count(session: requests.Session, gene_symbol: str) -> tuple[int, dict | None]:
    params = {
        "query.cond": "Alzheimer",
        "query.intr": gene_symbol,
        "filter.overallStatus": "COMPLETED,RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": 10,
        "countTotal": "true",
    }
    try:
        response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        total_count = int(payload.get("totalCount", len(payload.get("studies", []))))
        return total_count, None
    except Exception as exc:
        return 0, {"gene": gene_symbol, "error": str(exc)}


def collect_trial_counts(
    ranked_gene_scores: dict[str, int], genes: list[str], group_name: str
) -> tuple[list[dict], list[dict]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "ChronoABOR-ClinicalTrials-Validation/1.0"})
    records: list[dict] = []
    errors: list[dict] = []
    for gene in genes:
        trial_count, error = fetch_trial_count(session, gene)
        records.append(
            {
                "gene": gene,
                "group": group_name,
                "chronoabor_score_proxy": int(ranked_gene_scores.get(gene, 0)),
                "trial_count": int(trial_count),
            }
        )
        if error:
            errors.append(error)
        time.sleep(REQUEST_SLEEP_SECONDS)
    return records, errors


def safe_mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def precision_at(records: list[dict], k: int) -> float:
    subset = records[:k]
    if not subset:
        return float("nan")
    return float(np.mean([record["trial_count"] >= 1 for record in subset]))


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def make_plot(top_records: list[dict]) -> None:
    top20 = top_records[:20]
    labels = [record["gene"] for record in top20]
    counts = [record["trial_count"] for record in top20]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, counts, color="#336699")
    ax.set_title("ClinicalTrials.gov counts for top ChronoABOR proxy genes")
    ax.set_xlabel("Gene symbol")
    ax.set_ylabel("Trial count")
    ax.tick_params(axis="x", rotation=65)
    for idx, count in enumerate(counts):
        ax.text(idx, count + 0.2, str(count), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_signature_table(SIGNATURE_PATH)
    ranked_scores, gene_universe = build_gene_scores(df)
    if len(ranked_scores) < TOP_N:
        raise ValueError(f"Only found {len(ranked_scores)} ranked genes; cannot build top-{TOP_N} set.")

    top_records_seed = ranked_scores[:TOP_N]
    top_genes = [record["gene"] for record in top_records_seed]
    null_genes, null_with_replacement = choose_null_genes(top_genes, gene_universe)
    if len(null_genes) < NULL_N:
        raise ValueError(f"Only found {len(null_genes)} null genes; cannot build null-{NULL_N} set.")

    ranked_lookup = {record["gene"]: record["chronoabor_score_proxy"] for record in ranked_scores}
    top_records, top_errors = collect_trial_counts(ranked_lookup, top_genes, "top50")
    null_records, null_errors = collect_trial_counts(ranked_lookup, null_genes, "null50")

    top_counts = [record["trial_count"] for record in top_records]
    null_counts = [record["trial_count"] for record in null_records]
    mean_top = safe_mean(top_counts)
    mean_null = safe_mean(null_counts)
    u_stat, p_value = mannwhitneyu(top_counts, null_counts, alternative="two-sided")
    enrichment_ratio = float(mean_top / mean_null) if mean_null and not np.isnan(mean_null) else float("inf")
    precision_10 = precision_at(top_records, 10)
    precision_25 = precision_at(top_records, 25)

    make_plot(top_records)

    results = {
        "random_seed": RANDOM_SEED,
        "api_url": API_URL,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "sleep_between_calls_seconds": REQUEST_SLEEP_SECONDS,
        "n_signature_rows": int(df.shape[0]),
        "n_unique_valid_genes": int(len(gene_universe)),
        "top_n": TOP_N,
        "null_n": NULL_N,
        "null_sampling_with_replacement": bool(null_with_replacement),
        "summary": {
            "top50_mean_trial_count": mean_top,
            "null50_mean_trial_count": mean_null,
            "mannwhitney_u": float(u_stat),
            "p_value": float(p_value),
            "enrichment_ratio": enrichment_ratio,
            "precision_at_10": precision_10,
            "precision_at_25": precision_25,
        },
        "top50": top_records,
        "null50": null_records,
        "api_errors": top_errors + null_errors,
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("Experiment 1: ClinicalTrials.gov validation")
    print(f"Top-50 mean trial count: {mean_top:.3f}")
    print(f"Null-50 mean trial count: {mean_null:.3f}")
    print(f"Enrichment ratio: {enrichment_ratio:.3f}" if np.isfinite(enrichment_ratio) else "Enrichment ratio: inf")
    print(f"Mann-Whitney U p-value: {p_value:.6g}")
    print(f"Precision@10: {precision_10:.3f}")
    print(f"Precision@25: {precision_25:.3f}")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
