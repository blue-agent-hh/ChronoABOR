#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from nm_exp4_full_universe import (
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    load_tables,
    to_jsonable,
)
from nm_exp_p1_prior_control import SECRETED_GENES, assign_labels, build_first_peripheral_year

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TARGET_CSV = BASE_DIR / "target.csv"
LOG_FILE = RESULTS_DIR / "nm_exp_gemini_baseline.log"
CACHE_JSON = RESULTS_DIR / "nm_exp_gemini_cache.json"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_gemini_baseline.json"

FEATURE_CUTOFF_YEAR = 2021
TOTAL_GENES_EXPECTED = 2843
PROGRESS_EVERY = 50
SAVE_EVERY = 100

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PROXIES = None

MODEL_SPECS = [
    {"name": "gemini-2.5-flash", "sleep_seconds": 0.3},
    {"name": "gemini-2.5-pro", "sleep_seconds": 0.8},
]

NM_CORPUS_POSITIVES = [
    "NEFL",
    "GFAP",
    "MAPT",
    "APOE",
    "APP",
    "CLU",
    "VGF",
    "NRGN",
    "SNAP25",
    "YWHAZ",
    "SYT1",
    "NEFH",
    "NEFM",
    "GAP43",
    "ACHE",
    "VILIP1",
    "HSPB1",
    "PKM",
    "ALDOA",
    "CFL1",
    "TMOD2",
    "BASP1",
    "STMN2",
    "STMN1",
    "MAP2",
    "TUBB3",
    "ENO2",
    "NSE",
    "CAMK2A",
    "CAMK2B",
    "CAMK4",
    "BDNF",
]

logger = logging.getLogger("nm_exp_gemini_baseline")


def stdout_targets_log_file(path: Path) -> bool:
    try:
        stdout_stat = os.fstat(sys.stdout.fileno())
        log_stat = path.stat()
    except OSError:
        return False
    return (stdout_stat.st_dev, stdout_stat.st_ino) == (log_stat.st_dev, log_stat.st_ino)


def setup_logging() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not stdout_targets_log_file(LOG_FILE):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)


def write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    cache: dict[str, dict[str, float]] = {}
    for model_name, model_scores in raw.items():
        if not isinstance(model_scores, dict):
            continue
        parsed_scores: dict[str, float] = {}
        for gene, score in model_scores.items():
            try:
                parsed_scores[str(gene)] = float(score)
            except (TypeError, ValueError):
                continue
        cache[str(model_name)] = parsed_scores
    return cache


def save_cache(path: Path, cache: dict[str, dict[str, float]]) -> None:
    write_json(path, cache)


def build_data() -> tuple[dict, pd.DataFrame]:
    tables = load_tables()
    target_df = pd.read_csv(TARGET_CSV)
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    if len(gene_universe) != TOTAL_GENES_EXPECTED:
        raise RuntimeError(
            f"Expected {TOTAL_GENES_EXPECTED} genes in the universe, found {len(gene_universe)}."
        )

    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)

    first_peripheral_year = build_first_peripheral_year(tables["signature"])
    label_df = assign_labels(gene_universe, evidence_by_cutoff, first_peripheral_year)

    full_df = feature_df.merge(label_df, on="gene", how="left").sort_values("gene").reset_index(drop=True)
    full_df["label_external"] = full_df["gene"].isin(set(NM_CORPUS_POSITIVES)).astype(int)
    full_df["label_internal"] = full_df["label"].eq(1).astype(int)
    full_df["anonymous_gene_id"] = [f"[ANONYMOUS_GENE_ID_{index}]" for index in range(1, len(full_df) + 1)]

    data_summary = {
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "n_genes_universe": int(len(full_df)),
        "n_genes_with_2021_evidence": int(len(evidence_by_cutoff)),
        "n_internal_positives": int(full_df["label_internal"].sum()),
        "n_external_positives": int(full_df["label_external"].sum()),
        "n_nm_requested": int(len(NM_CORPUS_POSITIVES)),
        "nm_genes_in_universe": sorted(set(NM_CORPUS_POSITIVES) & set(full_df["gene"])),
        "nm_genes_missing_from_universe": sorted(set(NM_CORPUS_POSITIVES) - set(full_df["gene"])),
        "target_csv_rows": int(len(target_df)),
        "target_csv_columns": list(target_df.columns),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
    }
    return data_summary, full_df


def build_evidence_card(row: pd.Series) -> str:
    return (
        f"Gene: {row['anonymous_gene_id']}\n"
        f"Genomic: SNP={int(row['snp_count'])}, CpG={int(row['cpg_count'])}, "
        f"DEG={int(row['deg_count'])}(up={int(row['deg_up_count'])},down={int(row['deg_down_count'])}), "
        f"Hub={int(row['hub_count'])}, miRNA={int(row['mirna_count'])}\n"
        f"Signature: total_studies={int(row['sig_count_all'])}, peripheral={int(row['sig_count_peripheral'])}, "
        f"max_AUC={float(row['sig_max_auc']):.3f}, high_AUC_studies={int(row['sig_count_high_auc'])}\n"
        f"Prior: secreted={int(row['is_secreted'])}, layers={int(row['layer_count'])}, "
        f"num_studies={int(row['num_studies'])}"
    )


def build_generation_config(model_name: str) -> dict:
    config = {"temperature": 0.0, "maxOutputTokens": 15}
    if model_name == "gemini-2.5-flash":
        config["thinkingConfig"] = {"thinkingBudget": 0}
    elif model_name == "gemini-2.5-pro":
        config["maxOutputTokens"] = 256
        config["thinkingConfig"] = {"thinkingBudget": 128}
    return config


def query_gemini(model_name: str, gene: str, card: str, max_retries: int = 3) -> float | None:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={GEMINI_API_KEY}"
    )
    prompt = f"Rate Alzheimer disease blood biomarker potential 0-100. Respond with ONLY a number.\n\n{card}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": build_generation_config(model_name),
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, proxies=PROXIES, timeout=30)
            if response.status_code == 200:
                try:
                    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as exc:
                    logger.error("ERROR gene %s attempt %d: malformed response %s", gene, attempt, exc)
                    return None

                match = re.search(r"\d+(?:\.\d+)?", text)
                if match is None:
                    logger.error("ERROR gene %s attempt %d: no numeric score", gene, attempt)
                    return None
                return float(match.group())

            logger.error("ERROR gene %s attempt %d: status %s", gene, attempt, response.status_code)
            if response.status_code == 429:
                time.sleep(10 * attempt)
                continue
            return None
        except Exception as exc:
            logger.error("ERROR gene %s attempt %d: exception %s", gene, attempt, exc)
            time.sleep(3)

    return None


def score_gene(model_name: str, row: pd.Series) -> float:
    gene = str(row["gene"])
    card = build_evidence_card(row)

    for round_index in range(1, 3):
        score = query_gemini(model_name=model_name, gene=gene, card=card, max_retries=3)
        if score is not None:
            return float(score)
        if round_index < 2:
            time.sleep(5 * round_index)

    logger.error("ERROR gene %s: exhausted retries for %s", gene, model_name)
    raise RuntimeError(f"Failed to score gene {gene} for model {model_name}.")


def sort_ranked_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["score", "num_studies", "layer_count", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def safe_auroc(labels: pd.Series, scores: pd.Series) -> float | None:
    if int(labels.nunique()) < 2:
        return None
    return float(roc_auc_score(labels.astype(int), scores.astype(float)))


def precision_at_k(ranked_df: pd.DataFrame, label_column: str, k: int) -> float | None:
    if len(ranked_df) < k:
        return None
    return float(ranked_df.head(k)[label_column].mean())


def enrichment_factor_at_k(ranked_df: pd.DataFrame, label_column: str, k: int) -> float | None:
    precision = precision_at_k(ranked_df, label_column, k)
    base_rate = float(ranked_df[label_column].mean())
    if precision is None or base_rate <= 0:
        return None
    return float(precision / base_rate)


def evaluate_model(full_df: pd.DataFrame, model_name: str, sleep_seconds: float, cache: dict[str, dict[str, float]]) -> dict:
    model_cache = cache.get(model_name, {})
    missing_genes = sorted(set(full_df["gene"]) - set(model_cache))
    if missing_genes:
        raise RuntimeError(f"Model {model_name} is missing {len(missing_genes)} scored genes.")

    scored_df = full_df.copy()
    scored_df["score"] = scored_df["gene"].map(model_cache).astype(float)
    ranked_df = sort_ranked_df(scored_df)

    return {
        "model_name": model_name,
        "sleep_seconds": sleep_seconds,
        "n_scored_genes": int(len(scored_df)),
        "n_internal_positives": int(scored_df["label_internal"].sum()),
        "n_external_positives": int(scored_df["label_external"].sum()),
        "auroc_external": safe_auroc(scored_df["label_external"], scored_df["score"]),
        "auroc_internal": safe_auroc(scored_df["label_internal"], scored_df["score"]),
        "p_at_10": precision_at_k(ranked_df, "label_external", 10),
        "p_at_20": precision_at_k(ranked_df, "label_external", 20),
        "ef_at_10": enrichment_factor_at_k(ranked_df, "label_external", 10),
        "ef_at_20": enrichment_factor_at_k(ranked_df, "label_external", 20),
        "top_10_genes": ranked_df.head(10)["gene"].tolist(),
    }


def save_results(data_summary: dict, model_results: list[dict], start_time: float) -> None:
    payload = {
        "data_summary": data_summary,
        "models": model_results,
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, payload)


def score_model(
    model_name: str,
    score_df: pd.DataFrame,
    sleep_seconds: float,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    gene_set = set(score_df["gene"])
    existing_cache = cache.setdefault(model_name, {})
    model_cache = {gene: score for gene, score in existing_cache.items() if gene in gene_set}
    cache[model_name] = model_cache

    total = len(score_df)
    if len(model_cache) > total:
        raise RuntimeError(f"Cache for {model_name} has more entries than the scoring universe.")

    logger.info("%s cache: %d/%d genes already scored", model_name, len(model_cache), total)

    writes_since_flush = 0
    missing_df = score_df.loc[~score_df["gene"].isin(model_cache)].reset_index(drop=True)
    for _, row in missing_df.iterrows():
        gene = str(row["gene"])
        score = score_gene(model_name=model_name, row=row)
        model_cache[gene] = score
        writes_since_flush += 1

        completed = len(model_cache)
        if completed % PROGRESS_EVERY == 0 or completed == total:
            logger.info(
                "Model %s: scored %d/%d (gene: %s -> %.1f)",
                model_name,
                completed,
                total,
                gene,
                score,
            )

        if writes_since_flush >= SAVE_EVERY:
            save_cache(CACHE_JSON, cache)
            writes_since_flush = 0

        if completed < total:
            time.sleep(sleep_seconds)

    if writes_since_flush > 0 or not missing_df.empty:
        save_cache(CACHE_JSON, cache)

    return model_cache


def main() -> None:
    start_time = time.time()
    setup_logging()

    logger.info("Preparing Gemini baseline data")
    data_summary, full_df = build_data()
    cache = load_cache(CACHE_JSON)
    logger.info(
        "Prepared data: universe=%d internal_pos=%d external_pos=%d/%d",
        data_summary["n_genes_universe"],
        data_summary["n_internal_positives"],
        data_summary["n_external_positives"],
        data_summary["n_nm_requested"],
    )

    model_results: list[dict] = []
    for spec in MODEL_SPECS:
        model_name = str(spec["name"])
        sleep_seconds = float(spec["sleep_seconds"])

        logger.info("=== Starting %s ===", model_name)
        score_model(
            model_name=model_name,
            score_df=full_df,
            sleep_seconds=sleep_seconds,
            cache=cache,
        )

        metrics = evaluate_model(
            full_df=full_df,
            model_name=model_name,
            sleep_seconds=sleep_seconds,
            cache=cache,
        )
        model_results.append(metrics)
        save_results(data_summary, model_results, start_time)

        logger.info(
            "%s AUROC_ext=%.3f AUROC_int=%.3f P@10=%.2f P@20=%.2f EF@10=%.2f EF@20=%.2f",
            model_name,
            metrics["auroc_external"],
            metrics["auroc_internal"],
            metrics["p_at_10"],
            metrics["p_at_20"],
            metrics["ef_at_10"],
            metrics["ef_at_20"],
        )
        logger.info("%s top_10=%s", model_name, ", ".join(metrics["top_10_genes"]))

    logger.info("Wrote results to %s", OUTPUT_JSON)
    logger.info("Wrote cache to %s", CACHE_JSON)


if __name__ == "__main__":
    main()
