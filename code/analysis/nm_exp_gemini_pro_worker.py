#!/usr/bin/env python3

from __future__ import annotations

import fcntl
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
LOG_FILE = RESULTS_DIR / "nm_exp_gemini_pro_worker.log"
CACHE_FILE = RESULTS_DIR / "nm_exp_gemini_cache.json"
CACHE_LOCK_FILE = RESULTS_DIR / "nm_exp_gemini_cache.json.lock"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_gemini_pro_results.json"

FEATURE_CUTOFF_YEAR = 2021
TOTAL_GENES_EXPECTED = 2843
MODEL_NAME = "gemini-2.5-pro"
PROGRESS_EVERY = 5
SAVE_EVERY = 5
SLEEP_SECONDS = 2.0

GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY")
PROXIES = None

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

logger = logging.getLogger("nm_exp_gemini_pro_worker")


class FlushFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


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

    file_handler = FlushFileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not stdout_targets_log_file(LOG_FILE):
        stdout_handler = FlushStreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)


def write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(to_jsonable(payload), indent=2, sort_keys=True)
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def parse_cache(raw: object) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        return {}

    cache: dict[str, dict[str, float]] = {}
    for model_name, scores in raw.items():
        if not isinstance(scores, dict):
            continue
        parsed_scores: dict[str, float] = {}
        for gene, score in scores.items():
            try:
                parsed_scores[str(gene)] = float(score)
            except (TypeError, ValueError):
                continue
        cache[str(model_name)] = parsed_scores
    return cache


def read_cache_file() -> dict[str, dict[str, float]]:
    if not CACHE_FILE.exists():
        return {}

    try:
        with open(CACHE_FILE, encoding="utf-8") as handle:
            return parse_cache(json.load(handle))
    except Exception:
        return {}


def load_cache() -> dict[str, dict[str, float]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_LOCK_FILE.touch(exist_ok=True)

    with open(CACHE_LOCK_FILE, "r+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_SH)
        cache = read_cache_file()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return cache


def save_cache(model_scores: dict[str, float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_LOCK_FILE.touch(exist_ok=True)

    with open(CACHE_LOCK_FILE, "r+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        cache = read_cache_file()
        cache[MODEL_NAME] = {gene: float(score) for gene, score in sorted(model_scores.items())}
        write_json(CACHE_FILE, cache)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


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


def query_gemini_pro(gene: str, card: str, max_retries: int = 3) -> float | None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_NAME}:generateContent?key={GEMINI_API_KEY_2}"
    )
    prompt = f"Rate Alzheimer disease blood biomarker potential 0-100. Respond ONLY with a number.\n\n{card}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, proxies=PROXIES, timeout=(10, 180))
            if response.status_code == 200:
                try:
                    parts = response.json()["candidates"][0]["content"].get("parts", [])
                except Exception as exc:
                    logger.error("%s malformed response attempt %d: %s", gene, attempt + 1, exc)
                    return None
                for part in parts:
                    if part.get("thought"):
                        continue
                    text = part.get("text")
                    if not text:
                        continue
                    match = re.search(r"\d+(?:\.\d+)?", text)
                    if match is None:
                        logger.error("%s non-numeric response attempt %d: %s", gene, attempt + 1, text[:100])
                        return None
                    return float(match.group())
                logger.error("%s missing text parts attempt %d", gene, attempt + 1)
                return None
            if response.status_code == 429:
                logger.error("%s rate-limited attempt %d", gene, attempt + 1)
                time.sleep(15 * (attempt + 1))
                continue

            logger.error("%s API %s: %s", gene, response.status_code, response.text[:100])
            return None
        except Exception as exc:
            logger.error("%s exception attempt %d: %s", gene, attempt + 1, exc)
            time.sleep(5)

    return None


def score_gene(row: pd.Series) -> float:
    gene = str(row["gene"])
    card = build_evidence_card(row)

    for round_index in range(1, 3):
        score = query_gemini_pro(gene=gene, card=card, max_retries=3)
        if score is not None:
            return float(score)
        if round_index < 2:
            logger.error("Retrying gene %s after exhausted API attempts", gene)
            time.sleep(10 * round_index)

    logger.error("Failed to score gene %s after all retries", gene)
    raise RuntimeError(f"Failed to score gene {gene} for {MODEL_NAME}.")


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


def evaluate_model(full_df: pd.DataFrame, model_scores: dict[str, float]) -> dict:
    missing_genes = sorted(set(full_df["gene"]) - set(model_scores))
    if missing_genes:
        raise RuntimeError(f"{MODEL_NAME} is missing {len(missing_genes)} scored genes.")

    scored_df = full_df.copy()
    scored_df["score"] = scored_df["gene"].map(model_scores).astype(float)
    ranked_df = sort_ranked_df(scored_df)

    return {
        "model_name": MODEL_NAME,
        "sleep_seconds": SLEEP_SECONDS,
        "n_scored_genes": int(len(scored_df)),
        "n_internal_positives": int(scored_df["label_internal"].sum()),
        "n_external_positives": int(scored_df["label_external"].sum()),
        "auroc_external": safe_auroc(scored_df["label_external"], scored_df["score"]),
        "auroc_internal": safe_auroc(scored_df["label_internal"], scored_df["score"]),
        "p_at_10": precision_at_k(ranked_df, "label_external", 10),
        "p_at_20": precision_at_k(ranked_df, "label_external", 20),
        "ef_at_10": enrichment_factor_at_k(ranked_df, "label_external", 10),
        "top_10_genes": ranked_df.head(10)["gene"].tolist(),
    }


def save_results(data_summary: dict, metrics: dict, model_scores: dict[str, float], start_time: float) -> None:
    payload = {
        "data_summary": data_summary,
        "metrics": metrics,
        "scores": {gene: float(score) for gene, score in sorted(model_scores.items())},
        "runtime_seconds": float(time.time() - start_time),
    }
    write_json(OUTPUT_JSON, payload)


def main() -> None:
    start_time = time.time()
    setup_logging()

    logger.info("Preparing Gemini Pro data")
    data_summary, full_df = build_data()
    cache = load_cache()
    gene_set = set(full_df["gene"])
    model_scores = {
        gene: float(score)
        for gene, score in cache.get(MODEL_NAME, {}).items()
        if gene in gene_set
    }
    total = len(full_df)

    logger.info(
        "Prepared data: universe=%d internal_pos=%d external_pos=%d/%d",
        data_summary["n_genes_universe"],
        data_summary["n_internal_positives"],
        data_summary["n_external_positives"],
        data_summary["n_nm_requested"],
    )
    logger.info("%s cache: %d/%d genes already scored", MODEL_NAME, len(model_scores), total)

    missing_df = full_df.loc[~full_df["gene"].isin(model_scores)].reset_index(drop=True)
    writes_since_flush = 0

    for _, row in missing_df.iterrows():
        gene = str(row["gene"])
        logger.info("Scoring gene %s ...", gene)
        try:
            score = score_gene(row)
        except Exception as exc:
            logger.error("SKIP gene %s: %s", gene, exc)
            model_scores[gene] = 0.0
            score = 0.0
        model_scores[gene] = float(score) if score is not None else 0.0
        writes_since_flush += 1

        completed = len(model_scores)
        if completed % PROGRESS_EVERY == 0 or completed == total:
            logger.info("%s: scored %d/%d (gene: %s -> %.1f)", MODEL_NAME, completed, total, gene, score)

        if writes_since_flush >= SAVE_EVERY:
            save_cache(model_scores)
            writes_since_flush = 0

        if completed < total:
            time.sleep(SLEEP_SECONDS)

    if writes_since_flush > 0 or not missing_df.empty:
        save_cache(model_scores)

    metrics = evaluate_model(full_df, model_scores)
    save_results(data_summary, metrics, model_scores, start_time)

    logger.info(
        "%s metrics: auroc_external=%.3f auroc_internal=%.3f p_at_10=%.2f p_at_20=%.2f ef_at_10=%.2f",
        MODEL_NAME,
        metrics["auroc_external"],
        metrics["auroc_internal"],
        metrics["p_at_10"],
        metrics["p_at_20"],
        metrics["ef_at_10"],
    )
    logger.info("%s top_10=%s", MODEL_NAME, ", ".join(metrics["top_10_genes"]))
    logger.info("Wrote results to %s", OUTPUT_JSON)
    logger.info("Updated shared cache at %s", CACHE_FILE)


if __name__ == "__main__":
    main()
