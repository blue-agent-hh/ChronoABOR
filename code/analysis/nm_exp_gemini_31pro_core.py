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

from nm_exp_gemini_baseline import build_data, build_evidence_card

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
LOG_FILE = RESULTS_DIR / "nm_exp_gemini_31pro_core.log"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_gemini_31pro_core.json"

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"
PROXIES = None
TIMEOUT = (10, 180)
MAX_RETRIES = 3
SLEEP_SECONDS = 1.0
NEGATIVE_SAMPLE_SIZE = 300
RANDOM_SEED = 42

NM_CORPUS = [
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

logger = logging.getLogger("nm_exp_gemini_31pro_core")


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
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def build_eval_set(full_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    gene_set = set(full_df["gene"])
    nm_found = [gene for gene in NM_CORPUS if gene in gene_set]
    nm_missing = [gene for gene in NM_CORPUS if gene not in gene_set]

    nm_order = {gene: index for index, gene in enumerate(NM_CORPUS)}
    nm_df = full_df.loc[full_df["gene"].isin(nm_found)].copy()
    nm_df["_nm_order"] = nm_df["gene"].map(nm_order)
    nm_df = nm_df.sort_values("_nm_order").drop(columns="_nm_order")

    negative_df = full_df.loc[~full_df["gene"].isin(set(NM_CORPUS))].copy()
    negative_count = min(NEGATIVE_SAMPLE_SIZE, len(negative_df))
    negative_df = negative_df.sample(n=negative_count, random_state=RANDOM_SEED).sort_values("gene")

    eval_df = pd.concat([nm_df, negative_df], ignore_index=True)
    if not eval_df["gene"].is_unique:
        raise RuntimeError("Eval set contains duplicate genes.")

    eval_summary = {
        "model_name": MODEL_NAME,
        "nm_requested": len(NM_CORPUS),
        "nm_found": len(nm_found),
        "nm_missing": nm_missing,
        "negative_sample_size": int(negative_count),
        "random_seed": RANDOM_SEED,
        "eval_genes_total": int(len(eval_df)),
    }
    return eval_df, eval_summary


def extract_score(response_json: dict, gene: str) -> float | None:
    try:
        parts = response_json["candidates"][0]["content"].get("parts", [])
    except Exception as exc:
        logger.error("%s malformed response: %s", gene, exc)
        return None

    for part in parts:
        if part.get("thought"):
            continue
        text = part.get("text")
        if not text:
            continue
        match = re.search(r"\d+(?:\.\d+)?", text)
        if match is not None:
            return float(match.group())

    logger.error("%s response missing numeric text part", gene)
    return None


def query_pro(gene: str, card: str) -> float | None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL_NAME}:generateContent?key={API_KEY}"
    )
    prompt = f"Rate AD blood biomarker potential 0-100. Respond ONLY with a number.\n\n{card}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, proxies=PROXIES, timeout=TIMEOUT)
        except Exception as exc:
            logger.error("%s exception attempt %d/%d: %s", gene, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(5)
            continue

        if response.status_code == 200:
            try:
                return extract_score(response.json(), gene)
            except Exception as exc:
                logger.error("%s JSON parse failure attempt %d/%d: %s", gene, attempt, MAX_RETRIES, exc)
                return None

        if response.status_code == 429:
            logger.error("%s rate-limited attempt %d/%d", gene, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(20)
            continue

        logger.error(
            "%s API status %s attempt %d/%d: %s",
            gene,
            response.status_code,
            attempt,
            MAX_RETRIES,
            response.text[:200],
        )
        if attempt == MAX_RETRIES:
            return None

    return None


def sort_ranked_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["score", "num_studies", "layer_count", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def safe_auroc(labels: pd.Series, scores: pd.Series) -> float | None:
    if int(labels.nunique()) < 2:
        return None
    return float(roc_auc_score(labels.astype(int), scores.astype(float)))


def precision_at_k(ranked_df: pd.DataFrame, k: int) -> float | None:
    if len(ranked_df) < k:
        return None
    return float(ranked_df.head(k)["label_external"].mean())


def enrichment_factor_at_k(ranked_df: pd.DataFrame, k: int) -> float | None:
    precision = precision_at_k(ranked_df, k)
    base_rate = float(ranked_df["label_external"].mean())
    if precision is None or base_rate <= 0:
        return None
    return float(precision / base_rate)


def evaluate_scores(eval_df: pd.DataFrame, scores: dict[str, float]) -> dict:
    scored_df = eval_df.copy()
    scored_df["score"] = scored_df["gene"].map(scores).astype(float)
    ranked_df = sort_ranked_df(scored_df)

    return {
        "auroc": safe_auroc(scored_df["label_external"], scored_df["score"]),
        "p_at_10": precision_at_k(ranked_df, 10),
        "p_at_20": precision_at_k(ranked_df, 20),
        "ef_at_10": enrichment_factor_at_k(ranked_df, 10),
        "top_15_genes": ranked_df.head(15)["gene"].tolist(),
    }


def save_results(scores: dict[str, float], metrics: dict, eval_summary: dict) -> None:
    payload = {
        "scores": {gene: float(score) for gene, score in sorted(scores.items())},
        "auroc": metrics["auroc"],
        "p_at_10": metrics["p_at_10"],
        "p_at_20": metrics["p_at_20"],
        "ef_at_10": metrics["ef_at_10"],
        "top_15_genes": metrics["top_15_genes"],
        "eval_summary": eval_summary,
    }
    write_json(OUTPUT_JSON, payload)


def log_gene_score(index: int, total: int, gene: str, score: float, is_nm: bool) -> None:
    suffix = "  *NM*" if is_nm else ""
    logger.info("[%d/%d] %s: %.1f%s", index, total, gene, score, suffix)


def main() -> None:
    setup_logging()
    start_time = time.time()

    logger.info("Preparing reduced Gemini Pro evaluation set")
    _, full_df = build_data()
    eval_df, eval_summary = build_eval_set(full_df)
    nm_set = set(NM_CORPUS)

    logger.info(
        "Prepared eval set: nm_found=%d nm_missing=%d negatives=%d total=%d",
        eval_summary["nm_found"],
        len(eval_summary["nm_missing"]),
        eval_summary["negative_sample_size"],
        eval_summary["eval_genes_total"],
    )
    if eval_summary["nm_missing"]:
        logger.info("NM genes missing from full_df: %s", ", ".join(eval_summary["nm_missing"]))
    logger.info("Scoring with %s", MODEL_NAME)

    scores: dict[str, float] = {}
    total = len(eval_df)
    for index, (_, row) in enumerate(eval_df.iterrows(), start=1):
        gene = str(row["gene"])
        is_nm = gene in nm_set
        score = query_pro(gene=gene, card=build_evidence_card(row))
        if score is None:
            logger.error("%s failed after %d attempts; storing 0.0", gene, MAX_RETRIES)
            score = 0.0

        scores[gene] = float(score)
        log_gene_score(index=index, total=total, gene=gene, score=float(score), is_nm=is_nm)

        if index < total:
            time.sleep(SLEEP_SECONDS)

    metrics = evaluate_scores(eval_df, scores)
    save_results(scores=scores, metrics=metrics, eval_summary=eval_summary)

    logger.info(
        "Final metrics: AUROC=%.4f P@10=%.4f P@20=%.4f EF@10=%.4f",
        metrics["auroc"],
        metrics["p_at_10"],
        metrics["p_at_20"],
        metrics["ef_at_10"],
    )
    logger.info("Top 15 genes: %s", ", ".join(metrics["top_15_genes"]))
    logger.info("Wrote results to %s", OUTPUT_JSON)
    logger.info("Runtime seconds: %.1f", time.time() - start_time)


if __name__ == "__main__":
    main()
