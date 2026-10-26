#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from rev_exp_common import provenance
from rev_exp_e12_strict_windows import STRICT_SPECS, build_eval, rank_metrics

RESULTS_DIR = BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e17_gemini31_strict.json"

LLM_CACHE = json.loads((RESULTS_DIR / "nm_exp5_llm_cache.json").read_text())["models"]
GEMINI_CACHE = json.loads((RESULTS_DIR / "nm_exp_gemini_cache.json").read_text())
GEMINI31_CACHE = json.loads((RESULTS_DIR / "nm_exp_gemini_31pro_core.json").read_text())["scores"]

REFERENCE_MODELS = {
    "GPT-5.2": LLM_CACHE["gpt-5.2"],
    "GPT-5.4": LLM_CACHE["gpt-5.4"],
    "Gemini-2.5-Flash": GEMINI_CACHE["gemini-2.5-flash"],
    "Gemini-2.5-Pro": GEMINI_CACHE["gemini-2.5-pro"],
}
ALL_MODELS = {"Gemini-3.1-Pro": GEMINI31_CACHE, **REFERENCE_MODELS}


def evaluate_model(eval_df, scores: dict[str, float]) -> dict:
    genes_with_scores = set(scores)
    subset_df = eval_df.loc[eval_df["gene"].isin(genes_with_scores)].copy().reset_index(drop=True)
    positives_full = set(eval_df.loc[eval_df["label"].astype(int) == 1, "gene"])
    positives_present = sorted(subset_df.loc[subset_df["label"].astype(int) == 1, "gene"].tolist())
    positives_absent = sorted(positives_full - set(subset_df["gene"]))

    out = {
        "n_genes": int(len(subset_df)),
        "n_positives": int(subset_df["label"].sum()),
        "n_missing_scores_from_frame": int(len(eval_df) - len(subset_df)),
        "positives_present": positives_present,
        "positives_absent": positives_absent,
    }

    score_vec = subset_df["gene"].map(scores).to_numpy(dtype=float)
    label_vec = subset_df["label"].to_numpy(dtype=int)
    out["auroc"] = float(roc_auc_score(label_vec, score_vec))
    out["auprc"] = float(average_precision_score(label_vec, score_vec))

    top10 = rank_metrics(subset_df, score_vec, ks=(10,))["top10"]
    out["p_at_10"] = float(top10["precision"])
    out["ef_at_10"] = float(top10["enrichment_factor"]) if top10["enrichment_factor"] is not None else None
    out["top10_hits"] = int(top10["hits"])
    out["hypergeom_p_upper_tail_at_10"] = float(top10["hypergeom_p_upper"])
    out["top10_genes"] = top10["top_genes"]
    return out


def benchmark_entry(name: str, spec: dict[str, int]) -> dict:
    full_df, full_summary = build_eval(spec)
    gemini_gene_set = set(GEMINI31_CACHE)
    restricted_df = full_df.loc[full_df["gene"].isin(gemini_gene_set)].copy().reset_index(drop=True)
    positives_full = set(full_df.loc[full_df["label"].astype(int) == 1, "gene"])
    restricted_summary = {
        "n_genes": int(len(restricted_df)),
        "n_positives": int(restricted_df["label"].sum()),
        "n_missing_scores_from_frame": int(len(full_df) - len(restricted_df)),
        "positives_present": sorted(restricted_df.loc[restricted_df["label"].astype(int) == 1, "gene"].tolist()),
        "positives_absent": sorted(positives_full - set(restricted_df["gene"])),
    }

    return {
        "benchmark_name": name,
        "full_frame_summary": full_summary,
        "restricted_summary": restricted_summary,
        "models": {model_name: evaluate_model(restricted_df, score_map) for model_name, score_map in ALL_MODELS.items()},
    }


def main() -> None:
    t0 = time.time()
    payload = provenance(
        "rev_exp_e17_gemini31_strict.py",
        parameters={
            "strict_specs": STRICT_SPECS,
            "restricted_gene_set": "results/nm_exp_gemini_31pro_core.json:scores",
            "reference_models": list(REFERENCE_MODELS),
            "ranking_metric_source": "rev_exp_e12_strict_windows.rank_metrics",
            "eval_frame_source": "rev_exp_e12_strict_windows.build_eval",
            "top_k": 10,
        },
        input_paths=[
            "rev_exp_e12_strict_windows.py",
            "rev_exp_common.py",
            "results/nm_exp_gemini_31pro_core.json",
            "results/nm_exp5_llm_cache.json",
            "results/nm_exp_gemini_cache.json",
        ],
    )
    payload["benchmarks"] = {
        "strict_fold_c": benchmark_entry("strict_fold_c", STRICT_SPECS["strict_fold_c"]),
        "strict_fold_d": benchmark_entry("strict_fold_d", STRICT_SPECS["strict_fold_d"]),
    }
    payload["runtime_seconds"] = time.time() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    print("benchmark\tmodel\tn_genes\tn_pos\tauroc\tauprc\tp@10\tef@10\thypergeom_p")
    for bench_name, bench in payload["benchmarks"].items():
        for model_name, metrics in bench["models"].items():
            print(
                f"{bench_name}\t{model_name}\t{metrics['n_genes']}\t{metrics['n_positives']}\t"
                f"{metrics['auroc']:.4f}\t{metrics['auprc']:.4f}\t{metrics['p_at_10']:.4f}\t"
                f"{metrics['ef_at_10']:.4f}\t{metrics['hypergeom_p_upper_tail_at_10']:.6g}"
            )
        gs = bench["restricted_summary"]
        print(
            f"{bench_name}\tGemini-3.1-Pro positives present={gs['positives_present']} "
            f"absent={gs['positives_absent']}"
        )
    print(f"Wrote {OUT_PATH} in {payload['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
