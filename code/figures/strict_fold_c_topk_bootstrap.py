from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(PROJECT))

from rev_exp_e12_strict_windows import STRICT_SPECS, build_eval, full_fit_ranking, rank_metrics
from rev_exp_e7_external_benchmark import bootstrap_topk_precision_intervals

KS = [5, 10, 20, 50]
N_BOOT = 1000
SEED = 42
OUT = Path(__file__).resolve().parent / "strict_fold_c_topk_bootstrap.json"


def main() -> None:
    stored = json.load(open(PROJECT / "results" / "rev_exp_e12_strict_windows.json"))["benchmarks"]["strict_fold_c"]["xgb"]["full_fit_rank_metrics"]
    df, summary = build_eval(STRICT_SPECS["strict_fold_c"])
    scores, cfg = full_fit_ranking(df, "xgb")
    metrics = rank_metrics(df, scores, ks=tuple(KS))
    for k in KS:
        if metrics[f"top{k}"]["hits"] != stored[f"top{k}"]["hits"]:
            raise SystemExit(f"top{k} hits {metrics[f'top{k}']['hits']} differ from stored {stored[f'top{k}']['hits']}")
    ranked = df.assign(score=scores)[["gene", "label", "num_studies", "score"]]
    boot = bootstrap_topk_precision_intervals(ranked, KS, N_BOOT, SEED)
    out = {
        "source": "rev_exp_e12 strict Fold C full-fit XGBoost ranking; bootstrap procedure from rev_exp_e7",
        "n_genes": summary["n_genes"],
        "n_positives": summary["n_positives"],
        "config": cfg,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "rows": [
            {
                "k": k,
                "hits": metrics[f"top{k}"]["hits"],
                "precision": metrics[f"top{k}"]["precision"],
                "enrichment_factor": metrics[f"top{k}"]["enrichment_factor"],
                "hypergeom_p_upper": metrics[f"top{k}"]["hypergeom_p_upper"],
                "precision_bootstrap_ci95": boot[k]["ci95"],
                "precision_bootstrap_summary": boot[k]["summary"],
            }
            for k in KS
        ],
    }
    OUT.write_text(json.dumps(out, indent=2))
    for r in out["rows"]:
        print(r["k"], r["hits"], r["precision"], r["precision_bootstrap_ci95"])


if __name__ == "__main__":
    main()
