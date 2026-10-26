from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_exp_common import (
    BASE_DIR,
    FULL_FEATURES,
    build_feature_context,
    load_nm_corpus_table,
    provenance,
    utc_timestamp,
)
from rev_exp_annotations import build_annotation_bundle
from rev_exp_e9_nested_tuning import (
    LOGISTIC_GRID,
    XGB_GRID,
    N_INNER,
    N_OUTER,
    OUTER_SEEDS,
    INNER_SEED,
    make_logistic_from_config,
    make_xgb_from_config,
    repeated_nested_cv,
)

RESULTS_DIR = BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e12_strict_windows.json"

STRICT_SPECS = {
    "strict_fold_c": {"cutoff_year": 2020, "label_year_start": 2021, "label_year_end": 2025},
    "strict_fold_d": {"cutoff_year": 2021, "label_year_start": 2022, "label_year_end": 2025},
}
WITHIN_SECRETED_SPEC = {"cutoff_year": 2020, "label_year_start": 2021, "label_year_end": 2025}


def positives_for_window(nm_rows, start: int, end: int) -> set[str]:
    return {r["canonical_gene"] for r in nm_rows if start <= int(r["year"]) <= end}


def rank_metrics(eval_df, scores, ks=(5, 10, 20, 50)):
    order = np.argsort(-scores, kind="stable")
    y = eval_df["label"].to_numpy(dtype=int)[order]
    n, K = len(y), int(y.sum())
    out = {}
    for k in ks:
        hits = int(y[:k].sum())
        out[f"top{k}"] = {
            "hits": hits,
            "precision": hits / k,
            "enrichment_factor": (hits / k) / (K / n) if K else None,
            "hypergeom_p_upper": float(hypergeom.sf(hits - 1, n, K, k)) if hits > 0 else 1.0,
            "top_genes": eval_df["gene"].to_numpy()[order][:k].tolist(),
        }
    return out


def baselines(eval_df):
    y = eval_df["label"].to_numpy(dtype=int)
    res = {}
    for col, name in [("is_secreted", "secretion_status"), ("num_studies", "publication_count"), ("sig_max_auc", "max_signature_auc"), ("layer_count", "layer_count")]:
        if col in eval_df.columns and eval_df[col].nunique() > 1:
            s = eval_df[col].to_numpy(dtype=float)
            res[name] = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
        else:
            res[name] = {"auroc": 0.5, "note": "constant within evaluation set"}
    return res


def full_fit_ranking(eval_df, kind: str):
    X = eval_df[FULL_FEATURES].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    if kind == "xgb":
        cfg = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 1.0, "scale_pos_weight_mode": "none"}
        model = make_xgb_from_config(cfg, y)
    else:
        cfg = {"C": 0.1}
        model = make_logistic_from_config(cfg)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1], cfg


def build_eval(spec, kw_genes=None):
    ctx = build_feature_context(cutoff_year=spec["cutoff_year"])
    nm_rows = load_nm_corpus_table()
    pos = positives_for_window(nm_rows, spec["label_year_start"], spec["label_year_end"])
    df = ctx["feature_df"].copy()
    if kw_genes is not None:
        df = df.loc[df["gene"].isin(kw_genes)].copy()
    else:
        df = df.loc[df["has_evidence_by_cutoff"]].copy()
    df = df.reset_index(drop=True)
    df["label"] = df["gene"].isin(pos).astype(int)
    summary = {
        **spec,
        "positives_requested": sorted(pos),
        "n_positives_requested": len(pos),
        "positives_in_eval": sorted(pos & set(df["gene"])),
        "positives_missing_from_eval": sorted(pos - set(df["gene"])),
        "positives_with_year_equal_or_before_cutoff_in_old_fold_definition": sorted(
            {r["canonical_gene"] for r in nm_rows if int(r["year"]) <= spec["cutoff_year"] and (r["fold_c"] or r["fold_d"])}
        ),
        "n_genes": int(len(df)),
        "n_positives": int(df["label"].sum()),
    }
    return df, summary


def main() -> None:
    t0 = time.time()
    print(f"[{utc_timestamp()}] E12 strict windows starting", flush=True)
    payload = provenance(
        "rev_exp_e12_strict_windows.py",
        parameters={
            "strict_specs": STRICT_SPECS,
            "within_secreted_spec": WITHIN_SECRETED_SPEC,
            "rule": "a gene is positive iff its corpus first-report year is strictly greater than the feature cutoff year",
            "n_outer": N_OUTER, "n_inner": N_INNER, "outer_seeds": OUTER_SEEDS, "inner_seed": INNER_SEED,
            "xgb_grid": XGB_GRID, "logistic_grid": LOGISTIC_GRID,
        },
        input_paths=["rev_exp_common.py", "rev_exp_e9_nested_tuning.py", "paper/supplementary_table_s1.tex"],
    )
    payload["benchmarks"] = {}

    nm_rows = load_nm_corpus_table()
    payload["corpus_year_audit"] = {
        "old_fold_c_positives_with_year_le_2021": sorted(r["canonical_gene"] for r in nm_rows if r["fold_c"] and int(r["year"]) <= 2021),
        "old_fold_c_positives_with_year_le_2020": sorted(r["canonical_gene"] for r in nm_rows if r["fold_c"] and int(r["year"]) <= 2020),
        "old_fold_d_positives_with_year_le_2022": sorted(r["canonical_gene"] for r in nm_rows if r["fold_d"] and int(r["year"]) <= 2022),
        "year_histogram": {str(y): sum(1 for r in nm_rows if int(r["year"]) == y) for y in sorted({int(r["year"]) for r in nm_rows})},
    }

    for key, spec in STRICT_SPECS.items():
        df, summary = build_eval(spec)
        print(f"{key}: n={summary['n_genes']} pos={summary['n_positives']} ({summary['positives_in_eval']})", flush=True)
        entry = {"summary": summary, "baselines": baselines(df)}
        for kind in ["xgb", "logistic"]:
            entry[kind] = repeated_nested_cv(df, kind, key)
            s, cfg = full_fit_ranking(df, kind)
            entry[kind]["full_fit_rank_metrics"] = {"config": cfg, **rank_metrics(df, s)}
        payload["benchmarks"][key] = entry

    bundle = build_annotation_bundle()
    kw_genes = set(bundle["uniprot"]["secreted_kw_genes"])
    ws_df, ws_summary = build_eval(WITHIN_SECRETED_SPEC, kw_genes=kw_genes)
    print(f"within-secreted strict: n={ws_summary['n_genes']} pos={ws_summary['n_positives']} ({ws_summary['positives_in_eval']})", flush=True)
    ws_entry = {"summary": ws_summary, "baselines": baselines(ws_df)}
    for col in ["n_evidence_total", "num_modalities", "num_studies", "layer_count", "sig_max_auc", "deg_count"]:
        if col in ws_df.columns and ws_df[col].nunique() > 1:
            ws_entry["baselines"][col] = {"auroc": float(roc_auc_score(ws_df["label"], ws_df[col])), "auprc": float(average_precision_score(ws_df["label"], ws_df[col]))}
    for kind in ["xgb", "logistic"]:
        ws_entry[kind] = repeated_nested_cv(ws_df, kind, "within_secreted_strict")
        s, cfg = full_fit_ranking(ws_df, kind)
        ws_entry[kind]["full_fit_rank_metrics"] = {"config": cfg, **rank_metrics(ws_df, s, ks=(5, 10, 20))}
    payload["benchmarks"]["within_secreted_strict"] = ws_entry

    payload["runtime_seconds"] = time.time() - t0
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
