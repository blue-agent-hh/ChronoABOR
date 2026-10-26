from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_exp_common import (
    BASE_DIR,
    FULL_FEATURES,
    RANDOM_SEED,
    build_external_fold_dataset,
    build_feature_context,
    load_nm_corpus_table,
    provenance,
    utc_timestamp,
)
from rev_exp_annotations import build_annotation_bundle

RESULTS_DIR = BASE_DIR / "results"
OUT_PATH = RESULTS_DIR / "rev_exp_e9_nested_tuning.json"
WITHIN_SECRETED_CUTOFF = 2021

N_OUTER = 5
N_INNER = 5
OUTER_SEEDS = [42, 137, 2024]
INNER_SEED = 7

XGB_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0, 10.0],
    "scale_pos_weight_mode": ["none", "balanced"],
}
LOGISTIC_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}


def xgb_param_grid() -> list[dict]:
    keys = list(XGB_GRID.keys())
    combos = list(itertools.product(*(XGB_GRID[k] for k in keys)))
    rng = np.random.default_rng(RANDOM_SEED)
    if len(combos) > 96:
        keep = rng.choice(len(combos), size=96, replace=False)
        combos = [combos[i] for i in sorted(keep)]
    return [dict(zip(keys, c)) for c in combos]


def make_xgb_from_config(cfg: dict, y_train: np.ndarray) -> XGBClassifier:
    spw = 1.0
    if cfg["scale_pos_weight_mode"] == "balanced":
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        spw = n_neg / n_pos
    return XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        reg_lambda=cfg["reg_lambda"],
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def make_logistic_from_config(cfg: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=cfg["C"],
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def inner_cv_select(X_train, y_train, kind: str):
    if kind == "xgb":
        grid = xgb_param_grid()
        builder = lambda cfg: make_xgb_from_config(cfg, y_train)
    else:
        grid = [{"C": c} for c in LOGISTIC_GRID["C"]]
        builder = make_logistic_from_config
    splitter = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=INNER_SEED)
    best_cfg, best_score = None, -np.inf
    for cfg in grid:
        fold_aucs = []
        for tr_idx, va_idx in splitter.split(X_train, y_train):
            y_tr = y_train[tr_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_train[va_idx])) < 2:
                continue
            model = builder(cfg)
            model.fit(X_train[tr_idx], y_tr)
            s = model.predict_proba(X_train[va_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train[va_idx], s))
        if not fold_aucs:
            continue
        score = float(np.mean(fold_aucs))
        if score > best_score:
            best_score, best_cfg = score, cfg
    return best_cfg, best_score


def repeated_nested_cv(eval_df, kind: str, label: str) -> dict:
    X = eval_df[FULL_FEATURES].to_numpy(dtype=float)
    y = eval_df["label"].to_numpy(dtype=int)
    repeats = []
    for seed in OUTER_SEEDS:
        splitter = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        oof = np.full(len(eval_df), np.nan)
        fold_aucs, chosen = [], []
        for train_idx, test_idx in splitter.split(X, y):
            cfg, inner_score = inner_cv_select(X[train_idx], y[train_idx], kind)
            if kind == "xgb":
                model = make_xgb_from_config(cfg, y[train_idx])
            else:
                model = make_logistic_from_config(cfg)
            model.fit(X[train_idx], y[train_idx])
            s = model.predict_proba(X[test_idx])[:, 1]
            oof[test_idx] = s
            fold_aucs.append(float(roc_auc_score(y[test_idx], s)))
            chosen.append({**cfg, "inner_cv_auroc": inner_score})
        repeats.append(
            {
                "outer_seed": seed,
                "fold_aurocs": fold_aucs,
                "auroc_mean": float(np.mean(fold_aucs)),
                "auroc_oof": float(roc_auc_score(y, oof)),
                "auprc_oof": float(average_precision_score(y, oof)),
                "selected_configs": chosen,
            }
        )
        print(f"  [{label}/{kind}/seed={seed}] mean={repeats[-1]['auroc_mean']:.3f} oof={repeats[-1]['auroc_oof']:.3f}", flush=True)
    all_fold_aucs = [a for r in repeats for a in r["fold_aurocs"]]
    return {
        "n_repeats": len(OUTER_SEEDS),
        "n_outer_folds": N_OUTER,
        "n_inner_folds": N_INNER,
        "auroc_repeat_mean": float(np.mean([r["auroc_mean"] for r in repeats])),
        "auroc_repeat_sd": float(np.std([r["auroc_mean"] for r in repeats], ddof=1)),
        "auroc_allfolds_mean": float(np.mean(all_fold_aucs)),
        "auroc_allfolds_sd": float(np.std(all_fold_aucs, ddof=1)),
        "auroc_oof_mean": float(np.mean([r["auroc_oof"] for r in repeats])),
        "auprc_oof_mean": float(np.mean([r["auprc_oof"] for r in repeats])),
        "repeats": repeats,
    }


def build_within_secreted_eval_df():
    context = build_feature_context(cutoff_year=WITHIN_SECRETED_CUTOFF)
    feature_df = context["feature_df"].copy()
    bundle = build_annotation_bundle()
    kw_genes = set(bundle["uniprot"]["secreted_kw_genes"])
    nm_rows = load_nm_corpus_table()
    positives = {row["canonical_gene"] for row in nm_rows if row["fold_c"]}
    eval_df = feature_df.loc[feature_df["gene"].isin(kw_genes)].copy().reset_index(drop=True)
    eval_df["label"] = eval_df["gene"].isin(positives).astype(int)
    meta = {
        "cutoff_year": WITHIN_SECRETED_CUTOFF,
        "n_genes": int(len(eval_df)),
        "n_positives": int(eval_df["label"].sum()),
    }
    return eval_df, meta


def main() -> None:
    t0 = time.time()
    print(f"[{utc_timestamp()}] E9 nested tuning starting", flush=True)
    payload = provenance(
        "rev_exp_e9_nested_tuning.py",
        parameters={
            "n_outer": N_OUTER,
            "n_inner": N_INNER,
            "outer_seeds": OUTER_SEEDS,
            "inner_seed": INNER_SEED,
            "xgb_grid": XGB_GRID,
            "logistic_grid": LOGISTIC_GRID,
            "scoring": "AUROC",
            "protocol": "repeated nested stratified CV; hyperparameters selected inside each outer-training partition only",
        },
        input_paths=["rev_exp_common.py", "nm_exp_p6_secreted_ablation.py"],
    )
    payload["benchmarks"] = {}

    for fold_key in ["C", "D"]:
        eval_df, summary, _ctx = build_external_fold_dataset(fold_key)
        print(f"Fold {fold_key}: n={len(eval_df)} pos={int(eval_df['label'].sum())}", flush=True)
        entry = {"n_genes": int(len(eval_df)), "n_positives": int(eval_df["label"].sum())}
        for kind in ["xgb", "logistic"]:
            entry[kind] = repeated_nested_cv(eval_df, kind, f"fold{fold_key}")
        payload["benchmarks"][f"external_fold_{fold_key.lower()}"] = entry

    ws_df, ws_meta = build_within_secreted_eval_df()
    print(f"within-secreted: n={len(ws_df)} pos={int(ws_df['label'].sum())}", flush=True)
    ws_entry = {"n_genes": int(len(ws_df)), "n_positives": int(ws_df["label"].sum())}
    for kind in ["xgb", "logistic"]:
        ws_entry[kind] = repeated_nested_cv(ws_df, kind, "within_secreted")
    payload["benchmarks"]["within_secreted_uniprot_kw0964"] = ws_entry

    payload["runtime_seconds"] = time.time() - t0
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {OUT_PATH} in {payload['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
