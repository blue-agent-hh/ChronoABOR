from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_exp_e8_spec_model as e8
from rev_exp_common import BASE_DIR, build_gene_universe, load_all_tables, provenance
from rev_exp_annotations import build_annotation_bundle

OUT = BASE_DIR / "results" / "rev_exp_e13_strict_spec_ablation.json"
CUTOFF = 2020
LABEL_START, LABEL_END = 2021, 2025


def main() -> None:
    t0 = time.time()
    corpus_df = e8.load_nm_corpus_v2()
    bundle = build_annotation_bundle()
    annotated_tables, _, _ = e8.prepare_annotated_tables()
    scored_universe = build_gene_universe(load_all_tables())
    kw_set = set(bundle["uniprot"]["secreted_kw_genes"])
    ad_driver_set = set(bundle["kegg"]["ad_driver_genes"])

    base = e8.build_base_feature_matrix(CUTOFF, annotated_tables, scored_universe, ad_driver_set)
    feat = e8.attach_secreted_definition(base["feature_df"], kw_set)
    positives = set(corpus_df.loc[(corpus_df["year"] >= LABEL_START) & (corpus_df["year"] <= LABEL_END), "gene"])

    fold_df = feat.loc[feat["has_temporal_evidence_by_cutoff"]].copy().reset_index(drop=True)
    fold_df["label"] = fold_df["gene"].isin(positives).astype(int)
    print(f"strict Fold C spec matrix: n={len(fold_df)} pos={int(fold_df['label'].sum())}", flush=True)

    ws_df = feat.loc[feat["is_secreted"] == 1].copy().reset_index(drop=True)
    ws_df["label"] = ws_df["gene"].isin(positives).astype(int)
    print(f"within-secreted strict: n={len(ws_df)} pos={int(ws_df['label'].sum())}", flush=True)
    ws_results = {}
    for lr in e8.LEARNING_RATES:
        builder = e8.make_xgb_builder(lr)
        full_scores = e8.full_fit_scores(ws_df, e8.MANUSCRIPT_TEN_FEATURES, builder)
        prior_scores = e8.full_fit_scores(ws_df, e8.PRIOR_FEATURES, e8.make_spec_l2_logistic)
        ws_results[e8.lr_key(lr)] = {
            "full_model": {"rank_metrics": e8.compute_rank_metrics_from_scores_spec(ws_df, full_scores, ks=(5, 10, 20))},
            "prior_only": {"rank_metrics": e8.compute_rank_metrics_from_scores_spec(ws_df, prior_scores, ks=(5, 10, 20))},
        }
    fake_within = {
        "direct_methods": {"prior_only": {"rank_metrics": ws_results["lr_0.05"]["prior_only"]["rank_metrics"]}},
        "xgb_results_by_learning_rate": {k: {"full_model": {"rank_metrics": v["full_model"]["rank_metrics"]}} for k, v in ws_results.items()},
    }

    ablation = e8.evaluate_ablation(fold_df, e8.LEARNING_RATES, fake_within)
    shap_res = e8.evaluate_shap(fold_df, e8.LEARNING_RATES)

    payload = provenance(
        "rev_exp_e13_strict_spec_ablation.py",
        parameters={"cutoff_year": CUTOFF, "label_window": [LABEL_START, LABEL_END], "rule": "first-report year strictly after cutoff", "learning_rates": e8.LEARNING_RATES, "n_splits": e8.N_SPLITS},
        input_paths=["rev_exp_e8_spec_model.py", "nm_corpus_v2.csv"],
    )
    payload.update(
        {
            "fold_c_strict": {"n_genes": int(len(fold_df)), "n_positives": int(fold_df["label"].sum()), "positives": sorted(positives & set(fold_df["gene"]))},
            "within_secreted_strict": {"n_genes": int(len(ws_df)), "n_positives": int(ws_df["label"].sum()), "positives": sorted(positives & set(ws_df["gene"])), "results": ws_results},
            "fold_c_structured_ablation": ablation,
            "fold_c_shap": shap_res,
            "runtime_seconds": time.time() - t0,
        }
    )
    OUT.write_text(json.dumps(e8.sanitize_for_json(payload), indent=2, default=str))
    print(f"Wrote {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
