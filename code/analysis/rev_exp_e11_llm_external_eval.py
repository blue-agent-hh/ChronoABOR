import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import os
os.chdir(Path(__file__).parent)

from rev_exp_common import build_external_fold_dataset

RESULTS = Path("results")

LLM_CACHE = json.loads((RESULTS / "nm_exp5_llm_cache.json").read_text())["models"]
GEMINI_CACHE = json.loads((RESULTS / "nm_exp_gemini_cache.json").read_text())

MODELS = {
    "GPT-5.2": LLM_CACHE["gpt-5.2"],
    "GPT-5.4": LLM_CACHE["gpt-5.4"],
    "Gemini-2.5-Flash": GEMINI_CACHE["gemini-2.5-flash"],
    "Gemini-2.5-Pro": GEMINI_CACHE["gemini-2.5-pro"],
}

e8 = json.loads((RESULTS / "rev_exp_e8_spec_model.json").read_text())
ws_summary = e8["primary_secreted_definition"]["within_secreted"]["subset_summary"]
WS_POS = set(ws_summary["positive_genes"])
WS_GENES = set(ws_summary["unlabeled_genes"]) | WS_POS


def eval_scores(genes, labels, scores):
    pairs = [(g, scores[g]) for g in genes if g in scores]
    gs = [p[0] for p in pairs]
    sc = np.array([p[1] for p in pairs], dtype=float)
    y = np.array([1 if g in labels else 0 for g in gs])
    n_pos = int(y.sum())
    out = {"n_eval": len(gs), "n_pos": n_pos, "n_missing_scores": len(genes) - len(gs)}
    if n_pos == 0 or n_pos == len(gs) or len(set(sc)) < 2:
        out.update({"auroc": None})
        return out
    out["auroc"] = float(roc_auc_score(y, sc))
    out["auprc"] = float(average_precision_score(y, sc))
    order = np.argsort(-sc, kind="mergesort")
    base = n_pos / len(gs)
    for k in (10, 20):
        if len(gs) >= k:
            hits = int(y[order[:k]].sum())
            out[f"p_at_{k}"] = hits / k
            out[f"ef_at_{k}"] = (hits / k) / base if base > 0 else None
    return out


results = {"models": {}, "benchmarks": {}}

for fold in ["C", "D"]:
    df, summary, _ = build_external_fold_dataset(fold)
    genes = df["gene"].tolist()
    labels = set(df.loc[df["label"].astype(int) == 1, "gene"])
    bench = {}
    for name, scores in MODELS.items():
        bench[name] = eval_scores(genes, labels, scores)
    results["benchmarks"][f"external_fold_{fold.lower()}"] = {
        "n_genes": len(genes), "n_pos": len(labels), "models": bench,
    }

ws_models = {}
for name, scores in MODELS.items():
    ws_models[name] = eval_scores(sorted(WS_GENES), WS_POS, scores)
results["benchmarks"]["within_secreted_uniprot_kw0964"] = {
    "n_genes": len(WS_GENES), "n_pos": len(WS_POS), "models": ws_models,
}

core = json.loads((RESULTS / "archive_default2018" / "nm_exp_gemini_31pro_core.json").read_text())
results["benchmarks"]["gemini31_core_subset"] = {
    "auroc": core["auroc"], "p_at_10": core["p_at_10"], "ef_at_10": core["ef_at_10"],
    "n_eval": core["eval_summary"]["eval_genes_total"],
}

out = RESULTS / "rev_exp_e11_llm_external_eval.json"
out.write_text(json.dumps(results, indent=1))
for b, v in results["benchmarks"].items():
    print(f"== {b} ==")
    if "models" in v:
        for m, r in v["models"].items():
            print(f"  {m}: auroc={r.get('auroc')} auprc={r.get('auprc')} p@10={r.get('p_at_10')} ef@10={r.get('ef_at_10')}")
    else:
        print(" ", v)
