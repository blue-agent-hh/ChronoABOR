import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_exp_common import BASE_DIR, FULL_FEATURES, make_fixed_xgb, provenance
from rev_exp_e12_strict_windows import build_eval, STRICT_SPECS
OUT = BASE_DIR / "results" / "rev_exp_e14_strict_permutation.json"
def cv_auroc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed); aucs = []
    for tr, te in skf.split(X, y):
        m = make_fixed_xgb(); m.fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))
t0 = time.time()
df, summary = build_eval(STRICT_SPECS["strict_fold_c"])
X = df[FULL_FEATURES].to_numpy(float); y = df["label"].to_numpy(int)
obs = cv_auroc(X, y); print("observed", obs, flush=True)
rng = np.random.default_rng(42); null = []
for i in range(1000):
    null.append(cv_auroc(X, rng.permutation(y)))
    if (i + 1) % 100 == 0: print(i + 1, flush=True)
null = np.array(null)
res = provenance("rev_exp_e14_strict_permutation.py", parameters={"n_permutations": 1000, "model": "default XGBoost 200/4/0.05", "cv": "stratified 5-fold, seed 42"}, input_paths=["rev_exp_e12_strict_windows.py"])
res.update({"summary": summary, "observed_cv_auroc": obs, "null": {"mean": float(null.mean()), "std": float(null.std(ddof=1)), "p95": float(np.percentile(null, 95)), "p99": float(np.percentile(null, 99)), "max": float(null.max()), "min": float(null.min())}, "empirical_p": float((np.sum(null >= obs) + 1) / (len(null) + 1)), "runtime_seconds": time.time() - t0})
OUT.write_text(json.dumps(res, indent=2, default=str)); print(json.dumps(res["null"]), res["empirical_p"])
