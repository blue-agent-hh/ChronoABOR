#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl

from nm_exp4_full_universe import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    build_features_and_evidence,
    build_gene_universe,
    build_signature_pmid_year_map,
    load_tables,
    make_xgb,
    to_jsonable,
)
from nm_exp_p1_prior_control import SECRETED_GENES
from nm_exp_p5_external_validation import IADRC_UP_RAW, KNIGHT_UP_RAW, canonicalize_gene_list

_mpl.rcParams["svg.fonttype"] = "none"
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p7_ranked_enrichment.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p7_enrichment_plot.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_p7_enrichment_plot.pdf"
P5_JSON = RESULTS_DIR / "nm_exp_p5_external_validation.json"
P5_FALLBACK_JSON = RESULTS_DIR / "nm_exp_p5_overlap.json"

FEATURE_CUTOFF_YEAR = 2021
PERMUTATION_COUNT = 1000
WEIGHT_EXPONENT = 1.0

NM_CORPUS_REQUESTED = [
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

FULL_FEATURES = FEATURE_COLUMNS + ["is_secreted"]


def load_p5_hit_source() -> dict:
    for candidate_path in [P5_JSON, P5_FALLBACK_JSON]:
        if not candidate_path.exists():
            continue
        data = json.loads(candidate_path.read_text())
        external_validation = data.get("external_validation", {})
        external_upregulated = external_validation.get("external_upregulated", {})
        canonical_genes = external_upregulated.get("canonical_genes")
        if canonical_genes:
            return {
                "source": str(candidate_path),
                "source_mode": "json",
                "raw_count": external_upregulated.get("n_raw", len(IADRC_UP_RAW) + len(KNIGHT_UP_RAW)),
                "canonical_unique_count": external_upregulated.get("n_canonical_unique", len(canonical_genes)),
                "canonical_genes": canonical_genes,
                "genes_in_universe_from_json": external_upregulated.get("genes_in_universe"),
            }

    canonical_genes = canonicalize_gene_list(IADRC_UP_RAW + KNIGHT_UP_RAW)
    return {
        "source": "embedded_fallback_from_nm_exp_p5_external_validation.py",
        "source_mode": "embedded_constants",
        "raw_count": len(IADRC_UP_RAW) + len(KNIGHT_UP_RAW),
        "canonical_unique_count": len(canonical_genes),
        "canonical_genes": canonical_genes,
        "genes_in_universe_from_json": None,
    }


def build_fold_c_frame() -> tuple[pd.DataFrame, dict]:
    tables = load_tables()
    tables["signature"]["Year_int"] = pd.to_numeric(tables["signature"]["Year"], errors="coerce")
    tables["signature"]["AUC_num"] = pd.to_numeric(tables["signature"]["AUC"], errors="coerce")

    gene_universe = build_gene_universe(tables)
    pmid_year_map, pmid_conflicts = build_signature_pmid_year_map(tables["signature"])
    feature_df, evidence_by_cutoff, year_source_counts = build_features_and_evidence(
        tables=tables,
        gene_universe=gene_universe,
        pmid_year_map=pmid_year_map,
        cutoff_year=FEATURE_CUTOFF_YEAR,
    )

    nm_requested_set = set(NM_CORPUS_REQUESTED)
    feature_df["is_secreted"] = feature_df["gene"].isin(SECRETED_GENES).astype(int)
    feature_df["label"] = feature_df["gene"].isin(nm_requested_set).astype(int)

    summary = {
        "n_genes_universe": int(len(feature_df)),
        "n_genes_with_2021_evidence": int(len(evidence_by_cutoff)),
        "n_nm_requested": int(len(NM_CORPUS_REQUESTED)),
        "n_nm_in_universe": int(feature_df["label"].sum()),
        "nm_genes_in_universe": sorted(set(feature_df.loc[feature_df["label"] == 1, "gene"])),
        "nm_genes_missing_from_universe": sorted(nm_requested_set - set(feature_df["gene"])),
        "signature_pmid_year_conflicts": pmid_conflicts,
        "year_source_counts": {layer: dict(counter) for layer, counter in year_source_counts.items()},
    }
    return feature_df, summary


def train_fold_c_model(feature_df: pd.DataFrame) -> pd.DataFrame:
    train_df = feature_df.copy().reset_index(drop=True)
    X = train_df[FULL_FEATURES].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=int)

    model = make_xgb()
    model.fit(X, y)
    train_df["chronoabor_score"] = model.predict_proba(X)[:, 1]
    train_df["sig_max_auc_score"] = train_df["sig_max_auc"].astype(float)
    return train_df


def compute_enrichment_curve(
    ranked_df: pd.DataFrame,
    score_column: str,
    hit_set: set[str],
) -> dict:
    df = ranked_df.sort_values(
        by=[score_column, "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scores = df[score_column].to_numpy(dtype=float)
    genes = df["gene"].to_numpy(dtype=str)
    hit_mask = df["gene"].isin(hit_set).to_numpy(dtype=bool)
    if hit_mask.sum() == 0:
        raise RuntimeError("No external proteomics hits overlapped the ranked universe.")

    hit_weights = np.power(np.abs(scores[hit_mask]), WEIGHT_EXPONENT)
    hit_weight_sum = float(hit_weights.sum())
    if hit_weight_sum <= 0:
        hit_weight_sum = 1.0
    miss_count = int((~hit_mask).sum())
    miss_penalty = 1.0 / miss_count if miss_count > 0 else 0.0

    running_es = np.empty(len(df), dtype=float)
    running_value = 0.0
    for idx, (is_hit, score) in enumerate(zip(hit_mask, scores)):
        if is_hit:
            running_value += float(np.power(abs(score), WEIGHT_EXPONENT) / hit_weight_sum)
        else:
            running_value -= miss_penalty
        running_es[idx] = running_value

    max_es = float(np.max(running_es))
    min_es = float(np.min(running_es))
    if abs(max_es) >= abs(min_es):
        es = max_es
        peak_idx = int(np.argmax(running_es))
    else:
        es = min_es
        peak_idx = int(np.argmin(running_es))

    if es >= 0:
        leading_edge_genes = genes[: peak_idx + 1][hit_mask[: peak_idx + 1]].tolist()
    else:
        leading_edge_genes = genes[peak_idx:][hit_mask[peak_idx:]].tolist()

    return {
        "ranked_df": df,
        "genes": genes,
        "scores": scores,
        "hit_mask": hit_mask,
        "running_es": running_es,
        "es": es,
        "peak_index": peak_idx,
        "leading_edge_genes": leading_edge_genes,
    }


def compute_nes_and_pvalue(
    scores: np.ndarray,
    hit_count: int,
    observed_es: float,
    n_permutations: int = PERMUTATION_COUNT,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_genes = len(scores)
    permutation_es = np.empty(n_permutations, dtype=float)

    for idx in range(n_permutations):
        permuted_hit_mask = np.zeros(n_genes, dtype=bool)
        permuted_hit_mask[rng.choice(n_genes, size=hit_count, replace=False)] = True
        permuted_curve = compute_enrichment_curve(
            ranked_df=pd.DataFrame(
                {
                    "gene": np.arange(n_genes).astype(str),
                    "num_studies": np.zeros(n_genes, dtype=int),
                    "permuted_score": scores,
                }
            ),
            score_column="permuted_score",
            hit_set=set(np.arange(n_genes)[permuted_hit_mask].astype(str)),
        )
        permutation_es[idx] = permuted_curve["es"]

    if observed_es >= 0:
        null_same_sign = permutation_es[permutation_es >= 0]
        if len(null_same_sign) == 0:
            null_same_sign = np.array([1.0], dtype=float)
        nes = float(observed_es / np.mean(null_same_sign))
        p_value = float((1 + np.sum(permutation_es >= observed_es)) / (n_permutations + 1))
    else:
        null_same_sign = np.abs(permutation_es[permutation_es < 0])
        if len(null_same_sign) == 0:
            null_same_sign = np.array([1.0], dtype=float)
        nes = float(observed_es / np.mean(null_same_sign))
        p_value = float((1 + np.sum(permutation_es <= observed_es)) / (n_permutations + 1))
    return nes, p_value, permutation_es


def format_summary(result: dict) -> dict:
    ranked_df = result["ranked_df"]
    hit_mask = result["hit_mask"]
    peak_idx = result["peak_index"]
    return {
        "es": float(result["es"]),
        "nes": float(result["nes"]),
        "p_value": float(result["p_value"]),
        "n_hits_in_universe": int(hit_mask.sum()),
        "leading_edge_genes": result["leading_edge_genes"],
        "leading_edge_size": int(len(result["leading_edge_genes"])),
        "peak_index_1based": int(peak_idx + 1),
        "top_20_ranked_genes": ranked_df.head(20)["gene"].tolist(),
        "top_20_hit_genes": ranked_df.head(20).loc[ranked_df.head(20)["gene"].isin(result["hit_set"]), "gene"].tolist(),
        "top_100_hit_count": int(ranked_df.head(100)["gene"].isin(result["hit_set"]).sum()),
        "null_distribution": {
            "mean": float(np.mean(result["permutation_es"])),
            "std": float(np.std(result["permutation_es"], ddof=1)),
            "p025": float(np.quantile(result["permutation_es"], 0.025)),
            "p975": float(np.quantile(result["permutation_es"], 0.975)),
        },
    }


def plot_method_block(fig: plt.Figure, gs, result: dict, title: str) -> None:
    ranked_df = result["ranked_df"]
    running_es = result["running_es"]
    hit_mask = result["hit_mask"]
    scores = result["scores"]
    peak_idx = result["peak_index"]
    x = np.arange(1, len(ranked_df) + 1)

    ax_es = fig.add_subplot(gs[0])
    ax_hits = fig.add_subplot(gs[1], sharex=ax_es)
    ax_scores = fig.add_subplot(gs[2], sharex=ax_es)
    for ax in (ax_es, ax_hits, ax_scores):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_es.plot(x, running_es, color="#0072B2", linewidth=2.0)
    ax_es.axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    ax_es.axvline(peak_idx + 1, color="#D55E00", linestyle="--", linewidth=1.0)
    ax_es.scatter([peak_idx + 1], [running_es[peak_idx]], color="#D55E00", s=22, zorder=3)
    ax_es.set_ylabel("Running ES")
    ax_es.set_title(
        f"{title} | ES={result['es']:.3f}, NES={result['nes']:.3f}, p={result['p_value']:.4f}",
        fontsize=11,
        pad=8,
    )
    ax_es.grid(alpha=0.22)

    hit_positions = x[hit_mask]
    ax_hits.vlines(hit_positions, 0.0, 1.0, color="#009E73", linewidth=0.7)
    ax_hits.set_ylim(0.0, 1.0)
    ax_hits.set_yticks([])
    ax_hits.set_ylabel("Hits")

    ax_scores.fill_between(x, scores, color="#56B4E9", alpha=0.8)
    ax_scores.set_ylabel("Rank score")
    ax_scores.set_xlabel("Rank in 2,843-gene universe")
    ax_scores.grid(alpha=0.15)

    leading_edge_preview = ", ".join(result["leading_edge_genes"][:12]) or "None"
    ax_scores.text(
        0.01,
        0.96,
        f"Leading edge ({len(result['leading_edge_genes'])}): {leading_edge_preview}",
        transform=ax_scores.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )


def create_figure(method_results: dict[str, dict], hit_summary: dict) -> None:
    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(
        6,
        1,
        height_ratios=[3.0, 0.5, 1.5, 3.0, 0.5, 1.5],
        hspace=0.15,
    )
    plot_method_block(fig, [gs[0], gs[1], gs[2]], method_results["chronoabor_score"], "ChronoABOR Fold C model")
    plot_method_block(fig, [gs[3], gs[4], gs[5]], method_results["sig_max_auc_score"], "sig_max_auc baseline")
    fig.suptitle(
        (
            "GSEA-style enrichment of external plasma proteomics hits "
            f"({hit_summary['n_hits_in_universe']} in universe; "
            f"{hit_summary['n_hits_canonical_unique']} canonical hits requested)"
        ),
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.98, hspace=0.28)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hit_source = load_p5_hit_source()
    feature_df, dataset_summary = build_fold_c_frame()
    scored_df = train_fold_c_model(feature_df)

    universe_set = set(scored_df["gene"])
    canonical_hits = hit_source["canonical_genes"]
    hit_set = set(canonical_hits) & universe_set
    hit_summary = {
        "source": hit_source["source"],
        "source_mode": hit_source["source_mode"],
        "n_hits_raw_requested": int(hit_source["raw_count"]),
        "n_hits_canonical_unique": int(hit_source["canonical_unique_count"]),
        "n_hits_in_universe": int(len(hit_set)),
        "hit_genes_in_universe": sorted(hit_set),
        "hit_genes_not_in_universe": sorted(set(canonical_hits) - universe_set),
    }

    method_results: dict[str, dict] = {}
    for method_idx, score_column in enumerate(["chronoabor_score", "sig_max_auc_score"]):
        curve = compute_enrichment_curve(scored_df, score_column, hit_set)
        nes, p_value, permutation_es = compute_nes_and_pvalue(
            scores=curve["scores"],
            hit_count=int(curve["hit_mask"].sum()),
            observed_es=float(curve["es"]),
            seed=RANDOM_SEED + method_idx,
        )
        curve["nes"] = nes
        curve["p_value"] = p_value
        curve["permutation_es"] = permutation_es
        curve["hit_set"] = sorted(hit_set)
        method_results[score_column] = curve

    create_figure(method_results, hit_summary)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "permutation_count": PERMUTATION_COUNT,
        "weight_exponent": WEIGHT_EXPONENT,
        "dataset_summary": dataset_summary,
        "proteomics_hits": hit_summary,
        "methods": {
            "chronoabor": format_summary(method_results["chronoabor_score"]),
            "sig_max_auc": format_summary(method_results["sig_max_auc_score"]),
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))

    print("NM Experiment P7: Ranked enrichment")
    print(
        f"Proteomics hits in universe: {hit_summary['n_hits_in_universe']} "
        f"of {hit_summary['n_hits_canonical_unique']} canonical requested proteins"
    )
    print(f"P5 source used: {hit_summary['source']}")
    print()
    for method_name in ["chronoabor", "sig_max_auc"]:
        summary = results["methods"][method_name]
        print(
            f"{method_name}: ES={summary['es']:.3f}, NES={summary['nes']:.3f}, "
            f"p={summary['p_value']:.4f}, leading_edge_size={summary['leading_edge_size']}"
        )
        print(f"  Leading edge: {', '.join(summary['leading_edge_genes']) or 'None'}")
    print()
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
