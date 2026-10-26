#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl

from nm_exp4_full_universe import RANDOM_SEED, to_jsonable
from nm_exp_p6_secreted_ablation import summarize_distribution
from nm_exp_p7_ranked_enrichment import (
    FEATURE_CUTOFF_YEAR,
    WEIGHT_EXPONENT,
    build_fold_c_frame,
    train_fold_c_model,
)

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
INPUT_P7_JSON = RESULTS_DIR / "nm_exp_p7_ranked_enrichment.json"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p7b_matched_enrichment.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p7b_matched_enrichment.png"
OUTPUT_PDF = RESULTS_DIR / "nm_exp_p7b_matched_enrichment.pdf"

MATCHED_PERMUTATIONS = 1000


def compute_enrichment_fixed_ranking(
    ranked_genes: np.ndarray,
    ranked_scores: np.ndarray,
    hit_set: set[str],
) -> dict:
    hit_mask = np.isin(ranked_genes, list(hit_set))
    if hit_mask.sum() == 0:
        raise RuntimeError("No hits overlap the ranked universe.")

    hit_weights = np.power(np.abs(ranked_scores[hit_mask]), WEIGHT_EXPONENT)
    hit_weight_sum = float(hit_weights.sum())
    if hit_weight_sum <= 0:
        hit_weight_sum = 1.0
    miss_count = int((~hit_mask).sum())
    miss_penalty = 1.0 / miss_count if miss_count > 0 else 0.0

    running_es = np.empty(len(ranked_scores), dtype=float)
    running_value = 0.0
    for idx, (is_hit, score) in enumerate(zip(hit_mask, ranked_scores)):
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
        leading_edge_genes = ranked_genes[: peak_idx + 1][hit_mask[: peak_idx + 1]].tolist()
    else:
        leading_edge_genes = ranked_genes[peak_idx:][hit_mask[peak_idx:]].tolist()

    return {
        "es": es,
        "peak_index": peak_idx,
        "running_es": running_es,
        "hit_mask": hit_mask,
        "leading_edge_genes": leading_edge_genes,
    }


def load_existing_p7_summary() -> dict:
    if not INPUT_P7_JSON.exists():
        raise FileNotFoundError(f"Missing prerequisite file: {INPUT_P7_JSON}")
    return json.loads(INPUT_P7_JSON.read_text())


def build_scored_universe() -> tuple[pd.DataFrame, dict]:
    feature_df, dataset_summary = build_fold_c_frame()
    scored_df = train_fold_c_model(feature_df)
    scored_df["has_signature"] = (scored_df["sig_count_all"] > 0).astype(int)
    return scored_df, dataset_summary


def build_match_pools(
    scored_df: pd.DataFrame,
    hit_set: set[str],
    study_low: float = 0.5,
    study_high: float = 1.5,
    use_signature: bool = True,
    ignore_secreted: bool = False,
) -> tuple[list[dict], dict]:
    non_hits = scored_df.loc[~scored_df["gene"].isin(hit_set)].copy().reset_index(drop=True)
    hit_df = scored_df.loc[scored_df["gene"].isin(hit_set)].copy().sort_values("gene").reset_index(drop=True)

    pools: list[dict] = []
    for _, row in hit_df.iterrows():
        num_studies = int(row["num_studies"])
        study_min = int(math.floor(num_studies * study_low))
        study_max = int(math.ceil(num_studies * study_high))
        mask = non_hits["num_studies"].between(study_min, study_max)
        if not ignore_secreted:
            mask &= non_hits["is_secreted"] == int(row["is_secreted"])
        if use_signature:
            mask &= non_hits["has_signature"] == int(row["has_signature"])
        candidate_pool = non_hits.loc[mask, "gene"].sort_values().tolist()

        if not candidate_pool:
            raise RuntimeError(
                f"No matched null pool available for hit gene {row['gene']} "
                f"(is_secreted={int(row['is_secreted'])}, num_studies={num_studies}, "
                f"has_signature={int(row['has_signature'])})."
            )

        pools.append(
            {
                "hit_gene": str(row["gene"]),
                "is_secreted": int(row["is_secreted"]),
                "num_studies": num_studies,
                "sig_max_auc": float(row["sig_max_auc"]),
                "has_signature": int(row["has_signature"]),
                "study_range_min": study_min,
                "study_range_max": study_max,
                "candidate_pool": candidate_pool,
            }
        )

    pool_sizes = np.asarray([len(item["candidate_pool"]) for item in pools], dtype=float)
    diagnostics = {
        "n_hits": int(len(pools)),
        "n_non_hit_genes": int(len(non_hits)),
        "study_low_factor": float(study_low),
        "study_high_factor": float(study_high),
        "use_signature": bool(use_signature),
        "ignore_secreted": bool(ignore_secreted),
        "n_unique_candidate_genes_across_all_pools": int(len({gene for item in pools for gene in item["candidate_pool"]})),
        "pool_size_summary": summarize_distribution(pool_sizes),
        "per_hit_match_pools": [
            {
                "hit_gene": item["hit_gene"],
                "is_secreted": item["is_secreted"],
                "num_studies": item["num_studies"],
                "sig_max_auc": item["sig_max_auc"],
                "has_signature": item["has_signature"],
                "study_range_min": item["study_range_min"],
                "study_range_max": item["study_range_max"],
                "candidate_pool_size": int(len(item["candidate_pool"])),
                "candidate_pool_preview": item["candidate_pool"][:20],
                "candidate_pool_preview_truncated": bool(len(item["candidate_pool"]) > 20),
            }
            for item in pools
        ],
    }
    return pools, diagnostics


TOLERANCE_LEVELS: list[dict] = [
    {
        "label": "strict: num_studies in [0.5x, 1.5x], has_signature exact, is_secreted exact",
        "study_low": 0.5,
        "study_high": 1.5,
        "use_signature": True,
        "ignore_secreted": False,
    },
    {
        "label": "relaxed-1: num_studies in [0.33x, 2.0x], has_signature exact, is_secreted exact",
        "study_low": 1.0 / 3.0,
        "study_high": 2.0,
        "use_signature": True,
        "ignore_secreted": False,
    },
    {
        "label": "relaxed-2: num_studies in [0.25x, 3.0x], has_signature exact, is_secreted exact",
        "study_low": 0.25,
        "study_high": 3.0,
        "use_signature": True,
        "ignore_secreted": False,
    },
    {
        "label": "relaxed-3: num_studies in [0.25x, 3.0x], is_secreted exact (has_signature dropped)",
        "study_low": 0.25,
        "study_high": 3.0,
        "use_signature": False,
        "ignore_secreted": False,
    },
    {
        "label": "relaxed-4: is_secreted exact only (num_studies and has_signature dropped)",
        "study_low": 0.0,
        "study_high": float("inf"),
        "use_signature": False,
        "ignore_secreted": False,
    },
    {
        "label": "relaxed-5: any non-hit gene (all matching constraints dropped)",
        "study_low": 0.0,
        "study_high": float("inf"),
        "use_signature": False,
        "ignore_secreted": True,
    },
]


def matching_exists(match_pools: list[dict]) -> bool:
    adjacency = {item["hit_gene"]: item["candidate_pool"] for item in match_pools}
    matched_gene_to_hit: dict[str, str] = {}

    def dfs(hit_gene: str, seen: set[str]) -> bool:
        for candidate_gene in adjacency[hit_gene]:
            if candidate_gene in seen:
                continue
            seen.add(candidate_gene)
            previous_hit = matched_gene_to_hit.get(candidate_gene)
            if previous_hit is None or dfs(previous_hit, seen):
                matched_gene_to_hit[candidate_gene] = hit_gene
                return True
        return False

    for hit_gene in sorted(adjacency, key=lambda h: len(adjacency[h])):
        if not dfs(hit_gene, set()):
            return False
    return True


def build_feasible_match_pools(scored_df: pd.DataFrame, hit_set: set[str]) -> tuple[list[dict], dict]:
    failures: list[dict] = []
    for level in TOLERANCE_LEVELS:
        try:
            pools, diagnostics = build_match_pools(
                scored_df,
                hit_set,
                study_low=level["study_low"],
                study_high=level["study_high"],
                use_signature=level["use_signature"],
                ignore_secreted=level["ignore_secreted"],
            )
        except RuntimeError as err:
            failures.append({"level": level["label"], "error": str(err)})
            continue
        if matching_exists(pools):
            diagnostics["tolerance_level"] = level["label"]
            diagnostics["infeasible_stricter_levels"] = failures
            return pools, diagnostics
        failures.append({"level": level["label"], "error": "no perfect matching exists"})
    raise RuntimeError(
        "Unable to construct feasible matched-null pools at any tolerance level: "
        + json.dumps(failures)
    )


def sample_unique_matched_set(match_pools: list[dict], rng: np.random.Generator) -> tuple[set[str], dict[str, str]]:
    adjacency: dict[str, list[str]] = {}
    ordering_rows: list[tuple[int, float, str]] = []
    for item in match_pools:
        shuffled = rng.permutation(item["candidate_pool"]).tolist()
        adjacency[item["hit_gene"]] = shuffled
        ordering_rows.append((len(shuffled), float(rng.random()), item["hit_gene"]))
    hit_order = [row[2] for row in sorted(ordering_rows)]

    matched_gene_to_hit: dict[str, str] = {}

    def dfs(hit_gene: str, seen: set[str]) -> bool:
        for candidate_gene in adjacency[hit_gene]:
            if candidate_gene in seen:
                continue
            seen.add(candidate_gene)
            previous_hit = matched_gene_to_hit.get(candidate_gene)
            if previous_hit is None or dfs(previous_hit, seen):
                matched_gene_to_hit[candidate_gene] = hit_gene
                return True
        return False

    for hit_gene in hit_order:
        if not dfs(hit_gene, set()):
            raise RuntimeError(f"Unable to sample a unique matched-null set for hit gene {hit_gene}.")

    hit_to_gene = {hit_gene: candidate_gene for candidate_gene, hit_gene in matched_gene_to_hit.items()}
    return set(matched_gene_to_hit), hit_to_gene


def compute_matched_null(
    ranked_genes: np.ndarray,
    ranked_scores: np.ndarray,
    match_pools: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(RANDOM_SEED)
    matched_es = np.empty(MATCHED_PERMUTATIONS, dtype=float)
    previews: list[dict] = []

    for idx in range(MATCHED_PERMUTATIONS):
        matched_set, hit_to_gene = None, None
        last_err = None
        for _attempt in range(60):
            try:
                matched_set, hit_to_gene = sample_unique_matched_set(match_pools, rng)
                break
            except RuntimeError as err:
                last_err = err
        if matched_set is None:
            raise RuntimeError(
                "Matched-null sampling failed after 60 attempts: " + str(last_err)
            )
        curve = compute_enrichment_fixed_ranking(ranked_genes, ranked_scores, matched_set)
        matched_es[idx] = float(curve["es"])
        if idx < 5:
            previews.append(
                {
                    "iteration": int(idx + 1),
                    "matched_gene_set": sorted(matched_set),
                    "hit_to_matched_gene": dict(sorted(hit_to_gene.items())),
                    "es": float(curve["es"]),
                }
            )
    return matched_es, previews


def compute_nes_and_pvalue(observed_es: float, null_es: np.ndarray) -> tuple[float, float]:
    if observed_es >= 0:
        same_sign = null_es[null_es >= 0]
        if len(same_sign) == 0:
            same_sign = np.array([1.0], dtype=float)
        nes = float(observed_es / np.mean(same_sign))
        p_value = float((1 + np.sum(null_es >= observed_es)) / (len(null_es) + 1))
    else:
        same_sign = np.abs(null_es[null_es < 0])
        if len(same_sign) == 0:
            same_sign = np.array([1.0], dtype=float)
        nes = float(observed_es / np.mean(same_sign))
        p_value = float((1 + np.sum(null_es <= observed_es)) / (len(null_es) + 1))
    return nes, p_value


def create_figure(
    ranked_genes: np.ndarray,
    observed_curve: dict,
    matched_es: np.ndarray,
    original_summary: dict,
    matched_nes: float,
    matched_p_value: float,
    matching_diagnostics: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    x = np.arange(1, len(ranked_genes) + 1)
    ax.plot(x, observed_curve["running_es"], color="#0072B2", linewidth=2.0)
    hit_positions = x[observed_curve["hit_mask"]]
    ax.vlines(
        hit_positions,
        ymin=min(observed_curve["running_es"]) - 0.02,
        ymax=min(observed_curve["running_es"]) + 0.04,
        color="#009E73",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="#6c757d", linestyle="--", linewidth=1.0)
    ax.axvline(observed_curve["peak_index"] + 1, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.scatter(
        [observed_curve["peak_index"] + 1],
        [observed_curve["running_es"][observed_curve["peak_index"]]],
        color="#D55E00",
        s=24,
        zorder=3,
    )
    ax.set_xlabel("Rank in 2,843-gene universe")
    ax.set_ylabel("Running enrichment score")
    ax.set_title("Observed ChronoABOR enrichment curve")
    ax.text(
        0.02,
        0.98,
        (
            f"Original NES={original_summary['nes']:.3f}, p={original_summary['p_value']:.3f}\n"
            f"Matched NES={matched_nes:.3f}, p={matched_p_value:.3f}\n"
            f"Leading edge size={len(observed_curve['leading_edge_genes'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    ax = axes[1]
    ax.hist(matched_es, bins=35, color="#56B4E9", edgecolor="#1f2933", linewidth=0.8, alpha=0.9)
    ax.axvline(original_summary["es"], color="#D55E00", linewidth=2.0, label=f"Observed ES = {original_summary['es']:.3f}")
    ax.axvline(
        np.mean(matched_es),
        color="#0072B2",
        linestyle="--",
        linewidth=1.5,
        label=f"Matched-null mean = {np.mean(matched_es):.3f}",
    )
    ax.set_xlabel("Matched-null ES")
    ax.set_ylabel("Permutation count")
    ax.set_title("Matched-null distribution (1,000 permutations)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.02,
        0.98,
        (
            f"Pool size mean={matching_diagnostics['pool_size_summary']['mean']:.1f}\n"
            f"Pool size min={matching_diagnostics['pool_size_summary']['p025']:.1f} "
            f"(2.5th pct)\n"
            f"Unique candidates={matching_diagnostics['n_unique_candidate_genes_across_all_pools']}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95},
    )

    fig.suptitle("NM Experiment P7b: Matched-null ranked enrichment", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing_p7 = load_existing_p7_summary()
    original_summary = existing_p7["methods"]["chronoabor"]
    hit_set = set(existing_p7["proteomics_hits"]["hit_genes_in_universe"])

    scored_df, dataset_summary = build_scored_universe()
    ranked_df = scored_df.sort_values(
        by=["chronoabor_score", "num_studies", "gene"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked_genes = ranked_df["gene"].to_numpy(dtype=str)
    ranked_scores = ranked_df["chronoabor_score"].to_numpy(dtype=float)
    observed_curve = compute_enrichment_fixed_ranking(ranked_genes, ranked_scores, hit_set)

    match_pools, matching_diagnostics = build_feasible_match_pools(scored_df, hit_set)
    matched_es, preview_sets = compute_matched_null(ranked_genes, ranked_scores, match_pools)
    matched_nes, matched_p_value = compute_nes_and_pvalue(float(observed_curve["es"]), matched_es)

    results = {
        "random_seed": RANDOM_SEED,
        "feature_cutoff_year": FEATURE_CUTOFF_YEAR,
        "matched_permutation_count": MATCHED_PERMUTATIONS,
        "score_source": {
            "existing_p7_json": str(INPUT_P7_JSON),
            "existing_p7_json_contains_per_gene_scores": False,
            "ranking_rebuilt_from_shared_p7_pipeline": True,
            "observed_es_recomputed": float(observed_curve["es"]),
            "observed_es_from_existing_p7": float(original_summary["es"]),
            "observed_es_delta_vs_existing_p7": float(observed_curve["es"] - original_summary["es"]),
        },
        "dataset_summary": dataset_summary,
        "proteomics_hits": existing_p7["proteomics_hits"],
        "matching_criteria": {
            "tolerance_level_used": matching_diagnostics["tolerance_level"],
            "is_secreted": "Exact match" if not matching_diagnostics["ignore_secreted"] else "Dropped (infeasible at stricter levels)",
            "has_signature": "Exact match using sig_count_all > 0" if matching_diagnostics["use_signature"] else "Dropped (infeasible at stricter levels)",
            "num_studies": (
                f"Inclusive integer range [floor({matching_diagnostics['study_low_factor']:.4g} * num_studies), "
                f"ceil({matching_diagnostics['study_high_factor']:.4g} * num_studies)]"
            ),
            "candidate_space": "Non-hit genes only",
            "sampling": "One unique matched gene sampled per hit gene in each permutation",
            "feasibility": "Strictest tolerance level admitting a perfect matching is selected deterministically before permutation sampling",
        },
        "matching_diagnostics": matching_diagnostics,
        "observed_enrichment": {
            "es": float(observed_curve["es"]),
            "peak_index_1based": int(observed_curve["peak_index"] + 1),
            "leading_edge_genes": observed_curve["leading_edge_genes"],
            "leading_edge_size": int(len(observed_curve["leading_edge_genes"])),
            "original_nes_from_p7": float(original_summary["nes"]),
            "original_p_value_from_p7": float(original_summary["p_value"]),
        },
        "matched_null": {
            "matched_nes": float(matched_nes),
            "matched_p_value": float(matched_p_value),
            "null_distribution": {
                **summarize_distribution(matched_es),
                "ci95": {
                    "lower": float(np.quantile(matched_es, 0.025)),
                    "upper": float(np.quantile(matched_es, 0.975)),
                },
            },
            "permutation_es": matched_es.tolist(),
            "sampled_gene_sets_preview": preview_sets,
        },
        "runtime_seconds": float(time.time() - start_time),
    }
    OUTPUT_JSON.write_text(json.dumps(to_jsonable(results), indent=2))
    create_figure(
        ranked_genes=ranked_genes,
        observed_curve=observed_curve,
        matched_es=matched_es,
        original_summary=original_summary,
        matched_nes=matched_nes,
        matched_p_value=matched_p_value,
        matching_diagnostics=matching_diagnostics,
    )

    print("NM Experiment P7b: Matched-null ranked enrichment")
    print(f"Original NES / p-value: {original_summary['nes']:.3f} / {original_summary['p_value']:.4f}")
    print(f"Matched NES / p-value: {matched_nes:.3f} / {matched_p_value:.4f}")
    print(
        f"Matched pool sizes: mean={matching_diagnostics['pool_size_summary']['mean']:.2f}, "
        f"min={min(item['candidate_pool_size'] for item in matching_diagnostics['per_hit_match_pools'])}, "
        f"max={max(item['candidate_pool_size'] for item in matching_diagnostics['per_hit_match_pools'])}"
    )
    print(f"Results JSON: {OUTPUT_JSON}")
    print(f"Plot PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
