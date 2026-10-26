#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Patch

FIGURES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGURES_DIR))

from _common import (
    CORAL, STEELBLUE, DOUBLE_COLUMN_WIDTH,
    setup_style, polish_axes, save_pdf,
)

ROOT_DIR = FIGURES_DIR.parent / "analysis"
ROLLING_JSON = ROOT_DIR / "results" / "nm_exp_p9_rolling_origin.json"
E8_JSON = ROOT_DIR / "results" / "rev_exp_e8_spec_model.json"

BLUE = "#0072B2"
RED_ = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#AAAAAA"
TOP10_SHADE = "#F6E8A8"
GRID = "#E0E0E0"
TEXT = "#3A3A3A"
CUTOFF_COLOR = "#555555"


def _label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)


def _draw_trajectories(ax):
    years = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2021, 2022, 2024])
    tp = np.array([2, 4, 6, 9, 12, 15, 16, 17, 20], dtype=float)
    fp = np.array([1, 1, 2, 2, 2, 3, 4, 10, 18], dtype=float)
    nc = np.array([1, 2, 3, 4, 6, 8, 8.5, 9, 10], dtype=float)

    ax.plot(years, tp, color=BLUE, linewidth=1.8, linestyle="-", label="True positive (e.g. GFAP)", zorder=3)
    ax.plot(years, fp, color=RED_, linewidth=1.8, linestyle="--", label="Literature-inflated FP", zorder=3)
    ax.plot(years, nc, color=GREEN, linewidth=1.8, linestyle="-", label="Non-canonical (YWHAZ)", zorder=3)

    ax.axvline(2021, color=CUTOFF_COLOR, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(2020.7, 22.5, "Cutoff\n2021", ha="right", va="bottom", fontsize=7.5, color=CUTOFF_COLOR)
    ax.axvspan(2021, 2024, color="#AAA", alpha=0.10, zorder=0)
    ax.text(2022.5, 0.5, "Post-cutoff\n(withheld)", ha="center", va="bottom", fontsize=7, color="#777", style="italic")

    ax.set_xlim(2010, 2024)
    ax.set_ylim(0, 23)
    ax.set_xticks([2010, 2014, 2018, 2021, 2024])
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Cumulative evidence count", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, bbox_to_anchor=(0.0, 0.95))
    polish_axes(ax, grid_axis="y")


METHOD_ORDER = ["full_model", "is_secreted", "num_studies", "sig_max_auc", "prior_only"]
METHOD_LABELS = {"full_model": "ChronoABOR", "is_secreted": "Secretion\nstatus", "num_studies": "Publication\ncount", "sig_max_auc": "Max sig.\nAUC", "prior_only": "Prior only"}
CUTOFF_COLORS = {2019: "#0072B2", 2020: "#56B4E9", 2021: "#91CFF0"}

FULL_UNIPROT = {2019: (0.981, 0.036), 2020: (0.977, 0.035), 2021: (0.977, 0.035)}


def _draw_rolling(ax):
    lookup = {}
    if ROLLING_JSON.exists():
        data = json.loads(ROLLING_JSON.read_text())
        for fold in data["folds"]:
            lookup[fold["cutoff_year"]] = {m["method"]: m for m in fold["methods"]}
    else:
        fb = {
            2019: {"is_secreted": (0.500, 0.5, 0.5), "num_studies": (0.642, 0.474, 0.811), "sig_max_auc": (0.608, 0.448, 0.769), "prior_only": (0.675, 0.475, 0.846)},
            2020: {"is_secreted": (0.500, 0.5, 0.5), "num_studies": (0.661, 0.49, 0.83), "sig_max_auc": (0.635, 0.46, 0.80), "prior_only": (0.696, 0.50, 0.87)},
            2021: {"is_secreted": (0.500, 0.5, 0.5), "num_studies": (0.638, 0.47, 0.81), "sig_max_auc": (0.635, 0.46, 0.80), "prior_only": (0.689, 0.49, 0.86)},
        }
        for yr, d in fb.items():
            lookup[yr] = {m: {"cv_auroc": v[0], "bootstrap_ci": {"lower": v[1], "upper": v[2]}} for m, v in d.items()}

    cutoffs = [2019, 2020, 2021]
    n_methods = len(METHOD_ORDER)
    n_folds = len(cutoffs)
    group_w = 0.68
    width = group_w / n_folds
    offsets = np.linspace(-(group_w - width) / 2, (group_w - width) / 2, n_folds)
    x = np.arange(n_methods)

    for fi, year in enumerate(cutoffs):
        color = CUTOFF_COLORS[year]
        label = f"Features ≤{year}"
        aucs, lo_err, hi_err = [], [], []
        for meth in METHOD_ORDER:
            if meth == "full_model":
                auc, sd = FULL_UNIPROT[year]
                aucs.append(auc); lo_err.append(sd); hi_err.append(sd)
            else:
                row = lookup[year][meth]
                auc = row["cv_auroc"]
                aucs.append(auc)
                lo_err.append(max(auc - row["bootstrap_ci"]["lower"], 0.0))
                hi_err.append(max(row["bootstrap_ci"]["upper"] - auc, 0.0))

        bar_x = x + offsets[fi]
        for bi in range(n_methods):
            ax.bar(bar_x[bi], aucs[bi], width=width, yerr=[[lo_err[bi]], [hi_err[bi]]], capsize=2, color=color, edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#444"}, label=label if bi == 0 else "_nolegend_", zorder=3)
        for bi in range(n_methods):
            label_y = min(aucs[bi] + hi_err[bi] + 0.02, 1.16)
            ax.text(bar_x[bi] - 0.04, label_y, f"{aucs[bi]:.3f}", ha="left", va="bottom", fontsize=7, color="#333", rotation=85)

    ax.axhline(0.5, color="#6c757d", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m].replace("\n", " ") for m in METHOD_ORDER], rotation=30, ha="right", rotation_mode="anchor", fontsize=8.5)
    ax.set_ylabel("Within-secreted AUROC", fontsize=9)
    ax.set_ylim(0, 1.34)
    polish_axes(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.0)


def _r3(v):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _load_jackknife():
    if E8_JSON.exists():
        data = json.loads(E8_JSON.read_text())
        jk = data["primary_secreted_definition"]["within_secreted"]["xgb_results_by_learning_rate"]["lr_0.05"]["jackknife_leave_one_positive_out"]
        return jk["iterations"], jk["summary"]
    iterations = [
        {"held_out_gene": "APP", "held_out_rank": 3}, {"held_out_gene": "APOE", "held_out_rank": 8},
        {"held_out_gene": "CLU", "held_out_rank": 8}, {"held_out_gene": "MAPT", "held_out_rank": 8},
        {"held_out_gene": "BDNF", "held_out_rank": 22},
    ]
    summary = {"n_iterations": 5, "mean_rank": 9.8, "median_rank": 8.0, "fraction_top_10": 0.8, "std_auroc": 0.0055, "mean_auroc": 0.9951}
    return iterations, summary


def _draw_jackknife(ax):
    iterations, summary = _load_jackknife()
    rows = sorted(iterations, key=lambda r: (r["held_out_rank"], r["held_out_gene"]))
    genes = [r["held_out_gene"] for r in rows]
    ranks = np.asarray([r["held_out_rank"] for r in rows], dtype=float)
    y = np.arange(len(rows))

    ax.axvspan(1, 10, color=TOP10_SHADE, alpha=0.40, zorder=0)
    dot_colors = [GREEN if r <= 10 else (ORANGE if r <= 25 else GRAY) for r in ranks]
    for yi, r in zip(y, ranks):
        ax.plot([0, r], [yi, yi], color="#DDD", linewidth=1.0, zorder=1)
    ax.scatter(ranks, y, s=60, color=dot_colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(summary["mean_rank"], color=BLUE, linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(summary["median_rank"], color=ORANGE, linestyle=":", linewidth=1.2, zorder=2)

    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Held-out rank among 246 secreted genes (1=best)", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontstyle("italic")
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats = f"n={summary['n_iterations']}  Mean={summary['mean_rank']:.1f}  Med={summary['median_rank']:.0f}  Top-10: {summary['fraction_top_10']:.0%}  AUROC SD={_r3(summary['std_auroc'])}"
    ax.set_title(stats, fontsize=7, color=TEXT, pad=4)


def _draw_bootstrap(ax):
    full, base = 0.7826, 0.6255
    delta, lo, hi, p = 0.157, 0.032, 0.282, 0.009
    if ROLLING_JSON.exists():
        d = json.loads(ROLLING_JSON.read_text())["pooled_paired_bootstrap_full_model_vs_sig_max_auc"]
        full, base = d["observed_full_model_auroc"], d["observed_sig_max_auc_auroc"]
        delta, lo, hi, p = d["observed_delta"], d["bootstrap_ci"]["lower"], d["bootstrap_ci"]["upper"], d["paired_p_value_one_sided"]

    x = np.array([0, 1])
    vals = np.array([full, base])
    bars = ax.bar(x, vals, width=0.5, color=[BLUE, GRAY], edgecolor="white", linewidth=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full model", "Max signature AUC"], fontsize=8.5)
    ax.set_ylabel("Pooled AUROC (3 windows)", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    polish_axes(ax, grid_axis="y")


def main():
    setup_style()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 9.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.50, left=0.12, right=0.96, top=0.97, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_trajectories(ax_a)
    _draw_rolling(ax_b)
    _draw_jackknife(ax_c)
    _draw_bootstrap(ax_d)

    _label(ax_a, "a", x=-0.14)
    _label(ax_b, "b")
    _label(ax_c, "c", x=-0.14)
    _label(ax_d, "d")

    save_pdf(fig, "fig_supp_composite")


if __name__ == "__main__":
    main()
