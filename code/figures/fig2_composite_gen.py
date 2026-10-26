#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Patch

from _common import (
    ANNOT_BOX, DOUBLE_COLUMN_WIDTH,
    FOLD_C_COLOR, FOLD_D_COLOR, BASELINE_COLOR, PRIOR_COLOR,
    HIGHLIGHT_COLOR, CORAL, STEELBLUE, DARKORANGE, RED, GRAY,
    setup_style, polish_axes, save_pdf,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "results"
E12 = json.load(open(RESULTS_DIR / "rev_exp_e12_strict_windows.json"))["benchmarks"]
TOPK_BOOT = json.load(open(Path(__file__).resolve().parent / "strict_fold_c_topk_bootstrap.json"))


def _repeat_aurocs(fold, model):
    return [r["auroc_mean"] for r in E12[fold][model]["repeats"]]


REPEATS = {
    "Logistic Reg.": (_repeat_aurocs("strict_fold_c", "logistic"), _repeat_aurocs("strict_fold_d", "logistic")),
    "XGBoost": (_repeat_aurocs("strict_fold_c", "xgb"), _repeat_aurocs("strict_fold_d", "xgb")),
}


def _label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)


def _draw_auroc(ax):
    ROWS = [
        ("Logistic Reg.",    0.930, 0.004, 0.778, 0.022, "ml"),
        ("XGBoost",          0.882, 0.029, 0.732, 0.024, "ml"),
        ("GPT-5.4",          0.825, 0.000, 0.701, 0.000, "llm"),
        ("GPT-5.2",          0.811, 0.000, 0.693, 0.000, "llm"),
        ("Secretion status", 0.808, 0.000, 0.682, 0.000, "prior"),
        ("Citation freq.",   0.646, 0.000, 0.546, 0.000, "baseline"),
        ("Random",           0.500, 0.000, 0.500, 0.000, "baseline"),
    ]

    n = len(ROWS)
    bar_h = 0.30
    gap = 0.06
    y_positions = np.arange(n) * (2 * bar_h + gap + 0.12)

    def _lighten(hx, a=0.35):
        hx = hx.lstrip("#")
        r, g, b = (int(hx[i:i + 2], 16) for i in (0, 2, 4))
        return f"#{int(r + (255 - r) * a):02X}{int(g + (255 - g) * a):02X}{int(b + (255 - b) * a):02X}"

    def _color(grp, fold):
        if grp == "ml":
            return FOLD_C_COLOR if fold == "c" else FOLD_D_COLOR
        base = PRIOR_COLOR if grp == "prior" else ("#CC79A7" if grp == "llm" else BASELINE_COLOR)
        return base if fold == "c" else _lighten(base)

    for i, (lbl, fc, fce, fd, fde, grp) in enumerate(ROWS):
        yc = y_positions[i]
        yd = yc + bar_h + gap
        ax.barh(yc, fc, height=bar_h, color=_color(grp, "c"), edgecolor="white", linewidth=0.6, zorder=3)
        ax.barh(yd, fd, height=bar_h, color=_color(grp, "d"), edgecolor="white", linewidth=0.6, zorder=3)
        right_c, right_d = fc, fd
        if fce > 0:
            ax.errorbar(fc, yc, xerr=fce, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2, capthick=0.8, zorder=4)
            ax.errorbar(fd, yd, xerr=fde, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2, capthick=0.8, zorder=4)
            reps_c, reps_d = REPEATS[lbl]
            jitter = np.array([-0.07, 0.0, 0.07])
            ax.scatter(reps_c, yc + jitter, s=9, facecolor="white", edgecolor="#1F1F1F", linewidth=0.6, zorder=5)
            ax.scatter(reps_d, yd + jitter, s=9, facecolor="white", edgecolor="#1F1F1F", linewidth=0.6, zorder=5)
            right_c = max(fc + fce, max(reps_c))
            right_d = max(fd + fde, max(reps_d))
        right_c, right_d = right_c + 0.006, right_d + 0.006
        ax.text(right_c + 0.008, yc, f"{fc:.3f}", va="center", ha="left", fontsize=7.5, color="#1F1F1F", fontweight="bold" if i == 0 else "normal")
        ax.text(right_d + 0.008, yd, f"{fd:.3f}", va="center", ha="left", fontsize=7.5, color="#1F1F1F")

    mid = y_positions + bar_h + gap / 2
    ax.set_yticks(mid)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=8)
    for t in ax.get_yticklabels():
        if t.get_text() == ROWS[0][0]:
            t.set_fontweight("bold")
            t.set_color(HIGHLIGHT_COLOR)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#666", linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.set_xlim(0.46, 1.08)
    ax.set_xlabel("AUROC", fontsize=9)
    polish_axes(ax, grid_axis="x")
    ax.legend(handles=[
        Patch(facecolor=FOLD_C_COLOR, edgecolor="white", label="Fold C"),
        Patch(facecolor=FOLD_D_COLOR, edgecolor="white", label="Fold D"),
    ], frameon=False, loc="lower right", fontsize=7, handlelength=1.0, labelspacing=0.3)


def _draw_enrichment(ax):
    stored = E12["strict_fold_c"]["xgb"]["full_fit_rank_metrics"]
    rows = TOPK_BOOT["rows"]
    k_vals = np.array([r["k"] for r in rows])
    assert all(stored[f"top{k}"]["hits"] == r["hits"] for k, r in zip(k_vals, rows))
    prec = np.array([stored[f"top{k}"]["precision"] for k in k_vals])
    ci_lo = np.array([r["precision_bootstrap_ci95"]["lower"] for r in rows])
    ci_hi = np.array([r["precision_bootstrap_ci95"]["upper"] for r in rows])
    ef = np.array([stored[f"top{k}"]["enrichment_factor"] for k in k_vals])
    baseline = E12["strict_fold_c"]["summary"]["n_positives"] / E12["strict_fold_c"]["summary"]["n_genes"]

    x = np.arange(len(k_vals))
    yerr = np.vstack((prec - ci_lo, ci_hi - prec))
    dot_cols = ["#1A6FA8", "#2A83C2", "#56B4E9", "#91CFF0"]

    ax.errorbar(x, prec, yerr=yerr, fmt="none", ecolor="#444", elinewidth=0.9, capsize=3, capthick=0.9, zorder=3)
    ax.scatter(x, prec, s=70, c=dot_cols, edgecolor="#1F1F1F", linewidth=0.6, zorder=4)
    ax.set_xlim(-0.5, len(k_vals) - 0.5)
    ax.set_xticks(x, [f"Top {v}" for v in k_vals], fontsize=8)
    ax.set_xlabel("Top-k threshold", fontsize=9)
    ax.set_ylabel("Precision@k", fontsize=9)
    ax.set_ylim(0.0, 1.22)
    polish_axes(ax, grid_axis="y")
    ax.axhline(baseline, color="#666", linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)

    for i, (xv, val, upper) in enumerate(zip(x, prec, yerr[1])):
        ax.text(xv, min(val + upper + 0.06, 1.16), f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#1F1F1F", fontweight="bold")
        if i == 1:
            ax.text(xv, min(val + upper + 0.06, 1.16) + 0.05, r"$P=1.5\times10^{-14}$", ha="center", va="bottom", fontsize=7, color="#333", fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, ef, color=CORAL, marker="o", markersize=5, linewidth=2.0, zorder=4, markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_ylabel("Enrichment factor", color=CORAL, fontsize=9)
    ax2.set_ylim(0, 160)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(CORAL)
    ax2.tick_params(axis="y", colors=CORAL, labelsize=8)
    ax2.grid(False)
    for xv, e in zip(x, ef):
        ax2.text(xv + 0.09, e + 4, f"{e:.0f}×", ha="left", va="bottom", fontsize=8, color=CORAL, fontweight="bold")


def _draw_recovery(ax):
    genes = ["APP", "APOE", "MAPT", "TREM2", "BACE1", "GFAP", "NEFL", "SNAP25"]
    ranks = np.array([1, 2, 3, 5, 10, 933, 1442, 2022])
    colors = plt.cm.Blues(np.linspace(0.85, 0.45, len(genes)))

    y = np.arange(len(genes))
    ax.set_xscale("log")
    ax.axvspan(0.7, 10.5, color="#F6E8A8", alpha=0.45, zorder=0)
    for i, (gene, rank, col) in enumerate(zip(genes, ranks, colors)):
        recovered = rank <= 10
        ax.plot([0.7, rank], [y[i], y[i]], color="#CCC", linewidth=1.0, zorder=1, solid_capstyle="round")
        dot_size = 180 if rank == 1 else 130
        ax.scatter(rank, y[i], s=dot_size, color=col if recovered else GRAY, edgecolor="white", linewidth=1.2, zorder=3)
        txt = str(rank) if recovered else f"{rank:,}"
        if recovered:
            ax.text(rank, y[i], txt, ha="center", va="center", fontsize=8 if rank < 10 else 7, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(rank * 1.25, y[i], txt, ha="left", va="center", fontsize=7.5, color="#555", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=8.5)
    for lbl in ax.get_yticklabels():
        lbl.set_fontstyle("italic")
        lbl.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0.7, 6000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Rank among 2,843 genes (log scale)", fontsize=9)
    polish_axes(ax, grid_axis="x")
    ax.text(0.97, 0.97, "$P=9.1\\times10^{-12}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)


def _draw_ablation(ax):
    labels = ["All 16 features", "No DEG block", "No other-layer block", "Temporal only", "Prior only"]
    values = np.array([0.882, 0.760, 0.870, 0.793, 0.804])
    colors = [HIGHLIGHT_COLOR, STEELBLUE, "#91CFF0", DARKORANGE, RED]
    x_left = 0.48

    y = np.arange(len(labels))
    for yi in [0, 2, 4]:
        ax.axhspan(yi - 0.35, yi + 0.35, color="#F5F5F5", zorder=0)

    bars = ax.barh(y, values - x_left, left=x_left, height=0.58, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(x_left, 1.14)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC", fontsize=9)
    polish_axes(ax, grid_axis="x")
    ax.axvline(0.5, color="#888", linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)

    for bar, val in zip(bars, values):
        txt = f"{val:.3f}"
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, txt, va="center", ha="left", fontsize=7.5, color="#1F1F1F")

    ax.legend(handles=[
        Patch(facecolor=HIGHLIGHT_COLOR, label="Full (best)"),
        Patch(facecolor=STEELBLUE, label="Ablated"),
        Patch(facecolor=DARKORANGE, label="Temporal only"),
        Patch(facecolor=RED, label="Prior only"),
    ], frameon=False, loc="lower right", fontsize=7, handlelength=1.0, labelspacing=0.3)


def main():
    setup_style()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 9.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.45, left=0.12, right=0.94, top=0.97, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_auroc(ax_a)
    _draw_enrichment(ax_b)
    _draw_recovery(ax_c)
    _draw_ablation(ax_d)

    _label(ax_a, "a", x=-0.16)
    _label(ax_b, "b")
    _label(ax_c, "c", x=-0.16)
    _label(ax_d, "d")

    save_pdf(fig, "fig2_composite")


if __name__ == "__main__":
    main()
