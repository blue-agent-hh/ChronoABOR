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
    ANNOT_BOX, DOUBLE_COLUMN_WIDTH,
    CORAL, STEELBLUE, HIGHLIGHT_COLOR, LIGHTGRID,
    setup_style, polish_axes, save_pdf,
)

CHRONOABOR_COLOR = "#009E73"

_E12_WS = json.load(open(Path(__file__).resolve().parents[1] / "analysis" / "results" / "rev_exp_e12_strict_windows.json"))["benchmarks"]["within_secreted_strict"]
SEED_AUROCS = {
    "logistic": [r["auroc_oof"] for r in _E12_WS["logistic"]["repeats"]],
    "xgb": [r["auroc_oof"] for r in _E12_WS["xgb"]["repeats"]],
}


def _seed_dots(ax, x, values):
    jitter = np.array([-0.12, 0.0, 0.12])
    ax.scatter(x + jitter, values, s=10, facecolor="white", edgecolor="#1F1F1F", linewidth=0.6, zorder=5)
GRID = "#E0E0E0"
TEXT = "#3A3A3A"


def _label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)


def _draw_shap(ax):
    labels = ["deg_lfc_mean", "deg_lfc_max", "n_evidence_total", "is_secreted", "num_modalities", "is_ad_driver"]
    values = np.array([2.106, 1.505, 0.486, 0.311, 0.206, 0.071])
    colors = [STEELBLUE, STEELBLUE, STEELBLUE, CORAL, STEELBLUE, CORAL]
    subtitles = ["Mean DEG effect size", "Max DEG effect size", "Total evidence count", "Peripheral\naccessibility", "Evidence modalities", "AD pathway membership"]
    total_all_features = 4.878

    pct = values / total_all_features * 100
    y = np.arange(len(labels))

    for i, (val, col) in enumerate(zip(values, colors)):
        ax.plot([0, val], [y[i], y[i]], color="#CCC", linewidth=1.3, zorder=1, solid_capstyle="round")
    for i, (val, col) in enumerate(zip(values, colors)):
        ax.scatter(val, y[i], s=90, color=col, edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(values[0], y[0], s=200, facecolor="none", edgecolor=STEELBLUE, linewidth=1.3, zorder=2)

    for xref in [1.0, 2.0]:
        ax.axvline(xref, color="#E0E0E0", linewidth=0.6, linestyle="-", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 1 + 0.6, -0.45)
    ax.set_xlim(0.0, 3.3)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    polish_axes(ax, grid_axis="x")

    for i, (val, pv, sub) in enumerate(zip(values, pct, subtitles)):
        xo = 0.22 if i == 0 else 0.15
        ax.text(val + xo, y[i], f"{val:.3f} ({pv:.0f}%)", va="center", ha="left", fontsize=8, color="#1F1F1F")
        sub_y = y[i] + 0.28 if i == 3 else y[i] + 0.25
        ax.text(val + xo, sub_y, sub, va="top", ha="left", fontsize=7, color="#888", fontstyle="italic")

    ax.legend(handles=[
        Patch(facecolor=CORAL, label="Binary prior"),
        Patch(facecolor=STEELBLUE, label="Continuous"),
    ], frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7.5, handlelength=1.0)


_GRAY_OTHERS = ["#AAAAAA", "#56B4E9", "#E69F00", "#D55E00", "#0072B2", "#CC79A7"]


def _draw_secreted(ax):
    rows = [
        {"label": "Secreted",            "auroc": 0.500, "lo": 0.500, "hi": 0.500, "kind": "direct"},
        {"label": "Max signature\nAUC",  "auroc": 0.952, "lo": 0.952, "hi": 0.952, "kind": "direct"},
        {"label": "Publication\ncount",  "auroc": 0.989, "lo": 0.989, "hi": 0.989, "kind": "direct"},
        {"label": "GPT-5.4",             "auroc": 0.982, "lo": 0.982, "hi": 0.982, "kind": "direct"},
        {"label": "ChronoABOR\n(XGBoost)",  "auroc": 0.983, "sd": 0.002, "kind": "model"},
        {"label": "ChronoABOR\n(logistic)", "auroc": 0.985, "sd": 0.005, "kind": "full"},
    ]

    labels = [r["label"] for r in rows]
    aucs = [r["auroc"] for r in rows]
    lo_err = [r["auroc"] - r["lo"] if r["kind"] == "direct" else r["sd"] for r in rows]
    hi_err = [r["hi"] - r["auroc"] if r["kind"] == "direct" else r["sd"] for r in rows]

    gray_iter = iter(_GRAY_OTHERS)
    colors = [CHRONOABOR_COLOR if r["kind"] == "full" else next(gray_iter) for r in rows]
    chrono_auc = rows[-1]["auroc"]
    x = np.arange(len(labels))

    bars = ax.bar(x, aucs, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, r in zip(x, rows):
        if r["kind"] != "direct":
            ax.errorbar(xi, r["auroc"], yerr=r["sd"], fmt="none", ecolor="#444", elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)
            _seed_dots(ax, xi, SEED_AUROCS["xgb" if "XGBoost" in r["label"] else "logistic"])
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(chrono_auc, color=CHRONOABOR_COLOR, linestyle=":", linewidth=0.9, alpha=0.6, zorder=2)
    ax.set_ylabel("Within-secreted AUROC", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.12)
    polish_axes(ax, grid_axis="y")

    for bar, row, he in zip(bars, rows, hi_err):
        label_y = min(row["auroc"] + he + 0.03, 1.05)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{row['auroc']:.3f}", ha="center", va="bottom", fontsize=7.5, color="#222")


LLM_COLORS = {
    "GPT-5.2": "#D55E00", "GPT-5.4": "#56B4E9",
    "Gemini-2.5-Flash": "#009E73", "Gemini-2.5-Pro": "#CC79A7",
    "ChronoABOR\n(logistic)": "#E69F00", "ChronoABOR\n(XGBoost)": "#B8860B",
}

LLM_ROWS = [
    {"display_name": "GPT-5.2",          "auroc": 0.972, "auprc": 0.305, "p_at_10": 0.3, "model_family": "llm"},
    {"display_name": "GPT-5.4",          "auroc": 0.982, "auprc": 0.499, "p_at_10": 0.3, "model_family": "llm"},
    {"display_name": "Gemini-2.5-Flash", "auroc": 0.960, "auprc": 0.253, "p_at_10": 0.2, "model_family": "llm"},
    {"display_name": "Gemini-2.5-Pro",   "auroc": 0.966, "auprc": 0.276, "p_at_10": 0.2, "model_family": "llm"},
    {"display_name": "ChronoABOR\n(logistic)", "auroc": 0.985, "auroc_sd": 0.005, "auprc": 0.463, "p_at_10": 0.4, "model_family": "chronoabor"},
    {"display_name": "ChronoABOR\n(XGBoost)",  "auroc": 0.983, "auroc_sd": 0.002, "auprc": 0.510, "p_at_10": 0.1, "show_p10": False, "model_family": "chronoabor"},
]


def _draw_llm_panel(ax, rows, metric, ylabel):
    labels = [r["display_name"] for r in rows]
    x = np.arange(len(rows))
    colors = [LLM_COLORS.get(lbl, "#AAA") for lbl in labels]
    values = [r[metric] for r in rows]

    if metric == "auroc":
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        for xi, r in zip(x, rows):
            if "auroc_sd" in r:
                ax.errorbar(xi, r["auroc"], yerr=r["auroc_sd"], fmt="none", ecolor="#555", elinewidth=0.7, capsize=3, zorder=4)
                _seed_dots(ax, xi, SEED_AUROCS["xgb" if "XGBoost" in r["display_name"] else "logistic"])
        ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_ylim(0.0, 1.12)
    else:
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(4 / 246, color="#888", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_ylim(0.0, 1.0)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    polish_axes(ax, grid_axis="y")

    for bar, val, row in zip(bars, values, rows):
        if metric == "auroc":
            label_y = min(val + row.get("auroc_sd", 0.0) + 0.03, 1.05)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)


def main():
    setup_style()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 8.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.42, left=0.12, right=0.96, top=0.97, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_shap(ax_a)
    _draw_secreted(ax_b)
    _draw_llm_panel(ax_c, LLM_ROWS, "auroc", "Within-secreted AUROC")
    _draw_llm_panel(ax_d, LLM_ROWS, "auprc", "Within-secreted AUPRC")

    _label(ax_a, "a", x=-0.16)
    _label(ax_b, "b")
    _label(ax_c, "c", x=-0.16)
    _label(ax_d, "d")

    save_pdf(fig, "fig3_composite")


if __name__ == "__main__":
    main()
