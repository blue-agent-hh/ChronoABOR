#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import ANNOT_BOX, DOUBLE_COLUMN_WIDTH, setup_style

BLUE = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
LIGHT_GRAY = "#DCDCDC"
MID_GRAY = "#CCCCCC"
GRID = "#E0E0E0"
TEXT = "#3A3A3A"

ROOT_DIR = Path(__file__).resolve().parents[1] / "analysis"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "output"
FIGURES_DIR.mkdir(exist_ok=True)

MATCHED_JSON = RESULTS_DIR / "nm_exp_p7b_matched_enrichment.json"
DEFAULT_RANDOM_JSON = RESULTS_DIR / "nm_exp_p7_ranked_enrichment.json"
OUTPUT_PDF = FIGURES_DIR / "fig_matched_enrichment.pdf"
OUTPUT_PNG = FIGURES_DIR / "fig_matched_enrichment.png"
LEGACY_RESULTS_PNG = RESULTS_DIR / "nm_exp_p7b_matched_enrichment.png"


def load_results() -> tuple[dict, dict]:
    matched = json.loads(MATCHED_JSON.read_text())
    referenced_path = Path(matched["score_source"]["existing_p7_json"])
    random_json_path = referenced_path if referenced_path.exists() else DEFAULT_RANDOM_JSON
    random = json.loads(random_json_path.read_text())
    return matched, random


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#4C4C4C")
    ax.spines["bottom"].set_color("#4C4C4C")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, color="#4C4C4C")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.025) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def split_leading_edge_lines(genes: list[str]) -> tuple[str, str]:
    if not genes:
        return "Leading edge:", ""
    split_idx = max(1, len(genes) // 2)
    line_1 = f"Leading edge: {', '.join(genes[:split_idx])},"
    line_2 = ", ".join(genes[split_idx:])
    return line_1, line_2


def draw_summary_panel(ax: plt.Axes, matched: dict, random: dict) -> None:
    from matplotlib import transforms
    chrono = random["methods"]["chronoabor"]
    sig = random["methods"]["sig_max_auc"]
    summary_rows = [
        ("ChronoABOR\nrandom null", chrono["nes"], chrono["p_value"], BLUE),
        ("sig_max_auc\nrandom null", sig["nes"], sig["p_value"], MID_GRAY),
    ]

    y = np.arange(len(summary_rows))
    values = [row[1] for row in summary_rows]
    bars = ax.barh(
        y,
        values,
        height=0.58,
        color=[row[3] for row in summary_rows],
        edgecolor="#4C4C4C",
        linewidth=0.7,
        zorder=3,
    )

    ax.axvline(1.0, color="#707070", linestyle="--", linewidth=1.1, zorder=1)

    for bar, (_, nes, p_value, _) in zip(bars, summary_rows):
        ax.text(
            nes + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"NES={nes:.3f}\np={p_value:.3f}",
            ha="left",
            va="center",
            fontsize=9.2,
            color=TEXT,
            clip_on=False,
        )

    ax.set_xlim(0.8, 1.9)
    ax.set_xticks(np.arange(0.8, 1.81, 0.2))
    ax.set_ylim(-0.55, 1.8)
    ax.set_xlabel("Normalized enrichment score")
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in summary_rows])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.7)

    style_axis(ax)


def draw_hit_pattern_panel(ax: plt.Axes, matched: dict, random: dict) -> None:
    from matplotlib.patches import Patch

    chrono = random["methods"]["chronoabor"]
    top_genes = chrono["top_20_ranked_genes"]
    top_hits = set(chrono["top_20_hit_genes"])
    n_hits = len(top_hits)
    n_total = len(top_genes)

    y = np.arange(n_total - 1, -1, -1, dtype=float)
    colors = [GREEN if g in top_hits else LIGHT_GRAY for g in top_genes]

    ax.barh(y, [1.0] * n_total, height=0.72, color=colors,
            edgecolor="white", linewidth=0.5, zorder=3)

    for i, yi in enumerate(y):
        rank_num = i + 1
        bar_color = "white" if top_genes[i] in top_hits else "#777777"
        ax.text(0.5, yi, str(rank_num), va="center", ha="center",
                fontsize=8.5, color=bar_color, fontweight="bold", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(top_genes, fontsize=9.0)
    for tick_label, gene in zip(ax.get_yticklabels(), top_genes):
        if gene in top_hits:
            tick_label.set_color("#1B6B3A")
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color("#555555")

    ax.set_xticks([])
    ax.set_xlim(0, 1.35)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)


    legend_handles = [
        Patch(facecolor=GREEN, label=f"Confirmed hit ({n_hits})"),
        Patch(facecolor=LIGHT_GRAY, label=f"Not confirmed ({n_total - n_hits})"),
    ]
    return legend_handles


def draw_histogram_panel(ax: plt.Axes, matched: dict) -> None:
    from matplotlib.lines import Line2D

    null_es = np.asarray(matched["matched_null"]["permutation_es"], dtype=float)
    observed_es = matched["observed_enrichment"]["es"]
    null_summary = matched["matched_null"]["null_distribution"]
    ci95 = null_summary["ci95"]
    nes = matched["matched_null"]["matched_nes"]
    p_val = matched["matched_null"]["matched_p_value"]

    x_min, x_max = 0.94, 1.00
    bins = np.linspace(x_min, x_max, 25)
    ax.hist(null_es, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2)
    vline_null = ax.axvline(null_summary["mean"], color="#1A6FA8", linestyle="--", linewidth=2.0, zorder=4, label="Null mean")
    vline_obs  = ax.axvline(observed_es, color=ORANGE, linewidth=2.0, zorder=5, label=f"Observed ES = {observed_es:.3f}")

    ax.legend(
        handles=[vline_null, vline_obs],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=9.5,
        handlelength=2.5,
        ncol=1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0.94, 0.96, 0.98, 1.00])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.set_xlabel("Matched-null enrichment score")
    ax.set_ylabel("Permutation count")
    ax.grid(axis="y", color=GRID, linewidth=0.7)

    style_axis(ax)


def build_figure(matched: dict, random: dict) -> plt.Figure:
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_summary_panel(ax_a, matched, random)
    legend_handles = draw_hit_pattern_panel(ax_b, matched, random)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b", x=-0.04, y=1.025)

    ax_b.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.2, -0.15),
                ncol=1, frameon=False, fontsize=9.5, handlelength=1.0, handleheight=0.9)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.26, wspace=0.55)

    return fig


def main() -> None:
    setup_style()
    matched, random = load_results()
    fig = build_figure(matched, random)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(LEGACY_RESULTS_PNG, dpi=300)
    plt.close(fig)

    print(f"Loaded {MATCHED_JSON}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {LEGACY_RESULTS_PNG}")


if __name__ == "__main__":
    main()
