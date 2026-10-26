from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

STEELBLUE    = "#56B4E9"
DARKORANGE   = "#E69F00"
GREEN        = "#009E73"
RED          = "#D55E00"
PURPLE       = "#CC79A7"
GRAY         = "#888888"
DARKGREEN    = "#009E73"
CORAL        = "#D55E00"
LIGHTGRID    = "#E0E0E0"

MODEL_COLOR    = "#0072B2"
FOLD_C_COLOR   = "#0072B2"
FOLD_D_COLOR   = "#56B4E9"
PRIOR_COLOR    = "#E69F00"
BASELINE_COLOR = "#888888"
HIGHLIGHT_COLOR = "#009E73"

SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.2

FIGURES_DIR  = Path(__file__).resolve().parent / "output"
FIGURES_DIR.mkdir(exist_ok=True)
RANDOM_SEED  = 42

ANNOT_BOX = dict(
    boxstyle="round,pad=0.3",
    facecolor="white",
    edgecolor="#C0C0C0",
    linewidth=0.7,
    alpha=0.95,
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": LIGHTGRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=OKABE_ITO)


def add_panel_label(
    ax: Axes,
    label: str,
    x: float = -0.13,
    y: float = 1.02,
) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def polish_axes(ax: Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=LIGHTGRID, linewidth=0.5, alpha=1.0)
    ax.set_axisbelow(True)


def save_pdf(fig: Figure, stem: str) -> None:
    stem_name = Path(stem).stem
    pdf_path = FIGURES_DIR / f"{stem_name}.pdf"
    png_path = FIGURES_DIR / f"{stem_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path.name}  {png_path.name}")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
