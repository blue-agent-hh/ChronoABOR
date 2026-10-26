#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import textwrap
from collections import OrderedDict
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_JSON = RESULTS_DIR / "nm_exp4_full_universe.json"
OUTPUT_JSON = RESULTS_DIR / "nm_exp_p5_overlap.json"
OUTPUT_PNG = RESULTS_DIR / "nm_exp_p5_overlap.png"

VALID_GENE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
GENE_SPLIT_RE = re.compile(r"[,;/]")
ABOR_INVALID_GENE_TOKENS = {
    "",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
    "AMYLOID",
    "CSF",
    "PET",
    "MRI",
    "APOE4",
    "P-TAU181",
    "P-TAU217",
    "ABETA",
    "A-BETA",
}
EXTERNAL_INVALID_GENE_TOKENS = ABOR_INVALID_GENE_TOKENS | {"GENETIC"}
ALIAS_MAP = {
    "ACES": "ACHE",
    "ACHES": "ACHE",
    "TAGL": "TAGLN",
    "NFL": "NEFL",
}

LAYER_FILES = {
    "snp": ("snp.csv", "Gene_symbol"),
    "cpg": ("cpg.csv", "Gene_symbol"),
    "deg": ("degs.csv", "Gene_symbol"),
    "hub": ("hubgene.csv", "Gene_symbl"),
    "mirna": ("miRNA.csv", "Gene_symbol"),
    "signature": ("signature.csv", "Gene_symbol"),
}

IADRC_UP_RAW = [
    "ACES",
    "NPTXR",
    "GFAP",
    "CPLX2",
    "TAGL",
    "FBLN3",
    "ROR1",
    "FZD7",
    "UNC5D",
    "PGD2",
]
IADRC_DOWN_RAW = [
    "SMOC1",
    "CPLX1",
    "IGFBP2",
    "SLPI",
    "DSC2",
    "BGH3",
    "SPF30",
    "SH3K1",
    "T132D",
    "XLRS1",
]
KNIGHT_UP_RAW = [
    "ACHE",
    "SPC25",
    "CTF1",
    "LRRN1",
    "NEFL",
    "CPLX2",
    "SMOC1",
    "TBCA",
    "NPTXR",
    "SPP1",
    "SPARC",
    "CLU",
    "APOE",
    "BIN1",
    "APOB",
    "CRP",
    "PPBP",
    "C5",
    "CFB",
    "VEGFA",
    "COLEC12",
    "NCAM1",
    "NRCAM",
    "ROBO2",
    "CPLX1",
    "SYT1",
    "FOXO1",
    "YWHAQ",
    "SLITRK3",
    "SMOC2",
    "FAHD2A",
    "AKT1",
    "AKT3",
    "BMP6",
    "RXRA",
    "MAPK10",
    "GSK3A",
    "PPP1CC",
    "YWHAG",
    "TMOD2",
    "YWHAB",
    "PPP3R1",
    "TMOD3",
    "DLG2",
    "YWHAZ",
    "YWHAE",
    "MAPRE3",
    "ATP6V1F",
    "HOMER1",
    "RAB1B",
    "ATP1B2",
    "STX1A",
    "CDC42",
    "RAC1",
    "RAC2",
    "PF4",
    "COL10A1",
    "SCP2",
    "SMPD1",
    "PLA2G12A",
    "CTSF",
    "EIF4E",
    "NRCAM",
    "GPD1",
    "NRP2",
    "SHMT2",
    "ARHGAP30",
]


def ordered_unique(values: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(values))


def normalize_external_gene_symbol(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    gene = str(raw_value).strip().upper()
    if gene in EXTERNAL_INVALID_GENE_TOKENS:
        return None
    gene = ALIAS_MAP.get(gene, gene)
    if not VALID_GENE_RE.fullmatch(gene):
        return None
    return gene


def normalize_abor_gene_tokens(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    genes: list[str] = []
    for part in GENE_SPLIT_RE.split(str(raw_value).upper()):
        gene = part.strip()
        if gene in ABOR_INVALID_GENE_TOKENS:
            continue
        if not VALID_GENE_RE.fullmatch(gene):
            continue
        genes.append(gene)
    return sorted(set(genes))


def canonicalize_gene_list(raw_genes: list[str]) -> list[str]:
    canonical: list[str] = []
    for raw_gene in raw_genes:
        gene = normalize_external_gene_symbol(raw_gene)
        if gene is not None:
            canonical.append(gene)
    return ordered_unique(canonical)


def load_gene_universe() -> list[str]:
    universe: set[str] = set()
    for filename, gene_column in LAYER_FILES.values():
        df = pd.read_csv(BASE_DIR / filename)
        for raw_value in df[gene_column]:
            universe.update(normalize_abor_gene_tokens(raw_value))
    return sorted(universe)


def build_external_set_summary(
    raw_genes: list[str],
    universe_set: set[str],
) -> dict:
    canonical_genes = canonicalize_gene_list(raw_genes)
    in_universe = [gene for gene in canonical_genes if gene in universe_set]
    not_in_universe = [gene for gene in canonical_genes if gene not in universe_set]
    return {
        "raw_genes": raw_genes,
        "canonical_genes": canonical_genes,
        "genes_in_universe": in_universe,
        "genes_not_in_universe": not_in_universe,
        "n_raw": len(raw_genes),
        "n_canonical_unique": len(canonical_genes),
        "n_in_universe": len(in_universe),
        "n_not_in_universe": len(not_in_universe),
    }


def rank_order_overlap(reference_ranked_genes: list[str], comparison_set: set[str]) -> list[str]:
    return [gene for gene in reference_ranked_genes if gene in comparison_set]


def compute_overlap(
    label: str,
    chrono_ranked_genes: list[str],
    external_genes_in_universe: list[str],
    background_n: int,
) -> dict:
    top_k_set = set(chrono_ranked_genes)
    external_set = set(external_genes_in_universe)
    overlap_genes = rank_order_overlap(chrono_ranked_genes, external_set)
    observed = len(overlap_genes)
    effective_k = len(chrono_ranked_genes)
    external_n = len(external_set)
    expected = effective_k * external_n / background_n

    a = observed
    b = effective_k - observed
    c = external_n - observed
    d = background_n - a - b - c
    if d < 0:
        raise ValueError(f"Invalid contingency table for {label}: {[a, b, c, d]}")

    odds_ratio, fisher_p_greater = fisher_exact([[a, b], [c, d]], alternative="greater")
    _, fisher_p_two_sided = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    return {
        "comparison": label,
        "top_k_requested": 10 if "top10" in label else 50,
        "top_k_effective": effective_k,
        "external_set_size_in_universe": external_n,
        "overlap_count": observed,
        "overlap_genes": overlap_genes,
        "expected_overlap_by_chance": expected,
        "contingency_table": [[a, b], [c, d]],
        "odds_ratio": odds_ratio,
        "fisher_alternative": "greater",
        "fisher_p_value": fisher_p_greater,
        "fisher_p_value_two_sided": fisher_p_two_sided,
    }


def format_gene_list(genes: list[str], width: int = 36) -> str:
    if not genes:
        return "None"
    return textwrap.fill(", ".join(genes), width=width)


def format_p_value(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def create_figure(result: dict) -> None:
    comparisons = result["comparisons"]
    up10 = comparisons["top10_vs_external_upregulated"]
    up50 = comparisons["top50_vs_external_upregulated"]
    labels = [f"Top10\n(k={up10['top_k_effective']})", f"Top50\n(k={up50['top_k_effective']})"]
    observed = [up10["overlap_count"], up50["overlap_count"]]
    expected = [up10["expected_overlap_by_chance"], up50["expected_overlap_by_chance"]]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.15, 1.0])

    ax_bar = fig.add_subplot(gs[0, :])
    x = range(len(labels))
    width = 0.32
    ax_bar.bar(
        [idx - width / 2 for idx in x],
        observed,
        width=width,
        color="#2f6c8f",
        label="Observed overlap",
    )
    ax_bar.bar(
        [idx + width / 2 for idx in x],
        expected,
        width=width,
        color="#d9a441",
        label="Expected by chance",
    )
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylabel("Overlapping genes", fontsize=12)
    ax_bar.set_title(
        "ChronoABOR overlap with external AD plasma proteomics (primary analysis: AD-upregulated proteins)",
        fontsize=14,
        pad=12,
    )
    ymax = max(max(observed), max(expected)) + 1.4
    ax_bar.set_ylim(0, ymax)
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")
    ax_bar.legend(frameon=False, loc="upper left")

    for idx, comparison in enumerate([up10, up50]):
        ax_bar.text(
            idx - width / 2,
            observed[idx] + 0.12,
            f"p={format_p_value(comparison['fisher_p_value'])}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_genes = fig.add_subplot(gs[1, 0])
    ax_genes.axis("off")
    overlap_rows = [
        ["Top10 vs Up", format_gene_list(comparisons["top10_vs_external_upregulated"]["overlap_genes"])],
        ["Top50 vs Up", format_gene_list(comparisons["top50_vs_external_upregulated"]["overlap_genes"])],
        ["Top10 vs All", format_gene_list(comparisons["top10_vs_all_external"]["overlap_genes"])],
        ["Top50 vs All", format_gene_list(comparisons["top50_vs_all_external"]["overlap_genes"])],
    ]
    gene_table = ax_genes.table(
        cellText=overlap_rows,
        colLabels=["Comparison", "Overlap genes"],
        colWidths=[0.28, 0.72],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    gene_table.auto_set_font_size(False)
    gene_table.set_fontsize(10)
    gene_table.scale(1, 2.2)
    ax_genes.set_title("Overlapping genes", fontsize=12, pad=10)

    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis("off")
    stats_rows = []
    for key in [
        "top10_vs_external_upregulated",
        "top50_vs_external_upregulated",
        "top10_vs_all_external",
        "top50_vs_all_external",
    ]:
        comparison = comparisons[key]
        stats_rows.append(
            [
                comparison["comparison"].replace("_vs_", " vs ").replace("_", " "),
                str(comparison["overlap_count"]),
                f"{comparison['expected_overlap_by_chance']:.2f}",
                format_p_value(comparison["fisher_p_value"]),
            ]
        )
    stats_table = ax_stats.table(
        cellText=stats_rows,
        colLabels=["Comparison", "Obs", "Exp", "Fisher p"],
        colWidths=[0.55, 0.12, 0.14, 0.19],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(10)
    stats_table.scale(1, 2.0)
    ax_stats.set_title("Overlap statistics", fontsize=12, pad=10)

    fig.text(
        0.5,
        0.02,
        (
            "Sources: IADRC (PMC12351396, Alzheimer's & Dementia 2025) and Knight ADRC "
            "(PMC12674600, Nature Aging 2025). Alias normalization: ACES/ACHES->ACHE, "
            "TAGL->TAGLN, NFL->NEFL. 'GENETIC' was excluded from ChronoABOR top50, so effective top50 k=49."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nm_exp4 = json.loads(INPUT_JSON.read_text())
    background_n = int(nm_exp4["n_genes_universe"])
    chrono_top50_raw = nm_exp4["top50_genes"]
    chrono_top50 = [gene for gene in canonicalize_gene_list(chrono_top50_raw) if gene != "GENETIC"]
    chrono_top10 = chrono_top50[:10]

    universe = load_gene_universe()
    universe_set = set(universe)

    external_up_raw = ordered_unique(IADRC_UP_RAW + KNIGHT_UP_RAW)
    external_all_raw = ordered_unique(IADRC_UP_RAW + IADRC_DOWN_RAW + KNIGHT_UP_RAW)
    external_up = build_external_set_summary(external_up_raw, universe_set)
    external_all = build_external_set_summary(external_all_raw, universe_set)

    comparisons = {
        "top10_vs_external_upregulated": compute_overlap(
            label="top10_vs_external_upregulated",
            chrono_ranked_genes=chrono_top10,
            external_genes_in_universe=external_up["genes_in_universe"],
            background_n=background_n,
        ),
        "top50_vs_external_upregulated": compute_overlap(
            label="top50_vs_external_upregulated",
            chrono_ranked_genes=chrono_top50,
            external_genes_in_universe=external_up["genes_in_universe"],
            background_n=background_n,
        ),
        "top10_vs_all_external": compute_overlap(
            label="top10_vs_all_external",
            chrono_ranked_genes=chrono_top10,
            external_genes_in_universe=external_all["genes_in_universe"],
            background_n=background_n,
        ),
        "top50_vs_all_external": compute_overlap(
            label="top50_vs_all_external",
            chrono_ranked_genes=chrono_top50,
            external_genes_in_universe=external_all["genes_in_universe"],
            background_n=background_n,
        ),
    }

    result = {
        "input_json": str(INPUT_JSON.relative_to(BASE_DIR)),
        "background_universe": {
            "n_genes_universe_from_nm_exp4": background_n,
            "n_genes_universe_rebuilt_from_abor_layers": len(universe),
            "rebuilt_matches_nm_exp4": len(universe) == background_n,
        },
        "chronoabor": {
            "top50_genes_raw": chrono_top50_raw,
            "excluded_non_gene_tokens": sorted(set(chrono_top50_raw) - set(chrono_top50)),
            "top50_genes_filtered": chrono_top50,
            "top10_genes_filtered": chrono_top10,
        },
        "external_validation": {
            "alias_map": ALIAS_MAP,
            "source_a_iadrc": {
                "citation": "IADRC plasma proteomics, PMC12351396, Alzheimer's & Dementia 2025",
                "upregulated_raw": IADRC_UP_RAW,
                "downregulated_raw": IADRC_DOWN_RAW,
                "upregulated_canonical": canonicalize_gene_list(IADRC_UP_RAW),
                "downregulated_canonical": canonicalize_gene_list(IADRC_DOWN_RAW),
            },
            "source_b_knight": {
                "citation": "Knight ADRC plasma proteomics, PMC12674600, Nature Aging 2025",
                "upregulated_raw": KNIGHT_UP_RAW,
                "upregulated_canonical": canonicalize_gene_list(KNIGHT_UP_RAW),
            },
            "external_upregulated": external_up,
            "external_all_differential": external_all,
        },
        "comparisons": comparisons,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2))
    create_figure(result)

    print("ChronoABOR external validation overlap analysis")
    print(f"Background universe: {background_n} genes")
    print(
        "Rebuilt universe from ABOR layers: "
        f"{len(universe)} genes (match={len(universe) == background_n})"
    )
    print(
        "External upregulated set: "
        f"{external_up['n_canonical_unique']} canonical genes, "
        f"{external_up['n_in_universe']} in ABOR universe"
    )
    print(
        "External all-differential set: "
        f"{external_all['n_canonical_unique']} canonical genes, "
        f"{external_all['n_in_universe']} in ABOR universe"
    )
    print()
    for key in [
        "top10_vs_external_upregulated",
        "top50_vs_external_upregulated",
        "top10_vs_all_external",
        "top50_vs_all_external",
    ]:
        comparison = comparisons[key]
        print(comparison["comparison"])
        print(f"  effective_k: {comparison['top_k_effective']}")
        print(f"  overlap_count: {comparison['overlap_count']}")
        print(f"  overlap_genes: {', '.join(comparison['overlap_genes']) or 'None'}")
        print(f"  expected_by_chance: {comparison['expected_overlap_by_chance']:.3f}")
        print(f"  fisher_exact_greater_p: {comparison['fisher_p_value']:.6g}")
        print()
    print(f"Wrote {OUTPUT_JSON.relative_to(BASE_DIR)}")
    print(f"Wrote {OUTPUT_PNG.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
