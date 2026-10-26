#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from nm_exp4_full_universe import build_gene_universe, normalize_gene_tokens
from nm_exp_p5_external_validation import normalize_external_gene_symbol
from rev_exp_common import BASE_DIR, RANDOM_SEED, ensure_results_dir, load_all_tables, provenance, write_json

SCRIPT_NAME = Path(__file__).name
ANNOTATIONS_DIR = BASE_DIR / "external_data" / "annotations"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "rev_exp_annotations.json"

UNIPROT_ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_QUERY = "(organism_id:9606) AND (reviewed:true)"
UNIPROT_FIELDS = [
    "accession",
    "gene_primary",
    "gene_names",
    "keyword",
    "keywordid",
    "go_id",
    "cc_subcellular_location",
]
UNIPROT_PAGE_SIZE = 500
UNIPROT_RAW_PATH = ANNOTATIONS_DIR / "uniprot_reviewed_human_raw.tsv"
UNIPROT_MANIFEST_PATH = ANNOTATIONS_DIR / "uniprot_reviewed_human_manifest.json"
UNIPROT_SECRETED_PATH = ANNOTATIONS_DIR / "uniprot_secreted_human.tsv"

KEGG_LINK_URL = "https://rest.kegg.jp/link/hsa/hsa05010"
KEGG_LIST_URL = "https://rest.kegg.jp/list/hsa"
KEGG_LINK_PATH = ANNOTATIONS_DIR / "kegg_hsa05010_link.tsv"
KEGG_LIST_PATH = ANNOTATIONS_DIR / "kegg_hsa_list.tsv"
KEGG_DERIVED_PATH = ANNOTATIONS_DIR / "kegg_hsa05010_genes.tsv"

DISGENET_STATUS_PATH = ANNOTATIONS_DIR / "disgenet_status.json"

UNIPROT_SECRETED_KEYWORD_ID = "KW-0964"
GO_EXTRACELLULAR_REGION = "GO:0005576"
GO_EXTRACELLULAR_SPACE = "GO:0005615"
GO_EXTRACELLULAR_TERMS = {GO_EXTRACELLULAR_REGION, GO_EXTRACELLULAR_SPACE}

REQUEST_TIMEOUT = 180
NEXT_LINK_RE = re.compile(r"<([^>]+)>;\s*rel=\"next\"")
UNIPROT_GENE_SPLIT_RE = re.compile(r"[\s,;/]+")


def ensure_annotations_dir() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_cached_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _extract_next_url(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = NEXT_LINK_RE.search(link_header)
    return match.group(1) if match else None


def _clean_token(token: str) -> str | None:
    normalized = normalize_external_gene_symbol(token)
    if normalized is not None:
        return normalized
    fallback = token.strip().upper()
    if not fallback:
        return None
    parsed = normalize_gene_tokens(fallback.replace("-", " "))
    if len(parsed) == 1:
        return parsed[0]
    return None


def parse_uniprot_gene_tokens(primary: object, names: object) -> list[str]:
    tokens: set[str] = set()
    for raw_value in (primary, names):
        if pd.isna(raw_value):
            continue
        for part in UNIPROT_GENE_SPLIT_RE.split(str(raw_value).strip().upper()):
            if not part:
                continue
            normalized = _clean_token(part)
            if normalized is not None:
                tokens.add(normalized)
    return sorted(tokens)


def fetch_uniprot_reviewed_human(session: requests.Session) -> tuple[pd.DataFrame, dict]:
    cached_manifest = _load_cached_manifest(UNIPROT_MANIFEST_PATH)
    if cached_manifest is not None and UNIPROT_RAW_PATH.exists() and UNIPROT_SECRETED_PATH.exists():
        raw_df = pd.read_csv(UNIPROT_RAW_PATH, sep="\t")
        return raw_df, cached_manifest

    params = {
        "format": "tsv",
        "query": UNIPROT_QUERY,
        "fields": ",".join(UNIPROT_FIELDS),
        "size": str(UNIPROT_PAGE_SIZE),
    }
    next_url: str | None = UNIPROT_ENDPOINT
    next_params: dict[str, str] | None = params
    lines: list[str] = []
    page_count = 0
    total_rows = 0
    release = None
    release_date = None
    total_results = None
    started_at = time.time()

    while next_url is not None:
        response = session.get(next_url, params=next_params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        page_lines = response.text.splitlines()
        if not page_lines:
            raise RuntimeError("UniProt returned an empty page.")
        if page_count == 0:
            lines.extend(page_lines)
            release = response.headers.get("x-uniprot-release")
            release_date = response.headers.get("x-uniprot-release-date")
            total_results = int(response.headers["x-total-results"])
        else:
            lines.extend(page_lines[1:])
        total_rows += max(len(page_lines) - 1, 0)
        page_count += 1
        next_url = _extract_next_url(response.headers.get("link"))
        next_params = None

    if release is None or release_date is None or total_results is None:
        raise RuntimeError("UniProt release metadata headers were not returned.")

    UNIPROT_RAW_PATH.write_text("\n".join(lines) + "\n")
    raw_df = pd.read_csv(UNIPROT_RAW_PATH, sep="\t")
    if len(raw_df) != total_results:
        raise RuntimeError(
            f"UniProt row count mismatch: header announced {total_results} rows but fetched {len(raw_df)} rows."
        )

    manifest = {
        "source": "UniProt REST search API",
        "endpoint": UNIPROT_ENDPOINT,
        "query": UNIPROT_QUERY,
        "fields": UNIPROT_FIELDS,
        "page_size": UNIPROT_PAGE_SIZE,
        "release": release,
        "release_date": release_date,
        "x_total_results": total_results,
        "pages_fetched": page_count,
        "rows_fetched": total_rows,
        "accessed_at_utc": provenance(SCRIPT_NAME, {}, []).get("timestamp_utc"),
        "runtime_seconds": float(time.time() - started_at),
        "raw_tsv_path": str(UNIPROT_RAW_PATH),
    }
    UNIPROT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return raw_df, manifest


def derive_uniprot_secreted_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    derived = raw_df.copy()
    derived["gene_tokens"] = [
        ";".join(parse_uniprot_gene_tokens(primary, names))
        for primary, names in zip(derived["Gene Names (primary)"], derived["Gene Names"], strict=True)
    ]
    derived["keyword_id_text"] = derived["Keyword ID"].fillna("").astype(str)
    derived["go_id_text"] = derived["Gene Ontology IDs"].fillna("").astype(str)
    derived["is_secreted_kw_0964"] = derived["keyword_id_text"].apply(
        lambda text: int(UNIPROT_SECRETED_KEYWORD_ID in {part.strip() for part in text.split(";") if part.strip()})
    )
    derived["has_go_0005576"] = derived["go_id_text"].apply(
        lambda text: int(GO_EXTRACELLULAR_REGION in {part.strip() for part in text.split(";") if part.strip()})
    )
    derived["has_go_0005615"] = derived["go_id_text"].apply(
        lambda text: int(GO_EXTRACELLULAR_SPACE in {part.strip() for part in text.split(";") if part.strip()})
    )
    derived["is_secreted_or_extracellular_go"] = (
        (derived["is_secreted_kw_0964"] == 1)
        | (derived["has_go_0005576"] == 1)
        | (derived["has_go_0005615"] == 1)
    ).astype(int)

    secreted_df = derived.loc[
        (derived["is_secreted_kw_0964"] == 1) | (derived["is_secreted_or_extracellular_go"] == 1)
    ].copy()
    secreted_df = secreted_df[
        [
            "Entry",
            "Gene Names (primary)",
            "Gene Names",
            "Keywords",
            "Keyword ID",
            "Gene Ontology IDs",
            "Subcellular location [CC]",
            "gene_tokens",
            "is_secreted_kw_0964",
            "has_go_0005576",
            "has_go_0005615",
            "is_secreted_or_extracellular_go",
        ]
    ]
    secreted_df.to_csv(UNIPROT_SECRETED_PATH, sep="\t", index=False)

    kw_gene_set: set[str] = set()
    go_gene_set: set[str] = set()
    for _, row in derived.iterrows():
        gene_tokens = [token for token in str(row["gene_tokens"]).split(";") if token]
        if int(row["is_secreted_kw_0964"]) == 1:
            kw_gene_set.update(gene_tokens)
        if int(row["is_secreted_or_extracellular_go"]) == 1:
            go_gene_set.update(gene_tokens)

    summary = {
        "raw_entry_count": int(len(raw_df)),
        "derived_secreted_entry_count": int(len(secreted_df)),
        "secreted_kw_gene_count": int(len(kw_gene_set)),
        "secreted_or_extracellular_gene_count": int(len(go_gene_set)),
    }
    return secreted_df, {"summary": summary, "secreted_kw_genes": sorted(kw_gene_set), "secreted_go_genes": sorted(go_gene_set)}


def _parse_kegg_symbol_segment(segment: str) -> list[str]:
    before_semicolon = segment.split(";", 1)[0]
    tokens: set[str] = set()
    for token in [item.strip() for item in before_semicolon.split(",")]:
        if not token:
            continue
        normalized = _clean_token(token)
        if normalized is not None:
            tokens.add(normalized)
    return sorted(tokens)


def fetch_text_with_cache(session: requests.Session, url: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    path.write_text(response.text)
    return response.text


def fetch_kegg_ad_pathway(session: requests.Session) -> tuple[pd.DataFrame, dict]:
    link_text = fetch_text_with_cache(session, KEGG_LINK_URL, KEGG_LINK_PATH)
    list_text = fetch_text_with_cache(session, KEGG_LIST_URL, KEGG_LIST_PATH)

    link_rows = []
    for line in link_text.splitlines():
        if not line.strip():
            continue
        pathway_id, hsa_id = line.split("\t")
        link_rows.append({"pathway_id": pathway_id, "hsa_id": hsa_id})
    if not link_rows:
        raise RuntimeError("KEGG AD pathway link endpoint returned no rows.")
    link_df = pd.DataFrame(link_rows)

    hsa_to_symbols: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    for line in list_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise RuntimeError(f"Malformed KEGG list line: {line}")
        hsa_id = parts[0]
        description = parts[-1]
        symbols = _parse_kegg_symbol_segment(description)
        if symbols:
            hsa_to_symbols[hsa_id] = symbols
        descriptions[hsa_id] = description

    derived_rows: list[dict] = []
    pathway_gene_set: set[str] = set()
    for _, row in link_df.iterrows():
        hsa_id = row["hsa_id"]
        symbols = hsa_to_symbols.get(hsa_id)
        if not symbols:
            continue
        pathway_gene_set.update(symbols)
        derived_rows.append(
            {
                "pathway_id": row["pathway_id"],
                "hsa_id": hsa_id,
                "primary_symbol": symbols[0],
                "mapped_symbols": ";".join(symbols),
                "kegg_description": descriptions[hsa_id],
            }
        )
    if not derived_rows:
        raise RuntimeError("No KEGG hsa05010 entries could be mapped to gene symbols.")
    derived_df = pd.DataFrame(derived_rows).sort_values(["primary_symbol", "hsa_id"]).reset_index(drop=True)
    derived_df.to_csv(KEGG_DERIVED_PATH, sep="\t", index=False)
    summary = {
        "pathway_id": "hsa05010",
        "n_link_rows": int(len(link_df)),
        "n_mapped_hsa_ids": int(derived_df["hsa_id"].nunique()),
        "n_unique_gene_symbols": int(len(pathway_gene_set)),
    }
    return derived_df, {"summary": summary, "ad_driver_genes": sorted(pathway_gene_set)}


def attempt_disgenet() -> dict:
    api_key = os.environ.get("DISGENET_API_KEY")
    if not api_key:
        status = {
            "attempted": False,
            "available": False,
            "reason": "No DISGENET_API_KEY environment variable was present.",
        }
        DISGENET_STATUS_PATH.write_text(json.dumps(status, indent=2))
        return status

    status = {
        "attempted": True,
        "available": False,
        "reason": "Authenticated DisGeNET access was requested but no supported fetch implementation is checked in for this repo. KEGG-only prior retained.",
    }
    DISGENET_STATUS_PATH.write_text(json.dumps(status, indent=2))
    return status


def build_annotation_bundle() -> dict:
    ensure_results_dir()
    ensure_annotations_dir()
    tables = load_all_tables()
    scored_universe = build_gene_universe(tables)
    scored_universe_set = set(scored_universe)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "ChronoABOR-revision/1.0"})

    uniprot_raw_df, uniprot_manifest = fetch_uniprot_reviewed_human(session)
    _, uniprot_payload = derive_uniprot_secreted_table(uniprot_raw_df)
    kegg_df, kegg_payload = fetch_kegg_ad_pathway(session)
    disgenet_status = attempt_disgenet()

    secreted_kw_set = set(uniprot_payload["secreted_kw_genes"])
    secreted_go_set = set(uniprot_payload["secreted_go_genes"])
    ad_driver_set = set(kegg_payload["ad_driver_genes"])

    bundle = {
        "uniprot": {
            "manifest": uniprot_manifest,
            "derived_tsv_path": str(UNIPROT_SECRETED_PATH),
            "secreted_kw_gene_count_total": int(len(secreted_kw_set)),
            "secreted_go_gene_count_total": int(len(secreted_go_set)),
            "secreted_kw_overlap_with_scored_universe": int(len(secreted_kw_set & scored_universe_set)),
            "secreted_go_overlap_with_scored_universe": int(len(secreted_go_set & scored_universe_set)),
            "secreted_kw_genes": sorted(secreted_kw_set),
            "secreted_go_genes": sorted(secreted_go_set),
        },
        "kegg": {
            "summary": kegg_payload["summary"],
            "derived_tsv_path": str(KEGG_DERIVED_PATH),
            "ad_driver_gene_count_total": int(len(ad_driver_set)),
            "ad_driver_overlap_with_scored_universe": int(len(ad_driver_set & scored_universe_set)),
            "ad_driver_genes": sorted(ad_driver_set),
            "n_derived_rows": int(len(kegg_df)),
        },
        "disgenet": disgenet_status,
        "prior_definition": {
            "is_secreted_primary": "UniProt reviewed human keyword KW-0964 (Secreted) from the current release cached locally.",
            "is_secreted_sensitivity": "UniProt KW-0964 OR GO cellular-component term GO:0005576 / GO:0005615.",
            "is_ad_driver": "KEGG hsa05010 only" if not disgenet_status["available"] else "KEGG hsa05010 plus DisGeNET curated AD associations",
        },
        "scored_universe": {
            "n_genes": int(len(scored_universe)),
            "genes": scored_universe,
        },
    }
    return bundle


def main() -> None:
    start = time.time()
    bundle = build_annotation_bundle()
    payload = {
        "provenance": provenance(
            SCRIPT_NAME,
            {
                "random_seed": RANDOM_SEED,
                "uniprot_keyword_definition": UNIPROT_SECRETED_KEYWORD_ID,
                "go_extracellular_terms": sorted(GO_EXTRACELLULAR_TERMS),
                "kegg_pathway": "hsa05010",
            },
            [],
        ),
        "annotation_bundle": bundle,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(OUTPUT_JSON, payload)

    print(
        f"UniProt {bundle['uniprot']['manifest']['release']} ({bundle['uniprot']['manifest']['release_date']}): "
        f"KW-0964 overlap={bundle['uniprot']['secreted_kw_overlap_with_scored_universe']}, "
        f"GO-broader overlap={bundle['uniprot']['secreted_go_overlap_with_scored_universe']}"
    )
    print(
        f"KEGG hsa05010: {bundle['kegg']['ad_driver_gene_count_total']} genes total, "
        f"{bundle['kegg']['ad_driver_overlap_with_scored_universe']} in the 2,843-gene scored universe"
    )
    print(f"DisGeNET available: {bundle['disgenet']['available']} ({bundle['disgenet']['reason']})")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
