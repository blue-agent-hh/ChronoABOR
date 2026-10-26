#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CACHE_PATH = RESULTS_DIR / "pub_year_cache.json"
AUDIT_CSV = RESULTS_DIR / "pub_year_resolution_audit.csv"
from urllib.request import ProxyHandler, build_opener

opener = build_opener()

YEAR_RE = re.compile(r"(19|20)\d{2}")
DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)
PMC_RE = re.compile(r"PMC\d+", re.I)
PII_RE = re.compile(r"[A-Z0-9]{16,}X?", re.I)
PMID_IN_TEXT_RE = re.compile(r"pmid:?\s*(\d+)", re.I)

JOURNAL_PREFIX_HINTS = {
    "S41598": "10.1038/s41598",
    "S41592": "10.1038/s41592",
    "S43587": "10.1038/s43587",
    "S41392": "10.1038/s41392",
    "S13195": "10.1186/s13195",
    "S13024": "10.1186/s13024",
    "S13040": "10.1186/s13040",
    "S12967": "10.1186/s12967",
    "S41746": "10.1038/s41746",
    "13024": "10.1186/s13024",
}


def http_json(url: str, retries: int = 3, sleep: float = 0.4):
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "abor-year-resolve/1.0 (mailto:fbguoy@whu.edu.cn)"})
            with opener.open(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
            time.sleep(sleep * (attempt + 1))
    raise last_err


def fetch_pmid_years(pmids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    batch = 150
    for start in range(0, len(pmids), batch):
        chunk = pmids[start : start + batch]
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={','.join(chunk)}&retmode=json"
        )
        data = http_json(url)
        for pmid in chunk:
            rec = data.get("result", {}).get(pmid, {})
            pubdate = rec.get("pubdate", "") or rec.get("sortpubdate", "")
            match = YEAR_RE.search(str(pubdate))
            if match:
                out[pmid] = int(match.group(0))
        time.sleep(0.35)
    return out


def fetch_pmc_year(pmc_id: str) -> int | None:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pmc&id={pmc_id.replace('PMC','')}&retmode=json"
    )
    try:
        data = http_json(url)
    except Exception:
        return None
    uid = pmc_id.replace("PMC", "")
    rec = data.get("result", {}).get(uid, {})
    pubdate = rec.get("pubdate", "") or rec.get("pmid", "")
    match = YEAR_RE.search(str(rec.get("pubdate", "")))
    if match:
        return int(match.group(0))
    pmid = str(rec.get("pmid", ""))
    if pmid.isdigit():
        years = fetch_pmid_years([pmid])
        return years.get(pmid)
    return None


def normalize_doi(raw: str) -> str | None:
    text = raw.strip()
    text = re.sub(r"^doi:?\s*", "", text, flags=re.I)
    text = text.removesuffix("/FULL").removesuffix("/full")
    match = DOI_RE.search(text)
    if match:
        return match.group(0)
    return None


def reconstruct_from_suffix(raw: str) -> str | None:
    token = raw.strip().lstrip("Ss").upper()
    for hint_key, full_prefix in JOURNAL_PREFIX_HINTS.items():
        if token.startswith(hint_key.upper()):
            suffix = raw.strip()[len(hint_key):]
            return full_prefix + suffix
        if raw.strip().upper().startswith("S" + hint_key.upper()):
            return full_prefix + raw.strip()[len(hint_key) + 1 :]
    return None


def fetch_crossref_year(doi: str) -> int | None:
    from urllib.parse import quote

    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        data = http_json(url)
    except HTTPError:
        return None
    msg = data.get("message", {})
    for key in ("issued", "published-print", "published-online", "created"):
        parts = msg.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def resolve_one(raw_id: str, id_type: str, pmid_years: dict[str, int]) -> tuple[int | None, str]:
    text = raw_id.strip()
    if id_type == "PMID" and text.isdigit():
        if text in pmid_years:
            return pmid_years[text], "ncbi_pmid"
        return None, "ncbi_pmid_unresolved"

    if text.lower().startswith("pmid"):
        match = PMID_IN_TEXT_RE.search(text)
        if match:
            pmid = match.group(1)
            if pmid in pmid_years:
                return pmid_years[pmid], "url_embedded_pmid"
        return None, "url_embedded_pmid_unresolved"

    if id_type == "URL":
        host = urlsplit(text).netloc.lower()
        if "clinicaltrials.gov" in host or "fda.gov" in host or "alz.org" in host:
            return None, "non_literature_url"
        doi = normalize_doi(text)
        if doi:
            year = fetch_crossref_year(doi)
            return (year, "crossref_url_doi") if year else (None, "crossref_url_doi_unresolved")
        pmc_match = PMC_RE.search(text)
        if pmc_match:
            year = fetch_pmc_year(pmc_match.group(0))
            return (year, "ncbi_pmc") if year else (None, "ncbi_pmc_unresolved")
        return None, "url_unresolved"

    doi = normalize_doi(text)
    if doi:
        year = fetch_crossref_year(doi)
        if year:
            return year, "crossref_doi"
        pmc_match = PMC_RE.search(text)
        if pmc_match:
            year = fetch_pmc_year(pmc_match.group(0))
            if year:
                return year, "ncbi_pmc_from_doi_field"
        return None, "crossref_doi_unresolved"
    recon = reconstruct_from_suffix(text)
    if recon:
        year = fetch_crossref_year(recon)
        return (year, "crossref_reconstructed_doi") if year else (None, "reconstructed_doi_unresolved")
    pmc_match = PMC_RE.search(text)
    if pmc_match:
        year = fetch_pmc_year(pmc_match.group(0))
        return (year, "ncbi_pmc") if year else (None, "ncbi_pmc_unresolved")
    return None, "unresolved"


def main() -> None:
    sys.path.insert(0, str(BASE_DIR))
    from rev_exp_common import annotate_tables_with_study_metadata

    annotated, _, _ = annotate_tables_with_study_metadata()
    unique: dict[str, str] = {}
    for layer, df in annotated.items():
        for sid, stype in zip(df["_study_id"], df["_study_id_type"]):
            unique[sid] = stype

    pmids = [sid for sid, stype in unique.items() if stype == "PMID" and sid.isdigit()]
    print(f"Resolving {len(pmids)} PMIDs via NCBI esummary ...", flush=True)
    pmid_years = fetch_pmid_years(pmids)
    print(f"  resolved {len(pmid_years)}/{len(pmids)}", flush=True)

    cache: dict[str, dict] = {}
    audit_rows = []
    n_res, n_unres = 0, 0
    for sid, stype in sorted(unique.items()):
        year, method = resolve_one(sid, stype, pmid_years)
        cache[sid] = {"year": year, "method": method, "id_type": stype}
        audit_rows.append([sid, stype, method, year if year is not None else ""])
        if year is not None:
            n_res += 1
        else:
            n_unres += 1
        time.sleep(0.05)

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    with AUDIT_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["study_id", "id_type", "method", "year"])
        writer.writerows(audit_rows)

    print(f"resolved={n_res} unresolved={n_unres} of {len(unique)}")
    print(f"cache -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
