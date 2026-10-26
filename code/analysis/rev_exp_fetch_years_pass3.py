#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, build_opener, ProxyHandler

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "results" / "pub_year_cache.json"
opener = build_opener(ProxyHandler())

TARGETS = [
    "https://www.nature.com/articles/s41392-023-01547-9",
    "https://www.nature.com/articles/s41392-023-01588-0",
    "https://www.nature.com/articles/s41392-024-01839-8",
    "https://www.sciencedirect.com/science/article/pii/S0033062019300386",
    "https://www.sciencedirect.com/science/article/pii/S1673852721003581",
    "https://www.sciencedirect.com/science/article/pii/S2211383522000211",
    "https://www.sciencedirect.com/science/article/pii/S235230421630023X",
    "S092544392500273X",
    "S2666-3791(24)00465-8",
]


def get(url: str, retries: int = 3, as_text: bool = False):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
            with opener.open(req, timeout=30) as resp:
                body = resp.read()
                return body.decode("utf-8", errors="replace") if as_text else json.loads(body.decode("utf-8"))
        except Exception as exc:
            print(f"  retry {attempt+1} for {url[:80]}: {exc}")
            time.sleep(0.5 * (attempt + 1))
    return None


def crossref_year(doi: str) -> int | None:
    data = get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    if not data or not isinstance(data, dict):
        return None
    msg = data.get("message", {})
    for key in ("issued", "published-print", "published-online", "created"):
        parts = msg.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def doi_from_sciencedirect(pii: str) -> str | None:
    html = get(f"https://www.sciencedirect.com/science/article/pii/{pii}", as_text=True)
    if not html:
        return None
    m = re.search(r'citation_doi"?\s*(?:content=|:)\s*"?(10\.\d{4,9}/[^"&<\s]+)', html, re.I)
    if m:
        return m.group(1)
    m = re.search(r'"doi"\s*:\s*"(10\.\d{4,9}/[^"]+)"', html)
    if m:
        return m.group(1)
    return None


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text())
    for sid in TARGETS:
        if sid not in cache:
            print(f"SKIP (not in cache): {sid}")
            continue
        year, method = None, None
        if "nature.com/articles/" in sid:
            slug = sid.rsplit("/", 1)[-1]
            doi = f"10.1038/{slug}"
            year = crossref_year(doi)
            method = f"crossref_nature_doi:{doi}"
        elif "sciencedirect.com" in sid:
            pii = sid.rsplit("/", 1)[-1]
            doi = doi_from_sciencedirect(pii)
            if doi:
                year = crossref_year(doi)
                method = f"crossref_sd_doi:{doi}"
        elif sid.startswith("S092544392500273X"):
            doi = doi_from_sciencedirect("S092544392500273X")
            if doi:
                year = crossref_year(doi)
                method = f"crossref_sd_doi:{doi}"
        elif sid.startswith("S2666-3791"):
            doi = f"10.1016/j.dadm.{sid.split('(')[1][:2]}.xx"
            doi = None
            html = get(f"https://doi.org/10.1016/j.trci.2024.{sid[-6:-2]}", as_text=True)
            year = None
        if year is not None:
            cache[sid]["year"] = year
            cache[sid]["method"] = method
            print(f"FIXED {sid} -> {year} ({method})")
        else:
            cache[sid]["year"] = None
            cache[sid]["method"] = "unresolved_after_pass3"
            print(f"UNRESOLVED {sid}")
        time.sleep(0.2)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


if __name__ == "__main__":
    main()
