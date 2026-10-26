#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, build_opener, ProxyHandler

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "results" / "pub_year_cache.json"
opener = build_opener(ProxyHandler())

PREFIX_MAP = [
    ("S41598", "10.1038/S41598"),
    ("S41592", "10.1038/S41592"),
    ("S43587", "10.1038/S43587"),
    ("S41392", "10.1038/S41392"),
    ("S41746", "10.1038/S41746"),
    ("S13195", "10.1186/S13195"),
    ("S13024", "10.1186/S13024"),
    ("S13040", "10.1186/S13040"),
    ("S12967", "10.1186/S12967"),
    ("13195", "10.1186/S13195"),
    ("13024", "10.1186/S13024"),
]


def get(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "abor-year-resolve-pass2/1.0 (mailto:fbguoy@whu.edu.cn)"})
            with opener.open(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def crossref_year(doi: str) -> int | None:
    data = get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    if not data:
        return None
    msg = data.get("message", {})
    for key in ("issued", "published-print", "published-online", "created"):
        parts = msg.get(key, {}).get("date-parts")
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def crossref_query_year(query: str) -> int | None:
    data = get(f"https://api.crossref.org/works?query.bibliographic={quote(query)}&rows=3")
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    for item in items:
        for key in ("issued", "published-print", "published-online", "created"):
            parts = item.get(key, {}).get("date-parts")
            if parts and parts[0]:
                return int(parts[0][0])
    return None


def try_suffix_prefix(token: str) -> tuple[int | None, str | None]:
    for suffix, prefix in PREFIX_MAP:
        if token.upper().startswith(suffix.upper()):
            doi = prefix + token[len(suffix):]
            year = crossref_year(doi)
            if year:
                return year, f"crossref_suffix_reconstructed:{doi}"
    return None, None


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text())
    fixed = 0
    for sid, entry in list(cache.items()):
        if entry["year"] is not None:
            continue
        if entry["method"] == "non_literature_url":
            continue
        token = sid
        m = re.search(r"(S?\d{4,5}[-\w()]+$)", sid)
        if m:
            candidate = m.group(1)
        else:
            candidate = sid
        year, method = try_suffix_prefix(candidate)
        if year is None:
            inner = re.search(r"(S?\d{4,5}[-\w()]+)", sid)
            if inner and inner.group(1) != candidate:
                year, method = try_suffix_prefix(inner.group(1))
        if year is None:
            year = crossref_query_year(sid)
            method = f"crossref_query" if year else None
        if year is not None:
            cache[sid]["year"] = year
            cache[sid]["method"] = method
            fixed += 1
            print(f"FIXED {sid} -> {year} ({method})")
        else:
            print(f"STILL UNRESOLVED {sid}")
        time.sleep(0.15)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"fixed {fixed}")


if __name__ == "__main__":
    main()
