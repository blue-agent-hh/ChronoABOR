#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

_CACHE_PATH = Path(__file__).resolve().parent / "results" / "pub_year_cache.json"
_YEAR_MAP: dict[str, int] | None = None

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)
_PMID_RE = re.compile(r"^(\d{5,9})$")


def _load() -> dict[str, int]:
    global _YEAR_MAP
    if _YEAR_MAP is None:
        _YEAR_MAP = {}
        if _CACHE_PATH.exists():
            raw = json.loads(_CACHE_PATH.read_text())
            for key, entry in raw.items():
                year = entry.get("year")
                if year is not None:
                    _YEAR_MAP[key] = int(year)
    return _YEAR_MAP


def external_year_lookup(raw_value: object) -> int | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return None
    year_map = _load()

    pmid_match = _PMID_RE.match(text)
    if pmid_match and pmid_match.group(1) in year_map:
        return year_map[pmid_match.group(1)]

    if text in year_map:
        return year_map[text]
    if text.upper() in year_map:
        return year_map[text.upper()]

    doi_match = _DOI_RE.search(text)
    if doi_match:
        doi = doi_match.group(0).upper().rstrip("./FULL").removesuffix("/FULL")
        if doi in year_map:
            return year_map[doi]
        if text.upper() in year_map:
            return year_map[text.upper()]
    return None
