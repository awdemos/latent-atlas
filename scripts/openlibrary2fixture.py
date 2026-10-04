#!/usr/bin/env python3
"""Build a Latent Atlas curated works fixture from the Open Library API.

Target: 500-1000 well-known books with verifiable first-publication years,
drawn from the Open Library "subjects" API (no key required). For each work
the fixture records the first publication year, upgraded to confidence=high
when an edition-level publish_date on /works/<olid>/editions.json confirms
it (or supplies an even earlier plausible year), and kept at confidence=
medium when only the subject-level first_publish_year is available.

Sources (read-only, live HTTP):
  https://openlibrary.org/subjects/<subject>.json?limit=100
      -> works[].{key,title,first_publish_year,edition_count}
  https://openlibrary.org/works/<olid>/editions.json?limit=200&offset=N
      -> entries[].publish_date  (free-text, e.g. "c1985", "Jan 1985",
         "[1985]", "1985?")
  https://openlibrary.org/search.json?title=...&fields=key,title,
      first_publish_year  (used only to guarantee the spot-check titles)

Confidence rule (documented per row in `notes`):
  high   - start_year taken from the earliest plausible 4-digit year found
           in the work's edition publish_date strings
  medium - only the subject/search first_publish_year was available (or the
           edition-derived earliest year was implausibly LATER than
           first_publish_year, suggesting the true first edition is not
           catalogued among the fetched editions)

Selection: a fixed list of broad "best of" subject keys is fetched with
limit=100 each. Works are de-duplicated by OLID in subject order (the
subject API roughly orders by catalogue weight) and capped at --max-works.
Five canonical titles (Pride and Prejudice, Frankenstein, The Time
Machine, Nineteen Eighty-Four, Dune) are force-included via the search API
if a subject did not surface them, so the spot checks always have rows.

Year convention: all years are CE publication years (>= 1450 accepted by
the publish-date parser), so astronomical numbering changes nothing.

Output schema (fixtures/curated_works_openlibrary.csv):
  entity_id,entity_type,canonical_name,aliases,description,relation,
  start_year,end_year,confidence,notes
  entity_id   = ol:<olid-lowercase>   e.g. ol:ol138052w
  entity_type = work, relation = available, end_year = "" (open bound)

Attribution: Open Library data is provided by the Internet Archive under
the CC0 1.0 public-domain dedication (see
https://openlibrary.org/developers/licensing); attribution to Open
Library is appreciated but not required. No Open Library dumps are
downloaded -- only these small per-work JSON documents.

Usage:
  python3 scripts/openlibrary2fixture.py [--max-works 900]
      [--raw-dir DIR] [--out fixtures/curated_works_openlibrary.csv]
      [--spot-check]

Stdlib only. Raw downloads are cached under the system temp dir by
default (outside the repo); re-running is cheap once cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://openlibrary.org"
USER_AGENT = "latent-atlas-fixture/1.0 (openlibrary2fixture.py; research)"

HEADER = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]

# Broad, well-populated subject keys. Chosen to span periods and genres so
# the fixture covers 19th-century classics through modern bestsellers.
SUBJECTS = [
    "classic_literature", "science_fiction", "fantasy",
    "mystery_and_detective_stories", "horror", "romance",
    "historical_fiction", "adventure", "literary_fiction",
    "dystopias", "short_stories", "philosophy", "poetry",
    "childrens_books", "biography",
]

# Canonical titles guaranteed to appear in the fixture (spot-check anchors).
ANCHOR_TITLES = [
    "Pride and Prejudice",
    "Frankenstein",
    "The Time Machine",
    "Nineteen Eighty-Four",
    "Dune",
]

PLAUSIBLE_YEAR = range(1450, 2027)  # Gutenberg-era onward; reject OCR garbage

DATE_YEAR_RE = re.compile(r"(?<!\d)(1[4-9]\d{2}|20[0-2]\d)(?!\d)")


# ---------------------------------------------------------------------------
# HTTP with on-disk cache


def fetch_json(url: str, raw_dir: Path, name: str) -> dict:
    """GET url, caching the response body under raw_dir/name.json.

    Retries up to 3 times with backoff on network errors / 5xx / 429."""
    path = raw_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    body = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            code = getattr(getattr(exc, "fp", None), "status", None) or \
                getattr(exc, "code", None)
            if attempt == 2 or (code is not None and code < 500
                                and code != 429):
                raise RuntimeError(f"fetch failed for {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    if body is None:
        raise RuntimeError(f"fetch failed for {url}: no response")
    path.write_bytes(body)
    time.sleep(0.25)  # be polite to the API
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Year extraction from free-text publish_date strings


def years_from_date(s: str) -> list[int]:
    """Extract plausible 4-digit years from a publish_date string.

    Handles "1985", "c1985", "1985?", "[1985]", "Jan 1, 1985",
    "1st pub 1965", "1985-03"."""
    if not s:
        return []
    s = str(s)
    out = []
    for m in DATE_YEAR_RE.finditer(s):
        y = int(m.group(1))
        if y in PLAUSIBLE_YEAR:
            out.append(y)
    return out


def earliest_edition_year(olid: str, raw_dir: Path,
                          max_pages: int = 5) -> tuple[int | None, int]:
    """Return (earliest plausible year among edition publish_dates, size).

    Fetches up to max_pages * 200 editions. Dates are free text and often
    imprecise ("c1985"), so the earliest plausible 4-digit year wins."""
    earliest: int | None = None
    size = 0
    for page in range(max_pages):
        d = fetch_json(
            f"{API}/works/{olid}/editions.json?limit=200&offset={page * 200}",
            raw_dir, f"editions_{olid}_{page}")
        size = int(d.get("size", 0))
        for e in d.get("entries", []):
            for y in years_from_date(e.get("publish_date", "")):
                if earliest is None or y < earliest:
                    earliest = y
        if (page + 1) * 200 >= size:
            break
    return earliest, size


# ---------------------------------------------------------------------------
# Candidate collection


def collect_candidates(raw_dir: Path, max_works: int) -> dict[str, dict]:
    """work OLID -> {title, fpy}. Subjects first, then anchor-title search."""
    works: dict[str, dict] = {}
    anchor_olids: set[str] = set()

    def add(key: str, title: str, fpy: int | None, anchor: bool = False) -> None:
        olid = key.rstrip("/").rsplit("/", 1)[-1]
        if olid and olid not in works:
            works[olid] = {"olid": olid, "title": title, "fpy": fpy}
            if anchor:
                anchor_olids.add(olid)

    for subj in SUBJECTS:
        try:
            d = fetch_json(f"{API}/subjects/{subj}.json?limit=100",
                           raw_dir, f"subject_{subj}")
        except RuntimeError as exc:
            print(f"WARN: subject {subj} failed: {exc}", file=sys.stderr)
            continue
        for w in d.get("works", []):
            add(w.get("key", ""), w.get("title", ""), w.get("first_publish_year"))
        print(f"subject {subj}: total unique works now {len(works)}",
              file=sys.stderr)
        if len(works) >= max_works:
            break

    # Guarantee the spot-check anchors are present. Accept an exact title
    # match or a "<title>; <subtitle>" / "<title>: <subtitle>" form, and
    # among matches pick the earliest first_publish_year so the canonical
    # work wins over later books sharing the title (e.g. the 2017
    # "Frankenstein" vs Mary Shelley's 1818 work).
    for title in ANCHOR_TITLES:
        if any(title.lower() == w["title"].lower() for w in works.values()):
            continue
        q = urllib.parse.urlencode({"title": title, "limit": 10,
                                    "fields": "key,title,first_publish_year"})
        try:
            d = fetch_json(f"{API}/search.json?{q}", raw_dir,
                           f"search_{title.lower().replace(' ', '_')}")
        except RuntimeError as exc:
            print(f"WARN: anchor search {title!r} failed: {exc}",
                  file=sys.stderr)
            continue
        matches = []
        for doc in d.get("docs", []):
            t = doc.get("title", "").lower()
            if t == title.lower() or t.startswith(title.lower() + ";") \
                    or t.startswith(title.lower() + ":"):
                matches.append(doc)
        if matches:
            doc = min(matches,
                      key=lambda x: (x.get("first_publish_year") or 9999))
            add(doc["key"], doc["title"], doc.get("first_publish_year"),
                anchor=True)
            print(f"anchor {title}: added {doc['key']} "
                  f"(fpy={doc.get('first_publish_year')})", file=sys.stderr)
        else:
            print(f"WARN: no search match for anchor {title!r}",
                  file=sys.stderr)

    # Cap at max_works WITHOUT dropping anchors: keep the first
    # (max_works - n_anchors) subject-order works, then the anchors.
    cap = max_works - len(anchor_olids)
    kept = {k: v for k, v in works.items() if k not in anchor_olids}
    kept = dict(list(kept.items())[:cap])
    kept.update({k: works[k] for k in anchor_olids})
    return kept


# ---------------------------------------------------------------------------
# Fixture assembly


def build_rows(candidates: dict[str, dict], raw_dir: Path) -> tuple[list[dict], list[str]]:
    rows, dropped = [], []
    for i, (olid, w) in enumerate(sorted(candidates.items())):
        title = (w["title"] or "").strip()
        if not title:
            dropped.append(f"{olid}: empty title")
            continue
        ed_year = ed_size = None
        try:
            ed_year, ed_size = earliest_edition_year(olid, raw_dir)
        except RuntimeError as exc:
            print(f"WARN: editions fetch failed for {olid}: {exc}",
                  file=sys.stderr)
        fpy = w["fpy"] if w["fpy"] in PLAUSIBLE_YEAR else None
        if ed_year is not None and (fpy is None or ed_year <= fpy):
            year, confidence = ed_year, "high"
            basis = (f"earliest edition publish_date {ed_year} across "
                     f"{ed_size} catalogued editions")
        elif fpy is not None:
            year, confidence = fpy, "medium"
            basis = (f"work-level first_publish_year {fpy} only"
                     + (f"; edition dates start {ed_year} (later; first "
                        f"edition likely uncatalogued)" if ed_year else ""))
        elif ed_year is not None:
            year, confidence = ed_year, "medium"
            basis = (f"edition publish_date {ed_year}; work-level "
                     f"first_publish_year missing")
        else:
            dropped.append(f"{olid} ({title}): no date information")
            continue
        if year not in PLAUSIBLE_YEAR:
            dropped.append(f"{olid} ({title}): year {year} implausible")
            continue
        rows.append({
            "entity_id": f"ol:{olid.lower()}",
            "entity_type": "work",
            "canonical_name": title,
            "aliases": "",
            "description": "",
            "relation": "available",
            "start_year": year,
            "end_year": "",
            "confidence": confidence,
            "notes": ("Open Library (openlibrary.org), retrieved 2026-10-04; "
                      "CC0 1.0, attribution appreciated; " + basis),
        })
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(candidates)} works",
                  file=sys.stderr)
    return rows, dropped


def validate(path: Path, expect_min: int, expect_max: int) -> None:
    """Re-read the written CSV and assert every parser rule."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == HEADER, f"header mismatch: {header}"
        rows = list(reader)
    assert expect_min <= len(rows) <= expect_max, \
        f"row count {len(rows)} outside [{expect_min}, {expect_max}]"
    ids = set()
    for r in rows:
        assert len(r) == len(HEADER), f"column count: {r}"
        rid, etype, name, aliases, desc, rel, sy, ey, conf, notes = r
        assert rid.startswith("ol:") and rid not in ids, f"bad/dup id: {rid}"
        ids.add(rid)
        assert etype == "work" and rel == "available"
        assert conf in {"high", "medium", "low"}
        int(sy)  # must parse as int
        assert ey == ""  # open bound: the work remains available
        assert name and notes
    conf = Counter(r[8] for r in rows)
    print(f"VALIDATION OK: {len(rows)} rows, header exact, ids unique, "
          f"enums valid; confidence high={conf['high']} "
          f"medium={conf['medium']} low={conf['low']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    ap.add_argument("--max-works", type=int, default=900,
                    help="cap on unique works in the fixture")
    ap.add_argument("--raw-dir",
                    default=str(Path(tempfile.gettempdir())
                                / "latent-atlas-openlibrary-raw"),
                    help="cache dir for raw API downloads (outside the repo)")
    ap.add_argument("--out", default=str(here.parent.parent / "fixtures"
                                         / "curated_works_openlibrary.csv"))
    ap.add_argument("--spot-check", action="store_true",
                    help="print the anchor-title rows to stderr")
    args = ap.parse_args()

    assert 500 <= args.max_works <= 1000, "target is 500-1000 works"
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(raw_dir, args.max_works)
    print(f"candidates: {len(candidates)} unique works", file=sys.stderr)
    rows, dropped = build_rows(candidates, raw_dir)

    rows.sort(key=lambda r: r["entity_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    validate(out, 500, args.max_works)
    for d in dropped:
        print(f"DROPPED: {d}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)

    if args.spot_check:
        print("\nSPOT CHECK anchors:", file=sys.stderr)
        for title in ANCHOR_TITLES:
            match = [r for r in rows if r["canonical_name"].lower()
                     .startswith(title.lower())]
            if match:
                r = match[0]
                print(f"  {r['canonical_name']:24s} {r['entity_id']:16s} "
                      f"{r['start_year']} {r['confidence']}", file=sys.stderr)
            else:
                print(f"  {title:24s} ABSENT", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
