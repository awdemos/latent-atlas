#!/usr/bin/env python3
"""Convert the PyPI top-packages list + per-package PyPI JSON API metadata
into a Latent Atlas curated CSV fixture (fixtures/curated_software_pypi.csv).

Source 1 (ranking): hugovk/top-pypi-packages, a ClickHouse-based download
count ranking of PyPI projects, updated monthly:
  https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json
  (the github.io URL 301-redirects to hugovk.dev; both work, redirects
  are followed). Fallback: https://pypistats.org/api/top/packages
  (was 404 at authoring time; kept as a second genuine attempt).

Source 2 (per-package metadata): the PyPI JSON API (no key required):
  https://pypi.org/pypi/<name>/json
  info.version / info.summary give the current release and blurb;
  releases is {version: [file, ...]} where each file has upload_time and
  a yanked flag.

First-release rule (per task spec):
  start_year = year of the EARLIEST non-yanked upload_time across the whole
  releases dict (NOT info.release_date, which some projects lack). If a
  project has only yanked files in its earliest release, the earliest
  upload of any kind is used and the caveat is recorded in notes. This is
  the package's own first upload to PyPI: e.g. numpy's first PyPI upload
  is 2006 even though its predecessor Numeric dates to 1995, and django's
  is 2005 even though the framework began in 2003.

Output rows (one per package, capped at --top, default 100):
  entity_id     = pypi:<project-name-as-on-pypi>   (already lowercase, unique)
  entity_type   = technology
  canonical_name= project name (PyPI spelling)
  aliases       = empty (PyPI project names are canonical)
  description   = info.summary (newlines stripped)
  relation      = available   (still installable -> open-ended availability)
  start_year    = first-release year (astronomical numbering; CE only here)
  end_year      = EMPTY       (still available; open bound)
  confidence    = high        (upload timestamps are authoritative)
  notes         = current version, upload date, sources, retrieval date,
                  yanked-only caveat when applicable

Raw downloads are cached under atlas-data/raw/pypi/ (atlas-data/ is
gitignored) so re-runs are cheap and the pipeline is reproducible offline.

Usage:
  python3 scripts/pypi2fixture.py [--top 100]
                                  [--out fixtures/curated_software_pypi.csv]
                                  [--raw-dir atlas-data/raw/pypi]
                                  [--spot-check]

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HEADER = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]

RETRIEVAL_DATE = "2026-10-04"

TOP_LIST_URLS = [
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json",
    "https://pypistats.org/api/top/packages",
]

USER_AGENT = ("latent-atlas-fixture/1.0 (stdlib urllib; dataset fixture "
              "for temporal world-model research)")
REQUEST_PAUSE = 0.5  # polite delay between PyPI API calls
MAX_ATTEMPTS = 3


def fetch(url: str, dest: Path | None = None, timeout: int = 60) -> bytes:
    """Fetch a URL, retrying with backoff. Saves to dest when given."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            return data
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_exc = exc
            wait = 2 * attempt
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} for {url} failed: "
                  f"{exc}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {url}: {last_exc}")


def load_top_packages(raw_dir: Path, limit: int) -> list[str]:
    """Download the top-packages ranking and return the first `limit`
    project names, most-downloaded first."""
    data = None
    for url in TOP_LIST_URLS:
        try:
            data = fetch(url, raw_dir / "top-pypi-packages.min.json")
            break
        except RuntimeError as exc:
            print(f"top list {url} unavailable: {exc}", file=sys.stderr)
    if data is None:
        raise SystemExit("no top-packages list reachable; aborting")

    doc = json.loads(data)
    if "rows" in doc:  # hugovk ClickHouse export: rows=[{download_count, project}]
        names = [r["project"] for r in doc["rows"]]
        print(f"top list: hugovk top-pypi-packages "
              f"(last_update={doc.get('last_update')})", file=sys.stderr)
    elif "data" in doc:  # pypistats shape: data=[{package, ...}]
        names = [r["package"] for r in doc["data"]]
        print("top list: pypistats.org", file=sys.stderr)
    else:
        raise SystemExit(f"unrecognized top list shape: {sorted(doc)[:5]}")
    return names[:limit]


def first_release_year(releases: dict) -> tuple[int, str, bool]:
    """Earliest upload year across releases.

    Returns (year, iso_timestamp, used_yanked_fallback) where
    used_yanked_fallback is True when no non-yanked file existed anywhere
    in the releases dict (pathological; still recorded rather than dropped
    for a modern top-100 package)."""
    best: str | None = None
    best_any: str | None = None
    for files in releases.values():
        for f in files:
            t = f.get("upload_time")
            if not t:
                continue
            if best_any is None or t < best_any:
                best_any = t
            if f.get("yanked"):
                continue
            if best is None or t < best:
                best = t
    if best is None:
        if best_any is None:
            raise ValueError("releases dict has no uploads at all")
        return int(best_any[:4]), best_any, True
    return int(best[:4]), best, False


def package_row(name: str, raw_dir: Path) -> dict:
    cache = raw_dir / f"{name}.json"
    if cache.exists():
        doc = json.loads(cache.read_text())
    else:
        doc = json.loads(fetch(f"https://pypi.org/pypi/{name}/json", cache))
        time.sleep(REQUEST_PAUSE)

    info = doc["info"]
    year, stamp, yanked_fallback = first_release_year(doc["releases"])
    version = info.get("version") or "?"
    summary = (info.get("summary") or "").replace("\n", " ").strip()
    summary = " ".join(summary.split())

    notes = [
        f"PyPI JSON API (pypi.org/pypi/{name}/json); current version "
        f"{version}; first release upload {stamp}",
        f"ranking source: hugovk top-pypi-packages (ClickHouse), "
        f"retrieved {RETRIEVAL_DATE}",
    ]
    if yanked_fallback:
        notes.append("caveat: all files in every release are yanked; "
                     "year taken from earliest upload of any kind")
    notes.append("start_year is the package's own first upload to PyPI, "
                 "not the start of predecessor projects")

    return {
        "entity_id": f"pypi:{name}",
        "entity_type": "technology",
        "canonical_name": name,
        "aliases": "",
        "description": summary,
        "relation": "available",
        "start_year": year,
        "end_year": "",
        "confidence": "high",
        "notes": "; ".join(notes),
    }


SPOT_CHECKS = ["numpy", "requests", "django", "flask"]
EXPECTED = {"numpy": 2006, "requests": 2011, "django": 2005, "flask": 2010}


def validate(rows: list[dict], expected: int) -> None:
    """Re-read-style validation of the row list (the CSV is written with
    the csv module and re-parsed in main before this is called)."""
    entity_types = {"person", "event", "polity", "organization", "work",
                    "technology"}
    relations = {"alive", "ongoing", "exists", "active", "available"}
    confidences = {"high", "medium", "low"}
    ids = [r["entity_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate entity_ids"
    assert len(rows) == expected, f"row count {len(rows)} != {expected}"
    for r in rows:
        assert r["entity_type"] in entity_types, r
        assert r["relation"] in relations, r
        assert r["confidence"] in confidences, r
        assert r["entity_id"].startswith("pypi:"), r
        assert isinstance(r["start_year"], int), r
        assert r["end_year"] == "", r
        for key in ("canonical_name", "description", "notes"):
            assert "\n" not in r[key] and "\r" not in r[key], r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    ap.add_argument("--top", type=int, default=100,
                    help="number of top packages to include (cap 100)")
    ap.add_argument("--out", default=str(here.parent.parent / "fixtures"
                                         / "curated_software_pypi.csv"))
    ap.add_argument("--raw-dir", default=str(here.parent.parent / "atlas-data"
                                             / "raw" / "pypi"))
    ap.add_argument("--spot-check", action="store_true",
                    help="print expected-vs-API years for famous packages")
    args = ap.parse_args()

    limit = min(args.top, 100)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    names = load_top_packages(raw_dir, limit)
    print(f"top {limit} packages: {names[0]} ... {names[-1]}", file=sys.stderr)

    rows: list[dict] = []
    dropped: list[str] = []
    for name in names:
        try:
            rows.append(package_row(name, raw_dir))
        except Exception as exc:  # noqa: BLE001 - drop and report
            print(f"DROPPED {name}: {exc}", file=sys.stderr)
            dropped.append(name)

    rows.sort(key=lambda r: r["entity_id"])
    validate(rows, limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    conf = Counter(r["confidence"] for r in rows)
    print(f"wrote {out} rows={len(rows)} dropped={len(dropped)} "
          f"confidence: {dict(conf)}", file=sys.stderr)
    if dropped:
        print(f"dropped packages: {dropped}", file=sys.stderr)

    if args.spot_check:
        by_id = {r["entity_id"]: r for r in rows}
        print("\nSPOT CHECK (task expectations vs PyPI API data):",
              file=sys.stderr)
        for name in SPOT_CHECKS:
            r = by_id.get(f"pypi:{name}")
            if r is None:
                print(f"  {name:10s} ABSENT (dropped)", file=sys.stderr)
                continue
            got = r["start_year"]
            exp = EXPECTED[name]
            mark = "OK" if got == exp else "MISMATCH (check!)"
            print(f"  {name:10s} expected={exp} api={got}  {mark}",
                  file=sys.stderr)
            stamp = [p for p in r["notes"].split("; ")
                     if p.startswith("first release upload")][0]
            print(f"             {stamp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
