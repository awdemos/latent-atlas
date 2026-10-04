#!/usr/bin/env python3
"""Convert SNAP-DRGN ancient persons (http://snapdrgn.net) into a Latent Atlas
curated CSV fixture (schema of fixtures/curated_roman.csv).

Source
------
SNAP:DRGN (Standards for Networking Ancient Prosopography) publishes merged
person records at http://data.snapdrgn.net/person/<id>/ as HTML pages. Each
page renders, when known: name variants, publisher provenance (LGPN, PIR,
EDH-derived datasets...), and "Associated Date" ranges in the form
"0201/0250" (start/end year CE) or a single year "0298".

There is no working bulk download and no working SPARQL endpoint (the one
linked from the site, https://snap.dighum.kcl.ac.uk/query/, no longer
resolves; data.snapdrgn.net has no /sparql, and content negotiation on
person URIs returns HTML only — all verified 2026-10-04). This script
therefore scrapes the person HTML pages by id and saves the raw HTML
under a gitignored temp dir (.tmp-snapdrgn/) so the fixture can be
regenerated without re-hitting the server.

Two id ranges are scanned by default:
  11000-13400   Trismegistos/EDH-derived persons (Egyptian papyri and
                Roman epigraphy); century-grade attestation dates.
  671000-673936 VIAF/British-Museum famous-persons block; names like
                "Dio; Chrysostom" or "Tullia; ca76-ca45". NOTE: this
                block occasionally carries garbled date signs (e.g. a
                10th-c.-CE Richerus with associated date -0999); the raw
                date string is preserved in `notes` so bad rows can be
                filtered downstream.

Coverage caveat (spotty dates, per the task): many records carry only a
generic "0001/0300" attestation window (three centuries wide). A person is
kept only if its NARROWEST associated-date range spans at most --max-span
years (default 120, i.e. about one century or tighter); the century is then
derived from the range midpoint and expanded to a +/-50 year interval.

Date convention (Latent Atlas stores ASTRONOMICAL years): SNAP-DRGN
associated dates are zero-padded years; a leading '-' marks BCE
("-0225/-0175" = 225-175 BCE). Source BCE year Y maps to astronomical
1 - Y (-225 -> -224; 1 BCE would be -1 -> 0); CE years map identically.
Centuries are computed on the ASTRONOMICAL midpoint so the same code
handles BCE and CE: bucket s = floor((mid - 1) / 100) spans astronomical
years 100*s+1 .. 100*s+100 (s=2 -> 201..300 CE 3rd c.; s=-1 -> -99..0,
1st c. BCE); the emitted interval expands the bucket by +/-50 years.

Confidence:
  medium - narrowest attestation window spans <= 60 years (usable as a
           rough floruit)
  low    - century inferred from a broader window (> 60, <= 120 years)

Output schema (exactly fixtures/curated_roman.csv):
  entity_id,entity_type,canonical_name,aliases,description,relation,
  start_year,end_year,confidence,notes
  entity_id = snap:<id>            entity_type = person
  relation  = alive                (interval = "was plausibly alive then")

Licensing: no explicit license statement was found on snapdrgn.net or
data.snapdrgn.net (checked 2026-10-04); the underlying records remain the
property of their publisher datasets (LGPN (c) Lexicon of Greek Personal
Names, PIR, EDH...). Cite "SNAP:DRGN, http://snapdrgn.net" and verify terms
before any redistribution beyond this fixture.

Usage:
  python3 scripts/snapdrgn2fixture.py [--ranges 11000-13400,671000-673936]
      [--target 2000] [--cap 2000] [--workers 16] [--max-span 120]
      [--raw-dir .tmp-snapdrgn] [--out fixtures/curated_persons_snapdrgn.csv]
      [--spot-check] [--no-fetch]

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://data.snapdrgn.net/person/{}/"
RETRIEVED = "2026-10-04"
HEADER = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]

MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"}

# Known-bad source rows quarantined from the fixture (documented curation;
# the raw date string in the source conflicts with the person's own name):
#   snap:673229 "Richerus; 900-talet" ("900-talet" = Swedish for "the 900s",
#   i.e. 10th century CE) carries associated date -0999 (999 BCE) — a sign
#   error in the upstream VIAF/British-Museum data, preserved in the cached
#   raw page but not propagated into the curated CSV.
EXCLUDE_IDS = {673229}


# ---------------------------------------------------------------------------
# Fetching

def fetch_person(pid: int, raw_dir: Path, timeout: float = 30.0) -> str | None:
    """Download one person page into raw_dir (skipped if already cached).
    Returns 'ok', '404', or 'error'."""
    dest = raw_dir / f"person_{pid}.html"
    if dest.exists():
        return "ok"
    url = BASE.format(pid)
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "latent-atlas-fixture/1.0 "
                                           "(research; contact: repo owner)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            dest.write_bytes(body)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
        except Exception:
            continue
    return "error"


# ---------------------------------------------------------------------------
# Parsing

LI_RE = re.compile(r"<li>\s*(.*?)\s*</li>", re.S)
DATE_TOKEN_RE = re.compile(r"^\s*-?\d{3,4}(?:\s*/\s*-?\d{3,4})?\s*$")


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def section(html: str, marker: str, end_markers: tuple[str, ...]) -> str:
    """Text between `marker` and the first of `end_markers` after it."""
    i = html.find(marker)
    if i < 0:
        return ""
    j = len(html)
    for m in end_markers:
        k = html.find(m, i + len(marker))
        if 0 <= k < j:
            j = k
    return html[i + len(marker):j]


def parse_person_page(html: str) -> dict | None:
    """Extract names, associated-date ranges, publishers. None if not a
    person page."""
    if "<b>SNAP ID</b>" not in html:
        return None
    # Names: first <ul> after the <h3>Name</h3>.
    name_block = section(html, "<h3>Name</h3>", ("<h3>", "<h4>"))
    names = []
    for li in LI_RE.findall(name_block):
        t = _strip_tags(li)
        if t and not t.startswith("http"):
            names.append(t)
    # Associated dates: <ul> after <h4>Associated Date</h4>.
    date_block = section(html, "<h4>Associated Date</h4>", ("<h4>", "<h3>"))
    date_ranges = []
    for li in LI_RE.findall(date_block):
        t = _strip_tags(li)
        if DATE_TOKEN_RE.match(t):
            date_ranges.append(t.replace(" ", ""))
    # Publishers.
    pub_block = section(html, "<b>Publisher(s)</b>", ("</table>",))
    pubs = [_strip_tags(li) for li in LI_RE.findall(pub_block)]
    return {"names": names, "date_ranges": date_ranges,
            "publishers": [p for p in pubs if p]}


PUB_LABEL = {
    "lgpn": "LGPN", "lexicon of greek personal names": "LGPN",
    "paregorios": "PIR", "pir": "PIR",
    "edh": "EDH", "epigraphic": "EDH", "heidelberg": "EDH",
    "trismegistos": "Trismegistos",
}


def pub_label(pub: str) -> str:
    low = pub.lower()
    for key, label in PUB_LABEL.items():
        if key in low:
            return label
    host = re.sub(r"^https?://", "", low).split("/")[0]
    return host or pub


# ---------------------------------------------------------------------------
# Fixture assembly


def century_of(mid_astro: float) -> int:
    """Astronomical-year midpoint -> century bucket s, where bucket s spans
    astro years 100*s+1 .. 100*s+100 (s=2 -> 201..300 = 3rd c. CE;
    s=-1 -> -99..0 = 1st c. BCE)."""
    return int((mid_astro - 1) // 100)


def century_interval(s: int) -> tuple[int, int]:
    """Century bucket expanded by +/-50 years (astronomical years).
    s=2 (3rd c. CE) -> 151..350; s=-1 (1st c. BCE) -> -149..50."""
    return 100 * s + 1 - 50, 100 * s + 100 + 50


def bucket_label(s: int) -> str:
    if s >= 0:
        return f"{ordinal(s + 1)} century CE"
    return f"{ordinal(-s)} century BCE"


def ordinal(c: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(c, f"{c}th")


def no_commas(s: str) -> str:
    return s.replace(",", ";").replace("\n", " ").strip()


def build_record(pid: int, parsed: dict, max_span: int) -> tuple[dict | None, str]:
    """Returns (row, drop_reason). Exactly one of the two is not None."""
    if not parsed["names"]:
        return None, "no_name"
    if not parsed["date_ranges"]:
        return None, "no_dates"

    # Narrowest valid range (astro years) wins.
    def to_astro(src_year: int) -> int:
        # Source: negative = BCE ("-0225" -> astro -224), positive = CE.
        return src_year + 1 if src_year < 0 else src_year

    best: tuple[int, int, str] | None = None  # (astro_start, astro_end, raw)
    for raw in parsed["date_ranges"]:
        parts = raw.split("/")
        try:
            a1 = to_astro(int(parts[0]))
            a2 = to_astro(int(parts[-1]))
        except ValueError:
            continue
        if a1 > a2 or not (-2000 <= a1 <= 3000 and -2000 <= a2 <= 3000):
            return None, f"odd_range:{raw}"
        if best is None or (a2 - a1) < (best[1] - best[0]):
            best = (a1, a2, raw)
    if best is None:
        return None, "no_dates"
    span = best[1] - best[0]
    if span > max_span:
        return None, f"broad_range:{best[2]}"

    s = century_of((best[0] + best[1]) / 2.0)
    start, end = century_interval(s)
    confidence = "medium" if span <= 60 else "low"

    canonical = parsed["names"][0]
    aliases = [n for n in parsed["names"][1:] if n != canonical][:5]
    labels = []
    for p in parsed["publishers"]:
        lab = pub_label(p)
        if lab not in labels:
            labels.append(lab)

    description = "SNAP-DRGN merged ancient person"
    if labels:
        description += "; sources: " + "; ".join(labels)
    notes = (
        f"SNAP-DRGN person {pid} (http://data.snapdrgn.net/person/{pid}/); "
        f"associated attestation dates: {';'.join(parsed['date_ranges'])}; "
        f"narrowest range {best[2]} ({span}y) -> {bucket_label(s)} "
        f"expanded +/-50y; confidence "
        f"{'medium (<=60y window; rough floruit)' if confidence == 'medium' else 'low (century inferred from attestation window)'}; "
        f"source SNAP-DRGN data.snapdrgn.net retrieved {RETRIEVED}"
    )
    return {
        "entity_id": f"snap:{pid}",
        "entity_type": "person",
        "canonical_name": no_commas(canonical),
        "aliases": "|".join(no_commas(a) for a in aliases),
        "description": no_commas(description),
        "relation": "alive",
        "start_year": start,
        "end_year": end,
        "confidence": confidence,
        "notes": no_commas(notes),
    }, ""


# ---------------------------------------------------------------------------
# Validation


def validate(rows: list[dict]) -> None:
    entity_types = {"person", "event", "polity", "organization", "work",
                    "technology"}
    relations = {"alive", "ongoing", "exists", "active", "available"}
    confs = {"high", "medium", "low"}
    assert [c for c in HEADER], "header constant empty"
    ids = set()
    for r in rows:
        assert r["entity_type"] in entity_types, r
        assert r["relation"] in relations, r
        assert r["confidence"] in confs, r
        for y in (r["start_year"], r["end_year"]):
            assert isinstance(y, int), (y, r)
        assert r["start_year"] <= r["end_year"], r
        assert r["entity_id"] not in ids, r["entity_id"]
        ids.add(r["entity_id"])
        assert r["entity_id"].startswith("snap:"), r
        assert r["canonical_name"], r


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    root = here.parent.parent
    ap.add_argument("--ranges", default="11000-13400,671000-673936",
                    help="comma-separated inclusive id ranges to scan; the "
                         "default covers the TM/EDH block with century-grade "
                         "dates (ids < ~11000 are an EDH-derived block whose "
                         "only associated date is the generic 0001/0300 "
                         "window, verified on a random sample) plus the "
                         "VIAF/British-Museum famous-persons block at "
                         "671000-673936")
    ap.add_argument("--target", type=int, default=2000,
                    help="stop fetching once this many rows are derivable")
    ap.add_argument("--cap", type=int, default=2000,
                    help="hard cap on output rows")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-span", type=int, default=120)
    ap.add_argument("--raw-dir", default=str(root / ".tmp-snapdrgn"))
    ap.add_argument("--out", default=str(root / "fixtures"
                                         / "curated_persons_snapdrgn.csv"))
    ap.add_argument("--spot-check", action="store_true",
                    help="print candidate famous-person rows to stderr")
    ap.add_argument("--no-fetch", action="store_true",
                    help="only parse already-cached raw files")
    args = ap.parse_args()

    ranges: list[tuple[int, int]] = []
    for part in args.ranges.split(","):
        lo, hi = part.split("-")
        ranges.append((int(lo), int(hi)))

    def in_range(pid: int) -> bool:
        return any(lo <= pid <= hi for lo, hi in ranges)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- fetch one range at a time (cache-first) until target is met -------
    drop_counts: dict[str, int] = {}
    kept: list[dict] = []
    scanned = 0
    chunk = args.workers * 8
    for lo, hi in ranges:
        i = lo
        while i <= hi:
            batch = list(range(i, min(i + chunk, hi + 1)))
            i += len(batch)
            if not args.no_fetch:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    results = list(ex.map(lambda p: fetch_person(p, raw_dir),
                                          batch))
                nerr = sum(1 for r in results if r == "error")
                if nerr:
                    print(f"  ids {batch[0]}..{batch[-1]}: {nerr} fetch "
                          f"errors", file=sys.stderr)
            scanned += len(batch)
        # (Re)parse everything cached so far and see if target is met.
        kept, drop_counts = parse_all(raw_dir, in_range, args.max_span)
        print(f"  range {lo}-{hi} done; scanned {scanned} -> kept "
              f"{len(kept)}", file=sys.stderr)

    # --- select final rows --------------------------------------------------
    # Identifiable famous persons (the VIAF/British-Museum block, ids
    # >= 671000) get priority even where their floruit window is wider than
    # a papyrological single-year attestation; within each group, narrowest
    # window first, then lowest id.
    def prio(row: dict) -> tuple[int, int, int]:
        pid = int(row["entity_id"].split(":")[1])
        span = int(row["end_year"]) - int(row["start_year"])
        return (0 if pid >= 671000 else 1, span, pid)

    kept.sort(key=prio)
    rows = kept[:args.cap]
    rows.sort(key=lambda r: int(r["entity_id"].split(":")[1]))

    validate(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    conf = Counter(r["confidence"] for r in rows)
    pubs = Counter()
    for r in rows:
        for lab in re.findall(r"sources: (.*?)$", r["description"]):
            pubs.update(lab.split("; "))
    print(f"scanned_ids={scanned} kept_in_window="
          f"{len(kept)} rows_written={len(rows)}")
    print(f"confidence: high={conf['high']} medium={conf['medium']} "
          f"low={conf['low']}")
    print(f"drop_reasons: {dict(sorted(drop_counts.items()))}")
    print(f"wrote {out}")
    print(f"validation: header exact, enums valid, years int, "
          f"ids unique ({len({r['entity_id'] for r in rows})}), "
          f"start<=end all rows: OK")

    if args.spot_check:
        print("\nSPOT-CHECK CANDIDATES (search Wikipedia for these):",
              file=sys.stderr)
        pats = ["TRAIAN", "HADRIAN", "ANTONIN", "NERO", "TIBERIVS",
                "CLAUDIVS", "VESPASIAN", "MARCVR", "AVRELIVS", "COMMOD",
                "SEPTIMIVS", "GALLIEN", "CONSTANTIN", "IVLIAN", "CAESAR",
                "CICERO", "VERGIL", "HORAT", "OVIDIVS", "LIVIVS", "TACIT",
                "PLINIVS", "SVETON", "SENECA", "MVRET",
                # VIAF-block famous persons (names may contain ';' where the
                # source had a comma):
                "DIO", "TERTULLIAN", "ZENO", "TULLIA", "KALIXT", "CALLIXT",
                "OLYMPIODOR", "APOLLON", "THEODERICH", "HIEROCLES",
                "BERNARD", "RICHIER", "RICHAR", "GAUDENTI", "ASTERIVS",
                "HERMOGENES", "JOHANNES", "IOHANNES", "ZENON"]
        shown = 0
        for r in rows:
            up = r["canonical_name"].upper()
            if any(p in up for p in pats):
                print(f"  {r['entity_id']:14s} {r['canonical_name'][:36]:36s} "
                      f"[{r['start_year']},{r['end_year']}] "
                      f"{r['confidence']:6s} "
                      f"{re.sub(',', ';', r['notes'])[:110]}", file=sys.stderr)
                shown += 1
                if shown >= 40:
                    break
    return 0


def parse_all(raw_dir: Path, in_range, max_span: int) -> tuple[list[dict], dict[str, int]]:
    from collections import Counter
    kept: list[dict] = []
    drops: Counter = Counter()
    for path in sorted(raw_dir.glob("person_*.html"),
                       key=lambda p: int(p.stem.split("_")[1])):
        pid = int(path.stem.split("_")[1])
        if not in_range(pid):
            continue
        if pid in EXCLUDE_IDS:
            drops["excluded_known_bad"] += 1
            continue
        parsed = parse_person_page(path.read_text(errors="replace"))
        if parsed is None:
            drops["not_a_person_page"] += 1
            continue
        row, why = build_record(pid, parsed, max_span)
        if row is None:
            key = why.split(":")[0]
            drops[key] += 1
        else:
            kept.append(row)
    return kept, dict(drops)


if __name__ == "__main__":
    raise SystemExit(main())
