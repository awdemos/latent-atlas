#!/usr/bin/env python3
"""Convert the Correlates of War (COW) State System Membership list into a
Latent Atlas curated CSV fixture (schema of fixtures/curated_romans_dprr.csv).

Source
------
Correlates of War Project, "State System Membership, v2024"
(released 2025; retrieved 2026-10-04).
Landing page : https://correlatesofwar.org/data-sets/state-system-membership/
Direct files : https://correlatesofwar.org/wp-content/uploads/States2024.zip
               https://correlatesofwar.org/wp-content/uploads/System2024.zip
               https://correlatesofwar.org/wp-content/uploads/State-System-Membership-Codebook-V2024.pdf

Required citation (COW terms of use; also embedded in each fixture row's notes):
  Correlates of War Project. 2025. "State System Membership, v2024."
  Online, https://correlatesofwar.org

Input file used: statelist2024.csv inside States2024.zip. One row per
state-system membership spell (a state that exits and re-enters the system
occupies multiple rows; 217 unique ccodes in 244 rows). Fields:
  stateabb, ccode, statenme, styear/stmonth/stday, endyear/endmonth/endday, version

Conversion rules
----------------
* entity_id = "cow:<ccode>" for the first spell of a ccode, "cow:<ccode>b",
  "cow:<ccode>c", ... for later spells of the same ccode (COW continuity:
  e.g. 255 Germany, 260 German Federal Republic and 265 German Democratic
  Republic are separate ccodes; 345 Yugoslavia -> Serbia shares ccode 345).
  Ids are unique per ROW, not per state name.
* start_year = styear. All years are 1816 CE or later, so astronomical
  numbering equals the source year (no BCE rows exist in this dataset).
* end_year   = EMPTY when endyear == 2024 (dataset's last year; the row is
  still a system member as of v2024). Otherwise endyear.
* relation = "exists" (entity_type = "polity"), confidence = "high".
* notes = full membership date range (with day precision), COW attribution,
  retrieval date, and a caveat flag where the row is a re-entry spell or a
  name change on a shared ccode.

Known COW conventions worth knowing (from the codebook; verified in data):
* The system list begins in 1816 (Congress of Vienna), so states independent
  before 1816 (e.g. USA 1776, Greece 1821 de facto) have start_year 1816 or
  their post-1816 (re)entry year. COW "Germany" therefore starts 1816, not
  1871; Korea 1887 reflects its entry, not independence.
* WWII-era exits/re-entries (France 1942/1944, Netherlands 1940/1945, ...)
  reflect loss/regain of sovereign control, not annexation into new states.
* Germany's post-war spells are coded as 260 German Federal Republic
  (1955-1990) and 265 German Democratic Republic (1954-1990), with 255
  Germany resuming 1990-10-03 at unification.

Usage:
  python3 scripts/cow2fixture.py                  # download + convert
  python3 scripts/cow2fixture.py --raw-dir PATH   # reuse existing downloads
"""

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "fixtures" / "curated_polities_cow.csv"
DEFAULT_TMP = Path("/tmp/latent_atlas_cow_raw")

CSV_URL = "https://correlatesofwar.org/wp-content/uploads/States2024.zip"
CITATION = ('COW citation: Correlates of War Project. 2025. '
            '"State System Membership, v2024." Online, '
            'https://correlatesofwar.org')
RETRIEVED = "2026-10-04"
DATASET_LAST_YEAR = 2024  # v2024 file: members still in system end 2024-12-31

FIELDS = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]

SUFFIXES = "abcdefghijklmnopqrstuvwxyz"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def load_statelist(raw_dir: Path):
    zip_path = raw_dir / "States2024.zip"
    if not zip_path.exists():
        download(CSV_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith("statelist2024.csv")]
        if not names:
            raise RuntimeError(f"statelist2024.csv not found in {zip_path}")
        with z.open(names[0]) as f:
            return list(csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")))


def convert(rows):
    out = []
    # COW file is already sorted by ccode then spell; sort defensively anyway.
    rows = sorted(rows, key=lambda r: (int(r["ccode"]), int(r["styear"]),
                                       int(r["stmonth"]), int(r["stday"])))
    spell_count = {}
    prev_by_ccode = {}
    for r in rows:
        ccode = int(r["ccode"])
        n = spell_count.get(ccode, 0)
        spell_count[ccode] = n + 1
        suffix = "" if n == 0 else SUFFIXES[min(n, len(SUFFIXES) - 1)]
        entity_id = f"cow:{ccode}{suffix}"

        start_year = int(r["styear"])
        endyear = int(r["endyear"])
        end_year = "" if endyear == DATASET_LAST_YEAR else str(endyear)

        start_date = f"{r['styear']}-{int(r['stmonth']):02d}-{int(r['stday']):02d}"
        if end_year:
            end_date = f"{endyear}-{int(r['endmonth']):02d}-{int(r['endday']):02d}"
            date_note = f"COW membership spell {start_date} to {end_date}"
        else:
            date_note = (f"COW membership spell {start_date} to "
                         f"{DATASET_LAST_YEAR}-12-31 (still a system member "
                         f"in v2024; end left open)")

        caveats = []
        prev = prev_by_ccode.get(ccode)
        if prev is not None:
            if prev["statenme"] == r["statenme"]:
                caveats.append("re-entry spell of a state that exited the "
                               "system and later returned (continuous "
                               "membership is NOT implied across the gap)")
            else:
                caveats.append(f"name change on shared COW ccode {ccode} "
                               f"(prior spell listed as "
                               f"\"{prev['statenme']}\")")
        prev_by_ccode[ccode] = r

        notes = "; ".join([date_note, CITATION, f"retrieved {RETRIEVED}"]
                          + caveats)

        out.append({
            "entity_id": entity_id,
            "entity_type": "polity",
            "canonical_name": r["statenme"],
            "aliases": r["stateabb"],
            "description": (f"Independent state in the Correlates of War "
                            f"interstate system (COW ccode {ccode}; "
                            f"abbreviation {r['stateabb']})"),
            "relation": "exists",
            "start_year": str(start_year),
            "end_year": end_year,
            "confidence": "high",
            "notes": notes,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_TMP,
                    help="gitignored dir for raw downloads "
                         f"(default: {DEFAULT_TMP})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output CSV (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    rows = load_statelist(args.raw_dir)
    fixture = convert(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(fixture)

    n_open = sum(1 for r in fixture if r["end_year"] == "")
    print(f"wrote {len(fixture)} rows ({n_open} open-ended) to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
