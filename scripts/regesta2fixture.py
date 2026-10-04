#!/usr/bin/env python3
"""Convert Regesta Imperii regesta CSV exports into a Latent Atlas curated
CSV fixture (schema of fixtures/curated_romans_dprr.csv).

Source
------
Regesta Imperii (RI) — the Academy project recording the documented acts of
Roman-German kings and emperors (ca. 500/750-1500). Raw data: the official
(non-current) CSV exports published at

  https://gitlab.rlp.net/adwmainz/regesta-imperii/lab/regesta-imperii-data
  (directory data/regesta-csv, branch main)

One row per regest (abstract of an imperial charter/document). Columns used:
  identifier            e.g. "[RI VII] H. 1 n. 2"   -> canonical_name
  title                 issuing ruler               -> notes
  locality_string       place of issue              -> notes
  start_date/end_date   "YYYY-MM-DD"; equal when the date is exact
  summary               regest text (HTML)          -> description
  date_string           human date                  -> notes
  persistent_identifier stable id, also the key of
                        https://www.regesta-imperii.de/id/<pid>

The www.regesta-imperii.de site itself sits behind an Anubis anti-bot wall
and offers no bulk JSON; the REST interface (/tei/<collection>/sources/<id>)
serves the same records one-by-one as TEI/CEI XML. The GitLab CSV export is
the project's own bulk-download channel, so it is used here. Every row keeps
its https://www.regesta-imperii.de/id/<pid> link in notes for provenance.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Required
attribution per the TEI header: name of the creator and a link to the
material, i.e. cite "Regesta Imperii, Akademie der Wissenschaften und der
Literatur | Mainz" plus the per-record URL. The attribution string is
embedded in every row's notes field.

Year convention
---------------
All RI dates are CE and well inside the 500-1500 window kept here, so the
astronomical conversion is the identity (year N CE -> N).

Fixture mapping
---------------
  entity_id     ri:<persistent_identifier, lowercased> (unique; verified)
  entity_type   work (a regest/charter is a documentary work)
  relation      available
  start_year    year of start_date (first possible issue date)
  end_year      "" (open bound by fixture spec; the source's own end_date —
                latest possible issue date — is carried inside notes instead)
  confidence    high when start_date == end_date (exact day), medium when the
                source gives a from/to range
  notes         issuing ruler; place; exact date or range; source name;
                retrieval date; CC BY 4.0 attribution + per-record URL

Sampling: deterministic, reproducible (no RNG). A fixed per-department quota
(floor + remainder proportional to file size) is filled with evenly spaced
picks over the file's rows sorted by persistent_identifier, so the fixture
spans all published departments (RI I-XIV) and the whole 500-1500 range.
Raw downloads are cached under .tmp-regesta/ (gitignored).
"""

import argparse
import csv
import html
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

GITLAB_RAW = ("https://gitlab.rlp.net/adwmainz/regesta-imperii/lab/"
              "regesta-imperii-data/-/raw/main/data/regesta-csv")

# RI departments published as CSV. "alles" variants are used where the
# department is split into partial exports, to avoid double-counting.
DEPARTMENTS = [
    ("RI_01alles.csv", "RI I"),
    ("RI_02.csv", "RI II"),
    ("RI_03.csv", "RI III"),
    ("RI_04.csv", "RI IV"),
    ("RI_05.csv", "RI V"),
    ("RI_06.csv", "RI VI"),
    ("RI_07.csv", "RI VII"),
    ("RI_08.csv", "RI VIII"),
    ("RI_11.csv", "RI XI"),
    ("RI_12.csv", "RI XII"),
    ("RI_13alles.csv", "RI XIII"),
    ("RI_14.csv", "RI XIV"),
]

FLOOR_PER_DEPT = 50
# Famous regesta force-included in the sample (by persistent_identifier) so
# the fixture always contains well-documented anchor events for spot-checks:
#   Heinrich IV's absolution at Canossa (Walk to Canossa, 1077-01-28)
#   Friedrich II's Mainz Reichslandfriede court (1235-08-15)
#   Karl IV proclaiming the first 23 chapters of the Golden Bull at
#   Nuremberg (1356-01-10)
FAMOUS = {
    "RI_03.csv": "a531852d-f2a3-41bc-8e10-49e42706453e",
    "RI_05.csv": "1235-08-15_1_0_5_1_1_3023_2099c",
    "RI_08.csv": "1356-01-10_1_0_8_0_0_2700_2397",
}
YEAR_MIN, YEAR_MAX = 500, 1500  # the fixture fills the 500-1500 CE gap
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MAX_DESC = 400
RETRIEVED = "2026-10-04"
ATTRIBUTION = ("Regesta Imperii, Akad. der Wissenschaften und der Literatur "
               "| Mainz, CC BY 4.0")

HEADER = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(HTML_TAG_RE.sub("", s or ""))).strip()


def download(raw_dir: Path, fname: str) -> Path:
    dest = raw_dir / fname
    url = f"{GITLAB_RAW}/{fname}"
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "latent-atlas-fixture/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    return dest


def load_rows(path: Path):
    """Yield valid regesta (dicts) from one department CSV, sorted by pid."""
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            sd = (row.get("start_date") or "").strip()
            ed = (row.get("end_date") or "").strip()
            pid = (row.get("persistent_identifier") or "").strip()
            m = DATE_RE.match(sd)
            if not m or not ed:
                continue  # malformed/empty dates (3 rows in the export)
            year = int(m.group(1))
            if not (YEAR_MIN <= year <= YEAR_MAX):
                continue  # outside the 500-1500 window (e.g. RI XIV > 1500)
            if not pid:
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["persistent_identifier"])
    return rows


def even_sample(items, n: int):
    """Deterministic evenly spaced sample of n items (all if len <= n)."""
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def build_row(row, dept_label):
    pid = row["persistent_identifier"].strip()
    sd = row["start_date"].strip()
    ed = row["end_date"].strip()
    exact = sd == ed
    year = int(sd[:4])
    ruler = strip_html(row.get("title", ""))
    place = strip_html(row.get("locality_string", ""))
    desc = strip_html(row.get("summary", ""))
    if len(desc) > MAX_DESC:
        desc = desc[:MAX_DESC].rstrip() + " …"

    notes = [f"dept: {dept_label}"]
    if ruler:
        notes.append(f"issuer: {ruler}")
    if place:
        notes.append(f"place: {place}")
    notes.append(f"date: {sd}" if exact else f"date range: {sd} to {ed} (year = first possible)")
    notes.append(f"source: {ATTRIBUTION}")
    notes.append(f"retrieved {RETRIEVED}")
    notes.append(f"https://www.regesta-imperii.de/id/{pid}")

    return {
        "entity_id": f"ri:{pid.lower()}",
        "entity_type": "work",
        "canonical_name": (row.get("identifier") or "").strip(),
        "aliases": "",
        "description": desc,
        "relation": "available",
        "start_year": year,
        "end_year": "",
        "confidence": "high" if exact else "medium",
        "notes": "; ".join(notes),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    ap.add_argument("--raw-dir", default=str(here.parent.parent / ".tmp-regesta"),
                    help="cache dir for the downloaded RI CSV exports "
                         "(gitignored; files are re-downloaded only if missing)")
    ap.add_argument("--target", type=int, default=1500,
                    help="total number of regesta to sample (default 1500)")
    ap.add_argument("--out", default=str(here.parent.parent / "fixtures"
                                         / "curated_documents_regesta.csv"))
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    counts = {}
    for fname, label in DEPARTMENTS:
        p = raw_dir / fname
        if not p.exists():
            download(raw_dir, fname)
        rows = load_rows(p)
        counts[fname] = len(rows)
        paths[fname] = rows

    # Quota: FLOOR per department, remainder proportional to row counts.
    n_dept = len(DEPARTMENTS)
    total_avail = sum(counts[f] for f, _ in DEPARTMENTS)
    budget = max(args.target - FLOOR_PER_DEPT * n_dept, 0)
    quotas = {}
    assigned = 0
    for fname, _ in DEPARTMENTS:
        share = FLOOR_PER_DEPT + (budget * counts[fname]) // total_avail
        quotas[fname] = min(share, counts[fname])
        assigned += quotas[fname]
    # Hand out rounding leftovers to the largest departments.
    leftovers = args.target - assigned
    for fname, _ in sorted(DEPARTMENTS, key=lambda d: -counts[d[0]]):
        while leftovers > 0 and quotas[fname] < counts[fname]:
            quotas[fname] += 1
            leftovers -= 1

    out_rows, seen_ids, dropped_dup = [], set(), 0
    per_dept = Counter()
    for fname, label in DEPARTMENTS:
        picked = even_sample(paths[fname], quotas[fname])
        famous_pid = FAMOUS.get(fname)
        if famous_pid and all(r["persistent_identifier"] != famous_pid
                              for r in picked):
            by_pid = {r["persistent_identifier"]: r for r in paths[fname]}
            if famous_pid in by_pid:
                picked[-1] = by_pid[famous_pid]  # swap out the last pick
        for row in picked:
            rec = build_row(row, label)
            if rec["entity_id"] in seen_ids:
                dropped_dup += 1  # one pid is duplicated inside RI_06
                continue
            seen_ids.add(rec["entity_id"])
            out_rows.append(rec)
            per_dept[label] += 1
    out_rows.sort(key=lambda r: r["entity_id"])

    # Self-validation: exact header, enums, unique ids, int-or-empty years.
    assert len(out_rows) == len(seen_ids), "duplicate entity_ids"
    for r in out_rows:
        assert r["entity_type"] == "work" and r["relation"] == "available"
        assert r["confidence"] in ("high", "medium", "low"), r
        assert isinstance(r["start_year"], int), r
        assert r["end_year"] == "" and r["aliases"] == "", r
        assert r["canonical_name"] and r["entity_id"].startswith("ri:"), r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(out_rows)

    conf = Counter(r["confidence"] for r in out_rows)
    years = [r["start_year"] for r in out_rows]
    print(f"rows={len(out_rows)} (target {args.target}) "
          f"dropped_duplicate_pids={dropped_dup}", file=sys.stderr)
    print(f"year range: {min(years)}..{max(years)}", file=sys.stderr)
    print(f"confidence: high={conf['high']} medium={conf['medium']}", file=sys.stderr)
    print("per department: " + ", ".join(f"{k}={per_dept[k]}"
                                         for _, k in DEPARTMENTS), file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
