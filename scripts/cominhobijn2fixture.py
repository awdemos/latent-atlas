#!/usr/bin/env python3
"""Convert the Comin & Hobijn CHAT cross-country technology-adoption dataset
into a Latent Atlas curated CSV fixture.

Source: Cross-country Historical Adoption of Technology (CHAT), Diego Comin
(NYU) and Bart Hobijn (FRB San Francisco / FRBNY), NBER Working Paper 15319
"The CHAT Dataset" (2009). CHAT covers the usage/adoption of ~115
technologies in 160+ countries since 1800. The canonical NBER download page
http://www.nber.org/data/chat/ now redirects to the HCCTA dataset page
(https://www.nber.org/research/data/historical-cross-country-technology-
adoption-hccta-dataset), which hosts the older 2003 Excel panel
(hcctad.xls, 23 countries, 80 variables). The full 2009 CHAT panel is
redistributed as a single CSV by the DataHub project:
  https://datahub.io/technology/historical-adoption-of-technology
  CSV: https://datahub.io/technology/historical-adoption-of-technology/_r/-/data/chat.csv
That mirror is what this script downloads (verified consistent with the NBER
hcctad.xls panel for shared variables: Telegraph first nonzero 1850 Germany,
Telephones 1876 United States, Televisions 1946 United States, Railways
1825 United Kingdom).

Data semantics (see hcctadhelp.pdf on the NBER page): each cell is a usage
level (counts, tonnage, kWh, line km, ...). The panel is built from sparse
historical benchmark observations that are interpolated between benchmark
years. "Year of adoption" is therefore approximated here as the FIRST YEAR
WITH A NON-ZERO RECORDED USAGE LEVEL (restricted to >= 1800, CHAT's stated
scope). This is a recorded-usage proxy, not a legal/commercial adoption
date: it is bounded by data coverage and can lag the true first use
(e.g. cellular phones show 1980 Finland, while the first commercial
cellular network launched in Japan in 1979; the dataset simply has no
Japanese cellphone observations before 1981).

Non-technology columns (population, real GDP, school enrollment, literacy,
investment shares) are excluded. "Venezuala" is a source typo for
Venezuela (ISO VE); "Russia" in the source denotes the USSR/Russia series.

Rows:
  PRIMARY   - one row per technology: first year with nonzero usage anywhere
              in the sample. entity_id tech:<code>, relation available,
              open end.
  SECONDARY - one row per (technology, country): first year with nonzero
              usage in that country. entity_id tech:<code>-<isocode>.
              Well-known technologies (PRIORITY) are included in full; the
              remainder fill the row budget in (tech, country) order.

Attribution: Comin & Hobijn (2009), "The CHAT Dataset", NBER Working Paper
15319. Cite this paper when using CHAT-derived data.

Usage:
  python3 scripts/cominhobijn2fixture.py [--out fixtures/curated_tech_adoption_cominhobijn.csv]
                                         [--cap 3000] [--spot-check]

Stdlib only. Raw downloads go to <repo>/.tmp-cominhobijn/ (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import Counter
from pathlib import Path

CHAT_CSV_URL = ("https://datahub.io/technology/historical-adoption-of-"
                "technology/_r/-/data/chat.csv")

RETRIEVED = "2026-10-04"

# Columns that are macro/demographic context, not technologies.
NON_TECH = {"xlpopulation", "xlrealgdp", "pctivprimeenroll", "pctivsecenroll",
            "pctivprivateinv", "pctivpublicinv", "pctivliteracy"}

MIN_YEAR = 1800  # CHAT documented scope: "since 1800"

# Well-known technologies whose per-country rows are all kept (spot-check
# favourites), before the remaining (tech, country) pairs fill the budget.
PRIORITY = ["railline", "telegram", "telephone", "radio", "tv", "cellphone",
            "computer", "internetuser", "elecprod", "vehicle_car",
            "vehicle_com", "atm", "mail", "newspaper", "ship_steam",
            "ship_motor", "ag_tractor", "spindle_ring", "steel_bof",
            "steel_eaf", "cabletv", "cheque", "creditdebit", "eft", "pos"]

# Human-readable names for well-known codes; anything else falls back to the
# raw code with underscores turned into spaces.
CANONICAL = {
    "railline": "Railways (line length open)",
    "telegram": "Telegraph",
    "telephone": "Telephone",
    "radio": "Radio receivers",
    "tv": "Television",
    "cellphone": "Mobile/cellular phones",
    "computer": "Personal computers",
    "internetuser": "Internet users",
    "elecprod": "Electricity production",
    "vehicle_car": "Passenger cars",
    "vehicle_com": "Commercial vehicles",
    "atm": "Automated teller machines",
    "mail": "Mail volume",
    "newspaper": "Newspaper circulation",
    "ship_sail": "Sailing ships",
    "ship_steam": "Steamships",
    "ship_motor": "Motor ships",
    "ship_all": "Ships (all)",
    "ag_tractor": "Agricultural tractors",
    "ag_harvester": "Mechanical harvesters",
    "ag_milkingmachine": "Milking machines",
    "spindle_mule": "Mule spindles (textiles)",
    "spindle_ring": "Ring spindles (textiles)",
    "loom_auto": "Automatic looms",
    "loom_total": "Looms (all)",
    "steel_acidbess": "Acid Bessemer steel",
    "steel_basicbess": "Basic Bessemer steel",
    "steel_bof": "Basic oxygen steel",
    "steel_eaf": "Electric arc steel",
    "steel_ohf": "Open hearth steel",
    "steel_stainless": "Stainless steel",
    "steel_other": "Other steel processes",
    "cabletv": "Cable television",
    "cheque": "Cheques",
    "creditdebit": "Credit/debit cards",
    "eft": "Electronic funds transfer",
    "pos": "Point-of-sale terminals",
    "aviationpkm": "Aviation (passenger-km)",
    "aviationtkm": "Aviation (tonne-km)",
    "railpkm": "Railways (passenger-km)",
    "railtkm": "Railways (tonne-km)",
    "railp": "Rail passengers",
    "railt": "Rail freight",
    "newspaper": "Newspaper circulation",
    "fert_total": "Fertilizer use",
    "pest_total": "Pesticide use",
    "irrigatedarea": "Irrigated area",
}

# CHAT country_name -> ISO 3166-1 alpha-2. Custom codes for historical or
# non-ISO entities: IC Indochina, NVN North Vietnam, SVN South Vietnam,
# SYE South Yemen (avoids colliding with Syria SY / Vietnam VN / Yemen YE).
COUNTRY_ISO = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Angola": "AO",
    "Argentina": "AR", "Armenia": "AM", "Australia": "AU", "Austria": "AT",
    "Azerbaijan": "AZ", "Bangladesh": "BD", "Belarus": "BY", "Belgium": "BE",
    "Belize": "BZ", "Benin": "BJ", "Bolivia": "BO",
    "Bosnia-Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR",
    "Bulgaria": "BG", "Burkina Faso": "BF", "Burma": "MM", "Burundi": "BI",
    "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA",
    "Central African Republic": "CF", "Chad": "TD", "Chile": "CL", "China": "CN",
    "Colombia": "CO", "Costa Rica": "CR", "Croatia": "HR", "Cuba": "CU",
    "Czech Republic": "CZ", "Czechoslovakia": "CS",
    "Democratic Republic of the Congo": "CD", "Denmark": "DK",
    "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "Equatorial Guinea": "GQ", "Eritrea": "ER",
    "Estonia": "EE", "Ethiopia": "ET", "Finland": "FI", "France": "FR",
    "French Guiana": "GF", "Gabon": "GA", "Gambia": "GM", "Georgia": "GE",
    "Germany": "DE", "Ghana": "GH", "Greece": "GR", "Guatemala": "GT",
    "Guinea": "GN", "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT",
    "Honduras": "HN", "Hong Kong": "HK", "Hungary": "HU", "Iceland": "IS",
    "India": "IN", "Indochina": "IC", "Indonesia": "ID", "Iran": "IR",
    "Iraq": "IQ", "Ireland": "IE", "Israel": "IL", "Italy": "IT",
    "Ivory Coast": "CI", "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ",
    "Kenya": "KE", "Kuwait": "KW", "Kyrgyzstan": "KG", "Laos": "LA",
    "Latvia": "LV", "Lebanon": "LB", "Lesotho": "LS", "Liberia": "LR",
    "Libya": "LY", "Lithuania": "LT", "Luxembourg": "LU", "Macedonia": "MK",
    "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY", "Mali": "ML",
    "Mauritania": "MR", "Mauritius": "MU", "Mexico": "MX", "Moldova": "MD",
    "Mongolia": "MN", "Montenegro": "ME", "Morocco": "MA", "Mozambique": "MZ",
    "Namibia": "NA", "Nepal": "NP", "Netherlands": "NL", "New Zealand": "NZ",
    "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG", "North Vietnam": "NVN",
    "Norway": "NO", "Oman": "OM", "Pakistan": "PK", "Panama": "PA",
    "Papua New Guinea": "PG", "Paraguay": "PY", "Peru": "PE",
    "Philippines": "PH", "Poland": "PL", "Portugal": "PT",
    "Republic of the Congo": "CG", "Romania": "RO", "Russia": "RU",
    "Rwanda": "RW", "Saudi Arabia": "SA", "Senegal": "SN", "Serbia": "RS",
    "Sierra Leone": "SL", "Singapore": "SG", "Slovak Republic": "SK",
    "Slovenia": "SI", "Somalia": "SO", "South Africa": "ZA", "South Korea": "KR",
    "South Vietnam": "SVN", "South Yemen": "SYE", "Spain": "ES",
    "Sri Lanka": "LK", "Sudan": "SD", "Suriname": "SR", "Swaziland": "SZ",
    "Sweden": "SE", "Switzerland": "CH", "Syria": "SY", "Taiwan": "TW",
    "Tajikistan": "TJ", "Tanzania": "TZ", "Thailand": "TH", "Togo": "TG",
    "Tunisia": "TN", "Turkey": "TR", "Turkmenistan": "TM", "Uganda": "UG",
    "Ukraine": "UA", "United Arab Emirates": "AE", "United Kingdom": "GB",
    "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Venezuala": "VE",  # source typo for Venezuela
    "Vietnam": "VN", "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW",
}

SOURCE_NOTE = ("Source: CHAT dataset (Comin & Hobijn; NBER Working Paper "
               "15319; cite this paper when reusing) via DataHub CSV mirror; "
               f"retrieved {RETRIEVED}")
PRIMARY_NOTE = (SOURCE_NOTE + "; start = first year with nonzero recorded "
                "usage anywhere in the sample (benchmark data interpolated "
                "between benchmarks; proxy for adoption year)")
SECONDARY_NOTE = (SOURCE_NOTE + "; start = first year with nonzero recorded "
                  "usage in this country (interpolated benchmark data; "
                  "proxy for adoption year)")


def fetch_raw(raw_dir: Path) -> Path:
    """Download the CHAT CSV into the gitignored raw dir (cached)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "chat.csv"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"using cached {dest}", file=sys.stderr)
        return dest
    print(f"downloading {CHAT_CSV_URL}", file=sys.stderr)
    with urllib.request.urlopen(CHAT_CSV_URL, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print(f"saved {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
    return dest


def load_first_years(path: Path):
    """Return (techs, first_any, first_country) where
    first_any[tech] = (year, country, value) and
    first_country[(tech, country)] = (year, value)."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        techs = [c for c in reader.fieldnames
                 if c not in ("country_name", "year") and c not in NON_TECH]
        first_any: dict[str, tuple[int, str, str]] = {}
        first_country: dict[tuple[str, str], tuple[int, str]] = {}
        for r in reader:
            year = int(r["year"])
            if year < MIN_YEAR:
                continue
            country = r["country_name"]
            for t in techs:
                v = r[t].strip()
                if not v or float(v) == 0:
                    continue
                if t not in first_any or year < first_any[t][0]:
                    first_any[t] = (year, country, v)
                k = (t, country)
                if k not in first_country or year < first_country[k][0]:
                    first_country[k] = (year, v)
    return techs, first_any, first_country


def canonical_name(code: str) -> str:
    return CANONICAL.get(code, code.replace("_", " "))


def no_commas(s: str) -> str:
    return s.replace(",", ";").replace("\n", " ").strip()


def build_rows(techs, first_any, first_country, cap: int):
    rows = []
    # PRIMARY: one row per technology.
    for t in techs:
        year, country, _ = first_any[t]
        rows.append({
            "entity_id": f"tech:{t}",
            "entity_type": "technology",
            "canonical_name": no_commas(canonical_name(t)),
            "aliases": "",
            "description": no_commas(
                f"Technology adoption (CHAT variable {t}); first recorded "
                f"nonzero usage anywhere in the sample: {year} in {country}"),
            "relation": "available",
            "start_year": year,
            "end_year": "",
            "confidence": "medium",
            "notes": PRIMARY_NOTE,
        })
    # SECONDARY: per (technology, country), priority techs first.
    budget = cap - len(rows)
    pairs = sorted(first_country.items(),
                   key=lambda kv: (0 if kv[0][0] in PRIORITY else 1,
                                   kv[0][0], kv[0][1]))
    for (t, country), (year, _v) in pairs[:budget]:
        iso = COUNTRY_ISO[country]
        rows.append({
            "entity_id": f"tech:{t}-{iso.lower()}",
            "entity_type": "technology",
            "canonical_name": no_commas(f"{canonical_name(t)} - {country}"),
            "aliases": "",
            "description": no_commas(
                f"Technology adoption (CHAT variable {t}) in {country}; "
                f"first recorded nonzero usage: {year}"),
            "relation": "available",
            "start_year": year,
            "end_year": "",
            "confidence": "medium",
            "notes": SECONDARY_NOTE,
        })
    return rows


def validate(rows: list[dict], techs: list[str], cap: int,
             first_country) -> None:
    header = ["entity_id", "entity_type", "canonical_name", "aliases",
              "description", "relation", "start_year", "end_year",
              "confidence", "notes"]
    assert header[0] == "entity_id" and len(header) == 10
    entity_types = {"person", "event", "polity", "organization", "work",
                    "technology"}
    relations = {"alive", "ongoing", "exists", "active", "available"}
    confs = {"high", "medium", "low"}
    ids = [r["entity_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate entity_ids"
    for r in rows:
        assert set(r) == set(header), r
        assert r["entity_type"] in entity_types, r
        assert r["relation"] in relations, r
        assert r["confidence"] in confs, r
        assert r["start_year"] == "" or int(r["start_year"]), r
        assert r["end_year"] == "" or int(r["end_year"]), r
        assert "," not in "".join(str(v) for v in r.values()), r
        assert r["entity_id"].startswith("tech:"), r
    assert len(rows) <= cap, (len(rows), cap)
    # every technology appears as a PRIMARY row
    assert {r["entity_id"] for r in rows if "-" not in r["entity_id"][5:]} \
        == {f"tech:{t}" for t in techs}
    n_secondary = sum(1 for r in rows if r["entity_id"].count("-") >= 1)
    assert n_secondary == min(len(first_country), cap - len(techs)), n_secondary


SPOT_CODES = ["railline", "telegram", "telephone", "radio", "tv", "cellphone",
              "computer", "internetuser", "atm", "vehicle_car", "elecprod"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    repo = here.parent.parent
    ap.add_argument("--out", default=str(repo / "fixtures"
                                         / "curated_tech_adoption_cominhobijn.csv"))
    ap.add_argument("--cap", type=int, default=3000,
                    help="maximum total rows (default 3000)")
    ap.add_argument("--raw-dir", default=str(repo / ".tmp-cominhobijn"),
                    help="gitignored directory for raw downloads")
    ap.add_argument("--spot-check", action="store_true",
                    help="print famous-technology spot-check table")
    args = ap.parse_args()

    raw = fetch_raw(Path(args.raw_dir))
    techs, first_any, first_country = load_first_years(raw)

    missing_iso = {c for (_, c) in first_country} - set(COUNTRY_ISO)
    assert not missing_iso, f"countries missing ISO map: {missing_iso}"

    rows = build_rows(techs, first_any, first_country, args.cap)
    rows.sort(key=lambda r: r["entity_id"])
    validate(rows, techs, args.cap, first_country)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "entity_id", "entity_type", "canonical_name", "aliases",
            "description", "relation", "start_year", "end_year",
            "confidence", "notes"])
        w.writeheader()
        w.writerows(rows)

    conf = Counter(r["confidence"] for r in rows)
    n_primary = sum(1 for r in rows if r["entity_id"].count("-") < 1)
    print(f"technologies={len(techs)} rows={len(rows)} "
          f"(primary={n_primary} secondary={len(rows) - n_primary}) "
          f"confidence: high={conf['high']} medium={conf['medium']} "
          f"low={conf['low']}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)

    if args.spot_check:
        print("\nSPOT CHECK (technology | first recorded usage anywhere in "
              "sample):", file=sys.stderr)
        for t in SPOT_CODES:
            year, country, v = first_any[t]
            print(f"  {t:14s} {year} in {country} (value {v})",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
