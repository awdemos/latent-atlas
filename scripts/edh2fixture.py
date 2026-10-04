#!/usr/bin/env python3
"""Convert the Epigraphic Database Heidelberg (EDH) CSV data dump into a
Latent Atlas curated CSV fixture (schema of fixtures/curated_roman.csv).

Source: https://edh.ub.uni-heidelberg.de (Epigraphic Database Heidelberg,
Heidelberger Akademie der Wissenschaften). The full database dump is at
  https://edh.ub.uni-heidelberg.de/data/download/edh_data_text.csv
(one row per inscription, documented at /data/api -> "Download Data Dumps").
All EDH open data is licenced CC BY-SA 4.0; the licence requires
attribution, so every row's notes cite EDH and the retrieval date.

Dump conventions (verified against the EDH web interface, e.g.
HD000001 "71 AD - 130 AD" and HD000007 "100 BC - 51 BC"):
  * dat_jahr_a = dating "not before" (year integer, BC = negative,
    e.g. -100 = 100 BCE); dat_jahr_e = "not after". The CSV dump labels
    these correctly. NOTE: the JSON search API (/data/api/inschrift/suche)
    returns the same two values with the KEYS SWAPPED (its "not_before"
    holds the upper bound); the CSV dump is authoritative here.
  * Years use signed historical convention (-100 = 100 BCE, no year 0 in
    the source). Latent Atlas stores ASTRONOMICAL years, so:
    BCE value y -> 1 + y (=-100 -> -99, -1 -> 0); CE value unchanged.
  * provinz = 2-3 letter province code; i_gattung = inscription genre
    code (a trailing "?" marks the genre attribution as uncertain).
    Both vocabularies are documented under "Controlled Vocabularies" on
    https://edh.ub.uni-heidelberg.de/data/api and inlined below.

Fixture rules:
  * entity_id  = edh:<hd-number lowercased> (HD numbers are unique).
  * entity_type "work", relation "available" (the physical inscription
    is extant/available; its *dating* is carried by start_year, the
    not-before year in astronomical numbering; end_year is left open).
  * confidence "high" if the EDH dating window (not_after - not_before)
    is <= 20 years, else "medium". Rows without both date bounds are
    dropped (see drop report printed at the end).
  * Selection: inscriptions are taken in HD-number order (the dump's
    accession order) with a per-province cap (--per-province-cap) so the
    fixture is not dominated by the large Italian corpora; taking the
    first --max-rows valid rows keeps the selection deterministic.

Raw downloads are cached under a gitignored temp dir (--raw-dir,
default <repo>/.tmp-edh-raw; see .gitignore) so the fixture can be
regenerated without re-downloading.
"""

import argparse
import csv
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

DUMP_URL = "https://edh.ub.uni-heidelberg.de/data/download/edh_data_text.csv"
EDH_HOME = "https://edh.ub.uni-heidelberg.de"
EDH_NAME = "Epigraphic Database Heidelberg"
RETRIEVED = "2026-10-04"

HEADER = ["entity_id", "entity_type", "canonical_name", "aliases",
          "description", "relation", "start_year", "end_year",
          "confidence", "notes"]

# Controlled vocabulary "Province", from /data/api (retrieved 2026-10-04).
PROVINCES = {
    "Ach": "Achaia", "Aeg": "Aegyptus", "Aem": "Aemilia (Regio VIII)",
    "Afr": "Africa Proconsularis", "AlC": "Alpes Cottiae",
    "AlG": "Alpes Graiae", "AlM": "Alpes Maritimae",
    "AlP": "Alpes Poeninae", "ApC": "Apulia et Calabria (Regio II)",
    "Aqu": "Aquitania", "Ara": "Arabia", "Arm": "Armenia",
    "Asi": "Asia", "Ass": "Assyria", "Bae": "Baetica",
    "Bar": "Barbaricum", "Bel": "Belgica", "BiP": "Bithynia et Pontus",
    "BrL": "Bruttium et Lucania (Regio III)", "Bri": "Britannia",
    "Cap": "Cappadocia", "Cil": "Cilicia", "Cor": "Corsica",
    "Cre": "Creta", "Cyp": "Cyprus", "Cyr": "Cyrene", "Dac": "Dacia",
    "Dal": "Dalmatia", "Epi": "Epirus", "Etr": "Etruria (Regio VII)",
    "Gal": "Galatia", "GeI": "Germania inferior",
    "GeS": "Germania superior", "HiC": "Hispania citerior",
    "Inc": "Provincia incerta", "Iud": "Iudaea",
    "LaC": "Latium et Campania (Regio I)", "Lig": "Liguria (Regio IX)",
    "Lug": "Lugdunensis", "Lus": "Lusitania",
    "LyP": "Lycia et Pamphylia", "MaC": "Mauretania Caesariensis",
    "MaT": "Mauretania Tingitana", "Mak": "Macedonia",
    "Mes": "Mesopotamia", "MoI": "Moesia inferior",
    "MoS": "Moesia superior", "Nar": "Narbonensis", "Nor": "Noricum",
    "Num": "Numidia", "PaI": "Pannonia inferior",
    "PaS": "Pannonia superior", "Pic": "Picenum (Regio V)",
    "Rae": "Raetia", "ReB": "Regnum Bospori", "Rom": "Roma",
    "Sam": "Samnium (Regio IV)", "Sar": "Sardinia",
    "Sic": "Sicilia, Melita", "Syr": "Syria", "Thr": "Thracia",
    "Tra": "Transpadana (Regio XI)", "Tri": "Tripolitania",
    "Umb": "Umbria (Regio VI)", "Val": "Valeria",
    "VeH": "Venetia et Histria (Regio X)",
}

# Controlled vocabulary "Type of Inscription", from /data/api, plus the
# conventional Latin titulus term used in the fixture notes.
GENRES = {
    "brief": ("Letter", "epistula"),
    "diplmil": ("Military diploma", "diploma militare"),
    "elogium": ("Elogium", "elogium"),
    "fasti": ("Calendar", "fasti"),
    "indexlaterc": ("List", "index"),
    "miliarium": ("Mile-/Leaguestone", "miliarium"),
    "nota": ("Identification inscription", "nota"),
    "oratio": ("Prayer", "oratio"),
    "titaccl": ("Acclamation", "acclamatio"),
    "titadnun": ("Adnuntiatio", "adnuntiatio"),
    "titadsig": ("Assignation inscription", "titulus adsignationis"),
    "titdefix": ("Defixio", "defixio"),
    "tithon": ("Honorific inscription", "titulus honorarius"),
    "titiurpriv": ("Private legal inscription", "titulus iuridicus privatus"),
    "titiurpub": ("Public legal inscription", "titulus iuridicus publicus"),
    "titoppubpriv": ("Building/dedicatory inscription", "titulus dedicatorius"),
    "titpossfabr": ("Owner/artist inscription", "titulus possessorius/fabrilis"),
    "titreiexpl": ("Label", "titulus rei explicandae"),
    "titsac": ("Votive inscription", "titulus votivus"),
    "titsedspect": ("Seat inscription", "titulus sedilium"),
    "titsep": ("Epitaph", "titulus sepulcralis"),
    "titterm": ("Boundary inscription", "terminus"),
}

LANGS = {"L": "Latin", "G": "Greek", "B": "Latin/Greek", "H": "Hebrew"}


def edh_year_to_astro(y):
    """EDH signed year -> astronomical year (-100=100 BCE -> -99, -1 -> 0)."""
    return 1 + y if y < 0 else y


def fmt_year(y):
    """EDH signed year -> human string ("100 BCE" / "71 CE")."""
    return f"{-y} BCE" if y < 0 else f"{y} CE"


def download(raw_dir, url):
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / Path(urllib.parse.urlparse(url).path).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"using cached {dest}", file=sys.stderr)
        return dest
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "latent-atlas-fixture-builder"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def build_rows(csv_path, max_rows, per_province_cap):
    with open(csv_path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    records.sort(key=lambda r: r["hd_nr"])

    drops = Counter()
    prov_count = Counter()
    rows = []
    for rec in records:
        hd = rec["hd_nr"].strip()
        a_s, e_s = rec["dat_jahr_a"].strip(), rec["dat_jahr_e"].strip()
        if not a_s and not e_s:
            drops["no dating"] += 1
            continue
        if not a_s or not e_s:
            drops["incomplete dating (only one bound)"] += 1
            continue
        try:
            a, e = int(a_s), int(e_s)
        except ValueError:
            drops["unparseable dating"] += 1
            continue
        if a > e:
            drops["inverted dating (not_before > not_after)"] += 1
            continue
        if len(rows) >= max_rows:
            break
        prov_code = rec["provinz"].strip()
        if prov_count[prov_code] >= per_province_cap:
            continue
        prov_count[prov_code] += 1

        window = e - a
        confidence = "high" if window <= 20 else "medium"

        g_code = rec["i_gattung"].strip()
        uncertain = g_code.endswith("?")
        eng_type, latin_type = GENRES.get(g_code.rstrip("?") or "", (None, None))
        if eng_type is None:
            eng_type, latin_type = "Inscription", None

        fo_antik = rec["fo_antik"].strip()
        fo_modern = rec["fo_modern"].strip()
        place_bits = [b for b in (fo_antik, fo_modern) if b]
        place = " - ".join(place_bits) if place_bits else "unknown findspot"
        canonical = f"{eng_type} from {place}"

        lang = LANGS.get(rec["nl_text"].strip(), rec["nl_text"].strip())
        bits = []
        if lang:
            bits.append(f"{lang} {eng_type.lower()}")
        else:
            bits.append(eng_type)
        bits.append(f"EDH {hd} from {place}")
        if rec["denkmaltyp"].strip():
            bits.append(f"monument type: {rec['denkmaltyp'].strip()}")
        if rec["material"].strip():
            bits.append(f"material: {rec['material'].strip()}")
        transcr = " ".join(rec["atext"].split())
        if transcr:
            bits.append("transcription: " + transcr[:120] + ("..." if len(transcr) > 120 else ""))
        description = "; ".join(bits)

        prov_name = PROVINCES.get(prov_code, f"unknown province code {prov_code!r}")
        type_note = (f"inscription type: {latin_type} (EDH genre '{g_code}')"
                     if latin_type else
                     f"inscription type not recorded in EDH (genre code '{g_code or 'empty'}')")
        if uncertain:
            type_note += "; genre attribution uncertain in EDH"
        caveats = []
        if window > 100:
            caveats.append(f"wide EDH dating window ({window} years)")
        note = (f"Source: {EDH_NAME} ({EDH_HOME}), inscription {hd}, "
                f"retrieved {RETRIEVED}; {type_note}; "
                f"province: {prov_name} (EDH code {prov_code}); "
                f"EDH dating: not before {fmt_year(a)}, not after {fmt_year(e)} "
                f"({window}-year window); astronomical start_year "
                f"{edh_year_to_astro(a)}. Data licence CC BY-SA 4.0.")
        if caveats:
            note += " Caveats: " + "; ".join(caveats) + "."

        rows.append({
            "entity_id": f"edh:{hd.lower()}",
            "entity_type": "work",
            "canonical_name": canonical,
            "aliases": "",
            "description": description,
            "relation": "available",
            "start_year": str(edh_year_to_astro(a)),
            "end_year": "",
            "confidence": confidence,
            "notes": note,
        })
    return rows, drops


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)


def validate(path, expected_rows=None):
    entity_types = {"person", "event", "polity", "organization", "work", "technology"}
    relations = {"alive", "ongoing", "exists", "active", "available"}
    confidences = {"high", "medium", "low"}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == HEADER, f"header mismatch: {header}"
        n = 0
        ids = set()
        conf_dist = Counter()
        for row in r:
            assert len(row) == len(HEADER), f"row {n}: {len(row)} fields"
            rid, etype, _name, _al, _desc, rel, sy, ey, conf, _notes = row
            assert rid not in ids, f"duplicate entity_id {rid}"
            ids.add(rid)
            assert etype in entity_types, f"{rid}: bad entity_type {etype}"
            assert rel in relations, f"{rid}: bad relation {rel}"
            assert conf in confidences, f"{rid}: bad confidence {conf}"
            assert re_match_int(sy), f"{rid}: bad start_year {sy!r}"
            assert ey == "" or re_match_int(ey), f"{rid}: bad end_year {ey!r}"
            conf_dist[conf] += 1
            n += 1
    if expected_rows is not None:
        assert n == expected_rows, f"row count {n} != expected {expected_rows}"
    print(f"validation OK: {n} rows, header exact, ids unique, "
          f"enums/years valid; confidence: {dict(conf_dist)}")
    return n, conf_dist


def re_match_int(s):
    s = s.lstrip("-")
    return s.isdigit() and s != ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--raw-dir", default=str(repo / ".tmp-edh-raw"),
                    help="gitignored dir for raw downloads (default: <repo>/.tmp-edh-raw)")
    ap.add_argument("--out", default=str(repo / "fixtures" / "curated_inscriptions_edh.csv"))
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--per-province-cap", type=int, default=30)
    ap.add_argument("--dump-url", default=DUMP_URL)
    args = ap.parse_args()

    dump = download(Path(args.raw_dir), args.dump_url)
    rows, drops = build_rows(dump, args.max_rows, args.per_province_cap)
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    print(f"dropped rows by reason: {dict(drops)}", file=sys.stderr)
    validate(args.out, expected_rows=len(rows))


if __name__ == "__main__":
    main()
