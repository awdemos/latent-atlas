#!/usr/bin/env python3
"""Convert DPRR (Digital Prosopography of the Roman Republic) MRR XML exports
into a Latent Atlas curated CSV fixture (schema of fixtures/curated_roman.csv).

Sources (read-only), relative to the dprr-django checkout:
  promrep/scripts/data/mrr1_all_MR_Officesv24.docx.html.xml  (509-100 BCE)
  promrep/scripts/data/mrr2_converted_html_MRv12.xml        (99-31 BCE)

Both files are Broughton's "Magistrates of the Roman Republic" (MRR) converted
to a simple XML: <year name="99 B.C. A.U.C. 655"> -> <office name="Consuls">
-> <person name="M. Antonius M. f. M. n. (28)" office-xref="Pr. 102"/>.
The trailing parenthetical holds the person's Realencyclopaedie (RE) number,
which together with the nomen (family name) is the stable identity key.

era_dates.csv joinability verdict: NOT JOINABLE. era_dates.csv keys on the
Django Person table primary key (see promrep/management/commands/
import_era_dates.py: Person.objects.get(id=person_id)). No file in the repo
maps those database ids to names: the only named person export,
data/SBPersonsExportV2.{csv,xlsx}, has person_id=0 in every row (verified in
both CSV and XLSX), so the ~4,899 era ranges cannot be attached to any name.
Consequently all intervals below are derived from MRR attestation years only.
(For the record: import_era_dates.py stores the integers verbatim and
promrep/forms.py clean_era_from negates a positive BCE input, so era_dates
values use signed-BCE convention, e.g. -540 = 540 BCE = astronomical -539.
This convention is documented here but unused, since the join is impossible.)

Year convention (Latent Atlas stores ASTRONOMICAL years):
  BCE year Y -> astronomical 1 - Y   (99 BCE -> -98, 1 BCE -> 0, 509 BCE -> -508)
  CE year Y  -> astronomical Y       (1 CE -> 1)
Helper bce_to_astro() below is unit-checked against exactly those cases.

Identity and name normalization:
  * Identity key = (normalized nomen, RE number). The trailing "(...)" group is
    parsed; accepted forms: "(28)", "(*36)", "(Cap. *1)", "(Iulius 132)",
    "(Iulius, no. 132)". Groups like "(not in RE)", "(cf. 5)", "(see Consuls)",
    "(RE 18.1.877)" carry no RE number and the person is skipped. Multi-number
    groups ("(152, 153)") use the first number.
  * A date prefix on the name ("53-43: M. Tullius Cicero (29)",
    "47 B.C. - 14 A.D.: C. Octavius (Thurinus) (Iulius 132)") contributes its
    years as additional attestations (BCE unless the segment contains "A. D.").
  * Praenomen abbreviations are expanded (M.=Marcus, C.=Gaius, Cn.=Gnaeus,
    L.=Lucius, P.=Publius, Q.=Quintus, Sex.=Sextus, T.=Titus, Ti.=Tiberius,
    A.=Aulus, D.=Decimus, N.=Numerius, M'=Manius, Ap.=Appius, Sp.=Spurius,
    Ser.=Servius, K.=Kaeso, Mam.=Mamercus).
  * Filiation tokens (praenomen-like token followed by "f."/"n.", plus "- f.",
    "Pat.") are dropped when building the canonical name, so
    "Cn. Pompeius Cn. f. Sex. n. Magnus (*27)" becomes "Gnaeus Pompeius Magnus".
  * Spelling variants are folded for the identity key: Iulius->Julius,
    Iunius->Junius.

Attestation years collected per person:
  1. The <year> the person appears under.
  2. Career years in office-xref such as "Pr. 102", "Cos. 104-100, 86",
     "Cos. 105, Pr. by 118" (only numbers immediately following an office
     abbreviation, filtered to the plausible 10..509 BCE window).
  3. Date-prefix years on the person's name attribute (see above).

Interval derivation rules (documented per row in `notes`):
  start_year = min(attestation years, astronomical) - 40
  end_year   = max(attestation years, astronomical) + 5
  Rationale: a first MRR attestation is typically a quaestorship/praetorship/
  consulship (consuls were >= 42 years old), so -40 is a defensible lower bound
  for "was plausibly alive by then"; +5 extends past the last attestation.
  Rows that would produce start_year > end_year are skipped (unreachable with
  these rules, asserted anyway).

Confidence:
  high   - >= 3 distinct attestation years spanning >= 15 years (tight floruit)
  medium - >= 2 distinct attestation years
  low    - exactly 1 attestation year (dates are pure guesswork)

Output schema (exactly fixtures/curated_roman.csv):
  entity_id,entity_type,canonical_name,aliases,description,relation,
  start_year,end_year,confidence,notes
  entity_id   = dprr:<nomen-lowercase>-<re-number>   e.g. dprr:antonius-28
  entity_type = person, relation = alive
  aliases     = raw MRR name strings, '|'-separated (max 6)
  description = "Roman magistrate" plus top offices, no commas
  no field contains a comma (sanitized to ';')

Usage:
  python3 scripts/dprr2fixture.py [--dprr /path/to/dprr-django]
                                [--out fixtures/curated_romans_dprr.csv]
                                [--spot-check]

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Year convention (astronomical years; 1 BCE = 0)

def bce_to_astro(year_bce: int) -> int:
    """Convert a BCE year (positive, e.g. 99) to astronomical year (-98)."""
    return 1 - year_bce


def astro_label(astro: int) -> str:
    """Astronomical year -> human label: -98 -> '98 BCE', 14 -> '14 CE'."""
    return f"{1 - astro} BCE" if astro <= 0 else f"{astro} CE"


def _unit_check_years() -> None:
    assert bce_to_astro(99) == -98, "99 BCE must be -98"
    assert bce_to_astro(1) == 0, "1 BCE must be 0"
    assert bce_to_astro(509) == -508, "509 BCE must be -508"
    assert astro_label(-98) == "99 BCE"
    assert astro_label(0) == "1 BCE"
    assert astro_label(1) == "1 CE"


# ---------------------------------------------------------------------------
# Name parsing

PRAENOMENS = {
    "M": "Marcus", "C": "Gaius", "Cn": "Gnaeus", "L": "Lucius",
    "P": "Publius", "Q": "Quintus", "Sex": "Sextus", "T": "Titus",
    "Ti": "Tiberius", "A": "Aulus", "D": "Decimus", "N": "Numerius",
    "M'": "Manius", "Ap": "Appius", "Sp": "Spurius", "Ser": "Servius",
    "K": "Kaeso", "Mam": "Mamercus", "Vol": "Volesus",
    # "Tr." appears as a source-data typo for Ti. (Tiberius) at name start.
    "Tr": "Tiberius",
}

# Fold common spelling variants for identity keys.
NOMEN_FOLD = {"iulius": "julius", "iunius": "junius"}
# Display fold for canonical names (capitalized).
NOMEN_FOLD_DISPLAY = {"iulius": "Julius", "iunius": "Junius"}

# Hand-curated canonical names for a few famous identities where the raw MRR
# strings are ambiguous or confusing on their own (e.g. both Caesar and the
# adopted Octavian appear as "C. Iulius ... Caesar ... (13x)"). Everything
# else is derived from the data.
DISPLAY_OVERRIDES = {
    ("julius", 131): "Gaius Julius Caesar",
    ("julius", 132): "Augustus",
    ("porcius", 10): "Marcus Porcius Cato (Cato the Elder)",
    ("porcius", 20): "Marcus Porcius Cato (Cato the Younger)",
    ("cornelius", 336): "Publius Cornelius Scipio Africanus",
    ("licinius", 68): "Marcus Licinius Crassus",
    ("junius", 53): "Marcus Junius Brutus",
}

RE_GROUP_STOPWORDS = {"no", "cf", "see", "not", "in", "re", "col", "suppl",
                      "supb", "fr", "vol", "p", "pp", "and", "note", "notes",
                      "n"}

GENERIC_PRae_RE = re.compile(r"^[A-Z]{1,3}\.?$")

DROP_TOKENS = {"?", "??", "-", "–", "—", "&", "and", "or", "pat", "f", "n",
               "cf", "see"}

# Career-summary markers that start the office/year tail of a person name
# ("Matho (*6) Cos. 231, Pr. 216 ? Augur ?- - -204 (see 217, note 4)").
CAREER_CUT_RE = re.compile(r"\s(?:Cos|Cens|Dict|Mon|Aed|Pr|Tr)\.\s|\sAugur\b|"
                           r"\sPont\b|\sConsul\b")


def _clean_token(tok: str) -> str:
    return tok.strip("[](){},;.")


def _is_praenomen_like(tok: str) -> bool:
    t = _clean_token(tok).rstrip(".")
    return t in PRAENOMENS or bool(GENERIC_PRae_RE.fullmatch(_clean_token(tok)))


def fold_nomen(nomen: str) -> str:
    key = re.sub(r"[^a-z]", "", nomen.lower())
    return NOMEN_FOLD.get(key, key)


DATE_PREFIX_RE = re.compile(r"^([\s\d?.,/A-Za-z\-–—]+):\s+(\S.*)$")


def parse_date_prefix(raw_name: str) -> tuple[list[int], str]:
    """Extract a leading life/activity-date prefix like '53-43: ' or
    '47 B.C. - 14 A.D.: ' from a person name. Returns (astro_years, rest)."""
    m = DATE_PREFIX_RE.match(raw_name)
    if not m or not any(ch.isdigit() for ch in m.group(1)):
        return [], raw_name
    prefix, rest = m.group(1), m.group(2)
    years: list[int] = []
    for piece in re.split(r"\s+or\s+", prefix.replace("–", "-")):
        for seg in piece.split("-"):
            is_ce = bool(re.search(r"A\.?\s*D\.?", seg))
            for num in re.findall(r"\d{1,4}", seg):
                n = int(num)
                if is_ce:
                    if 1 <= n <= 200:
                        years.append(n)
                elif 1 <= n <= 600:
                    years.append(bce_to_astro(n))
    return years, rest


def parse_re_group(inner: str) -> tuple[str | None, str | None]:
    """Parse the trailing '(...)' group. Returns (nomen_override, number)."""
    tokens = inner.replace("(", " ").replace(")", " ").replace(",", " ").split()
    words = []
    numbers = []
    for tok in tokens:
        t = tok.strip("?;:")
        if re.fullmatch(r"\*?\d+", t):
            numbers.append(t.lstrip("*"))
        else:
            w = t.rstrip(".").lower()
            if re.fullmatch(r"[A-Z][A-Za-z]{1,20}", t.rstrip(".")) \
                    and w not in RE_GROUP_STOPWORDS:
                words.append(t.rstrip("."))
    if not numbers:
        return None, None
    number = numbers[0]
    override = words[0] if words else None
    return override, number


def parse_person_name(raw_name: str) -> dict | None:
    """Parse an MRR person name. Returns None when no RE number is present.

    On success returns dict with:
      nomen_key, nomen_display, re_number, re_star, canonical, aliases=[raw]"""
    prefix_years, body = parse_date_prefix(raw_name.strip())
    groups = [(m.start(), m.end(), m.group(1).strip())
              for m in re.finditer(r"\(([^()]*)\)", body)]
    if not groups:
        return None
    # Identity group: the first parenthetical that yields an RE number
    # (cross-reference groups like "(see 217, note 4)" never win).
    override = number = None
    id_start = id_end = id_inner = None
    for id_start, id_end, id_inner in groups:
        if not re.search(r"\d", id_inner):
            continue
        override, number = parse_re_group(id_inner)
        if number is not None:
            break
    if number is None or not number.isdigit():
        return None
    num = int(number)
    if not (1 <= num <= 2000):
        return None

    # Split around the identity group. Text before it is the proper name;
    # text after it is a career summary whose years we harvest, up to the
    # first office marker. Remaining parentheticals are dropped.
    pre = re.sub(r"\([^()]*\)", " ", body[:id_start])
    post = body[id_end:]
    cut = CAREER_CUT_RE.search(post)
    post_head, career_tail = (post[:cut.start()], post[cut.start():]) if cut \
        else (post, "")
    extra_years = xref_years(career_tail)
    name_part = (pre + " " + re.sub(r"\([^()]*\)", " ", post_head)).strip()
    tokens = [_clean_token(t) for t in name_part.split()]
    tokens = [t for t in tokens if t]

    # Drop filiation pairs: praenomen-like token followed by f./n.
    kept: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt.lower().rstrip(".") in {"f", "n"} and (
            _is_praenomen_like(t) or _clean_token(t) in {"-", "?"}
        ):
            i += 2
            continue
        kept.append(t)
        i += 1

    # Determine the nomen: explicit override from the RE group wins; otherwise
    # the first token that is not a praenomen/marker/stopword.
    nomen_display = None
    rest: list[str] = []
    for t in kept:
        low = t.lower().rstrip(".?")
        if nomen_display is None:
            if low in DROP_TOKENS or _is_praenomen_like(t):
                continue
            if re.fullmatch(r"[A-Z][A-Za-z]{2,}", t):
                nomen_display = t
                continue
            continue  # skip stray markers before the nomen
        rest.append(t)
    if nomen_display is None and override:
        nomen_display = override
    if nomen_display is None:
        return None

    # Canonical name from the filiation-dropped tokens: expand the first
    # praenomen seen, keep nomen + cognomina (capitalized words), drop stray
    # markers, numerals, and abbreviation residue ("Mil", "Tr", "c", "p").
    canon_tokens: list[str] = []
    praenomen_done = False
    for t in kept:
        low = t.lower().rstrip(".?")
        if low in DROP_TOKENS or re.fullmatch(r"\*?\d+", t):
            continue
        key = t.rstrip(".")
        if key in PRAENOMENS:
            if not praenomen_done:
                canon_tokens.append(PRAENOMENS[key])
                praenomen_done = True
            continue
        if _is_praenomen_like(t):
            continue
        if not re.fullmatch(r"[A-Z][a-zA-Z]{2,}", t):
            continue
        # Display fold: Iulius -> Julius, Iunius -> Junius.
        canon_tokens.append(NOMEN_FOLD_DISPLAY.get(t.lower(), t))
    canonical = " ".join(canon_tokens).strip() or nomen_display

    nomen_for_key = override if override else nomen_display
    return {
        "nomen_key": fold_nomen(nomen_for_key),
        "nomen_display": nomen_for_key.strip("?"),
        "re_number": num,
        "re_star": "( *" in f"({id_inner})" or id_inner.strip().startswith("*"),
        "canonical": canonical,
        "raw": raw_name.strip(),
        "prefix_years": prefix_years + extra_years,
    }


# ---------------------------------------------------------------------------
# office-xref career-year extraction

XREF_RE = re.compile(
    r"(?:Cos|Cens|Dict|Pr|Aed|Quaest|Trib|Tr|Mon|Augur|Pont|Interrex|Pref)"
    r"[A-Za-z]*\.?(?:\s+[A-Z][a-z]{1,12}\.?){0,2}(?:\s+by)?\s*"
    r"(\??\d{2,3}(?:\s*[-–,]\s*\??\d{2,3})*)"
)

XREF_YEAR_WINDOW = range(10, 510)  # plausible BCE career years in MRR


def xref_years(xref: str) -> list[int]:
    out = []
    for m in XREF_RE.finditer(xref):
        for num in re.findall(r"\d{2,3}", m.group(1)):
            n = int(num)
            if n in XREF_YEAR_WINDOW:
                out.append(bce_to_astro(n))
    return out


# ---------------------------------------------------------------------------
# Office normalization for descriptions

OFFICE_RANK = [
    "dictator", "master of horse", "consul", "censor", "praetor", "aedile",
    "tribune of the plebs", "quaestor", "promagistrate", "triumvir",
    "decemvir", "legate", "ambassador", "envoy", "prefect", "pontifex",
    "augur", "rex sacrorum", "interrex", "duumvir", "quinquevir",
    "quindecimvir", "septemvir epulo", "flamen", "curio maximus",
    "lupercus", "salius", "fetialis", "priest", "vestal virgin",
    "military tribune", "iudex quaestionis", "special commission",
]


def norm_office(name: str) -> str:
    """Map an MRR office heading onto a short canonical label (keyword rules
    cover the many spelling/plural variants in the two XML files)."""
    k = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    k = re.sub(r"\s+", " ", k).strip()
    if "dictator" in k:
        return "dictator"
    if "master of horse" in k:
        return "master of horse"
    if "consul" in k:
        return "consul"
    if "censor" in k:
        return "censor"
    if "praetor" in k:
        return "praetor"
    if "aedil" in k:
        return "aedile"
    if "tribune" in k and ("soldier" in k or "solider" in k):
        return "military tribune"
    if "tribune" in k:
        return "tribune of the plebs"
    if "quaestor" in k or "questor" in k:
        return "quaestor"
    if "promagistrate" in k:
        return "promagistrate"
    if "lieutenant" in k:
        return "legate"
    if "ambassador" in k:
        return "ambassador"
    if "envoy" in k:
        return "envoy"
    if "prefect" in k or "praefectus" in k:
        return "prefect"
    if "pontif" in k:
        return "pontifex"
    if "augur" in k:
        return "augur"
    if "rex sacrorum" in k:
        return "rex sacrorum"
    if "interrex" in k or "interreges" in k:
        return "interrex"
    if "triumvir" in k or "tresviri" in k:
        return "triumvir"
    if "decemvir" in k:
        return "decemvir"
    if "duumvir" in k:
        return "duumvir"
    if "quinquevir" in k:
        return "quinquevir"
    if "septemvir" in k:
        return "septemvir epulo"
    if "quindec" in k:
        return "quindecimvir"
    if "vestal" in k or "virgin" in k:
        return "vestal virgin"
    if "luperc" in k:
        return "lupercus"
    if "sali" in k:
        return "salius"
    if "iudic" in k or "quaesitor" in k:
        return "iudex quaestionis"
    if "flamen" in k or "flamin" in k:
        return "flamen"
    if "curio" in k:
        return "curio maximus"
    if "fetial" in k:
        return "fetialis"
    if "priest" in k:
        return "priest"
    return "special commission"  # commissions, committees, boards, prosecutors


# ---------------------------------------------------------------------------
# MRR XML parsing

YEAR_RE = re.compile(r"^(\d+)\s*B\.\s*C\.")


def parse_mrr_xml(path: Path, source_label: str, persons: dict) -> None:
    """Feed one MRR XML file into `persons`: key (nomen_key, re) -> record."""
    root = ET.parse(path).getroot()
    for year in root.iter("year"):
        ym = YEAR_RE.match(year.get("name", ""))
        if not ym:
            continue
        year_astro = bce_to_astro(int(ym.group(1)))
        for office in year.iter("office"):
            office_norm = norm_office(office.get("name", ""))
            for person in office.iter("person"):
                parsed = parse_person_name(person.get("name", ""))
                if parsed is None:
                    persons["__skipped__"] += 1
                    continue
                key = (parsed["nomen_key"], parsed["re_number"])
                rec = persons["by_key"].get(key)
                if rec is None:
                    rec = {
                        "nomen_key": parsed["nomen_key"],
                        "nomen_display": parsed["nomen_display"],
                        "re_number": parsed["re_number"],
                        "re_star": parsed["re_star"],
                        "years": set(),
                        "offices": Counter(),
                        "raw_names": Counter(),
                        "canon_by_raw": {},
                        "sources": set(),
                    }
                    persons["by_key"][key] = rec
                rec["years"].add(year_astro)
                rec["years"].update(parsed["prefix_years"])
                rec["offices"][office_norm] += 1
                rec["raw_names"][parsed["raw"]] += 1
                rec["canon_by_raw"][parsed["raw"]] = parsed["canonical"]
                xref = person.get("office-xref", "")
                if xref:
                    rec["years"].update(xref_years(xref))
                rec["sources"].add(source_label)


# ---------------------------------------------------------------------------
# Fixture assembly


def no_commas(s: str) -> str:
    return s.replace(",", ";").replace("\n", " ").strip()


def build_row(rec: dict) -> dict:
    years = sorted(rec["years"])
    start, end = years[0] - 40, years[-1] + 5
    assert start <= end, f"start > end for {rec}"

    n_years = len(years)
    span = years[-1] - years[0]
    if n_years >= 3 and span >= 15:
        confidence = "high"
    elif n_years >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Canonical name: derive from the raw variant with the most attestations;
    # tie-break against imperial-title tokens, then longest, then alphabetical.
    weird = ("imp", "imperator", "divi", "caesar divi")

    def variant_score(item):
        raw, count = item
        canon = rec["canon_by_raw"][raw]
        bad = sum(1 for t in canon.lower().split() if t in weird)
        # Prefer names free of imperial-title residue ("Imp Caesar Divi"),
        # then the most-attested variant.
        return (bad, -count, -len(canon.split()), canon)

    best_raw = min(rec["raw_names"].items(), key=variant_score)[0]
    canonical = rec["canon_by_raw"][best_raw]
    canonical = DISPLAY_OVERRIDES.get(
        (rec["nomen_key"], rec["re_number"]), canonical)

    # Aliases: raw MRR name strings (most frequent first), canonical excluded.
    raws = [r for r, _ in sorted(rec["raw_names"].items(),
                                 key=lambda kv: (-kv[1], kv[0]))
            if r != best_raw][:5]

    # Description: top offices by seniority.
    top_offices = sorted(rec["offices"].items(),
                         key=lambda kv: (OFFICE_RANK.index(kv[0])
                                         if kv[0] in OFFICE_RANK else 99,
                                         -kv[1]))[:3]
    office_str = "; ".join(o for o, _ in top_offices if o)
    description = "Roman magistrate"
    if office_str:
        description += "; offices: " + office_str

    re_display = f"{rec['nomen_display']} {rec['re_number']}"
    vols = "+".join(sorted(rec["sources"]))
    notes = (
        f"RE {re_display}; {vols} attestations "
        f"{astro_label(years[0])} to {astro_label(years[-1])} (n={n_years}); "
        f"floruit derived: earliest attestation - 40 and latest + 5"
    )

    entity_id = f"dprr:{rec['nomen_key']}-{rec['re_number']}"
    return {
        "entity_id": entity_id,
        "entity_type": "person",
        "canonical_name": canonical,
        "aliases": "|".join(no_commas(a) for a in raws),
        "description": no_commas(description),
        "relation": "alive",
        "start_year": start,
        "end_year": end,
        "confidence": confidence,
        "notes": no_commas(notes),
    }


SPOT_CHECK = {
    "Julius Caesar": ("julius", 131),
    "Cicero": ("tullius", 29),
    "Pompey (Cn. Pompeius Magnus)": ("pompeius", 15),
    "Augustus": ("julius", 132),
    "Cato the Elder": ("porcius", 10),
    "Cato the Younger": ("porcius", 20),
    "Scipio Africanus": ("cornelius", 336),
    "Sulla": ("cornelius", 392),
    "Marius": ("marius", 14),
    "Crassus (M. Licinius Crassus)": ("licinius", 68),
    "Brutus (M. Iunius Brutus)": ("junius", 53),
    "Mark Antony": ("antonius", 28),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    ap.add_argument("--dprr", default=str(here.parent.parent.parent / "dprr-django"),
                    help="path to the dprr-django checkout")
    ap.add_argument("--out", default=str(here.parent.parent / "fixtures"
                                         / "curated_romans_dprr.csv"))
    ap.add_argument("--spot-check", action="store_true",
                    help="print the famous-person spot-check table to stderr")
    args = ap.parse_args()

    _unit_check_years()

    data = Path(args.dprr) / "promrep" / "scripts" / "data"
    persons = {"by_key": {}, "__skipped__": 0}
    parse_mrr_xml(data / "mrr1_all_MR_Officesv24.docx.html.xml", "MRR1", persons)
    parse_mrr_xml(data / "mrr2_converted_html_MRv12.xml", "MRR2", persons)

    rows = []
    for rec in persons["by_key"].values():
        rows.append(build_row(rec))
    rows.sort(key=lambda r: r["entity_id"])

    # Self-validation: unique ids, start <= end, no commas, no empty names.
    ids = [r["entity_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate entity_ids"
    for r in rows:
        assert r["start_year"] <= r["end_year"], r
        assert "," not in "".join(str(v) for v in r.values()), r
        assert r["canonical_name"] and r["entity_id"].startswith("dprr:"), r

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
    print(f"persons={len(rows)} skipped_person_elements="
          f"{persons['__skipped__']} "
          f"confidence: high={conf['high']} medium={conf['medium']} "
          f"low={conf['low']}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)

    if args.spot_check:
        print("\nSPOT CHECK (name | entity_id | interval astro [labels] | conf | "
              "n_attest):", file=sys.stderr)
        for name, key in SPOT_CHECK.items():
            rec = persons["by_key"].get(key)
            if rec is None:
                print(f"  {name:34s} ABSENT", file=sys.stderr)
                continue
            ys = sorted(rec["years"])
            row = build_row(rec)
            print(f"  {name:34s} {row['entity_id']:22s} "
                  f"[{row['start_year']},{row['end_year']}] "
                  f"({astro_label(row['start_year'])}..{astro_label(row['end_year'])}) "
                  f"{row['confidence']:6s} n={len(ys)}", file=sys.stderr)
        cleo = [k for k in persons["by_key"] if k[0] == "cleopatra"]
        print(f"  Cleopatra                        "
              f"{'ABSENT (not a Roman magistrate)' if not cleo else 'PRESENT?'}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
