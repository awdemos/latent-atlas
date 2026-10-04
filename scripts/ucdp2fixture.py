#!/usr/bin/env python3
"""Convert the UCDP Conflict Termination Dataset (conflict level) into a
Latent Atlas curated CSV fixture (schema of fixtures/curated_roman.csv).

Sources (downloaded read-only into the gitignored .tmp-ucdp/ dir):
  https://ucdp.uu.se/downloads/monadterm/UCDPConflictTerminationDataset_v4_2024_Conflict.csv
      UCDP Conflict Termination Dataset v.4-2024 (Joakim Kreutz, codebook
      dated June 24 2025; corresponds to UCDP/PRIO Armed Conflict Dataset
      v.25.1). One row per conflict-year, coded as-of each year: the
      definitive episode record (c_ep* fields, final actor identities) sits
      on the episode's LAST year row, so we take the max-year row per
      c_epid (= conflict_id concatenated with the episode counter c_epno).
  https://ucdp.uu.se/downloads/brd/ucdp-brd-conf-261-csv.zip
      UCDP Battle-Related Deaths Dataset v.26.1 (BattleDeaths_v26_1_conf.csv,
      one row per conflict-dyad-year with bd_best/bd_low/bd_high). Used only
      to enrich `notes` with a cumulative battle-deaths estimate per episode.
      BRD v.26.1 covers 1989 onward and is one annual version newer than the
      CT v.4-2024/ACD v.25.1 base, so deaths are summed only over years
      inside each episode window and small mismatches are possible;
      episodes of pre-1989 conflicts (or conflicts absent from BRD) get no
      estimate (noted in the row).

Row semantics:
  One row per conflict EPISODE (UCDP defines an episode as a continuous run
  of active conflict years; an episode ends when activity drops below the
  25-battle-death threshold, with c_outcome recording the termination type).
  entity_id   = ucdp:<conflict_id>-<c_epno>   (unique; c_epno restarts per
                conflict, so conflict_id alone would collide)
  entity_type = event, relation = ongoing
  start_year  = c_ep_startyear (astronomical numbering; UCDP only covers
                1946 onward and reports plain CE years, so values pass
                through unchanged)
  end_year    = c_ep_endyear, EMPTY when the episode is still open per
                c_epterm (0 = active in the following year, '' = open in
                the last year of the data window). NOTE WELL: UCDP codes an
                episode as ongoing until battle activity stays below the
                25-death threshold for a full year, so famous wars that
                ended politically without reaching that inactivity criterion
                (e.g. the Korean War 1953 armistice, the Iran-Iraq War
                1988 ceasefire, the 1991 Gulf War) are coded ONGOING with
                an empty end_year. This is faithful to the source; do not
                "fix" it from memory.
  confidence  = high for start_year >= 1946 (every UCDP row; the pre-1946
                medium rule never triggers because UCDP coverage starts in
                1946 - WWII and earlier wars are ABSENT from this source)

Caveats for maintainers:
  * WWII (1939-1945) is not in UCDP data at all; the famous-wars spot
    checks therefore use the Korean War, the Vietnam War and the
    Russia-Ukraine war instead.
  * UCDP splits long wars into multiple conflicts/episodes along actor
    lines: "Vietnam" appears as conflict 216 (France vs Viet Minh, from
    1946), 249 (South Vietnam vs FNL, from 1955) and 293 (South Vietnam
    vs North Vietnam, from 1965), each coded in its own terms.
  * Multi-value region strings like "1, 3" are mapped to all region names.
  * Attribution: UCDP requires citation. Required citation per the codebook:
    Kreutz, Joakim, 2010. "How and When Armed Conflicts End: Introducing
    the UCDP Conflict Termination Dataset." Journal of Peace Research
    47(2): 243-250 - plus the dataset itself ("UCDP Conflict Termination
    Dataset v.4-2024", Uppsala Conflict Data Program, ucdp.uu.se). Both
    are recorded in every row's notes.

Usage:
  python3 scripts/ucdp2fixture.py [--out fixtures/curated_conflicts_ucdp.csv]
                                  [--raw-dir .tmp-ucdp]
                                  [--max-rows 2000] [--spot-check]

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CT_URL = ("https://ucdp.uu.se/downloads/monadterm/"
          "UCDPConflictTerminationDataset_v4_2024_Conflict.csv")
BRD_URL = "https://ucdp.uu.se/downloads/brd/ucdp-brd-conf-261-csv.zip"
RETRIEVED = "2026-10-04"

FIELDNAMES = ["entity_id", "entity_type", "canonical_name", "aliases",
              "description", "relation", "start_year", "end_year",
              "confidence", "notes"]

REGION = {"1": "Europe", "2": "Middle East", "3": "Asia", "4": "Africa",
          "5": "Americas"}

TYPE_OF_CONFLICT = {"1": "extrasystemic", "2": "interstate",
                    "3": "intrastate"}
TYPE_OF_CONFLICT2 = {"2": "internal (internationalized)",
                     "3": "internationalized internal",
                     "4": "extrasystemic"}

OUTCOME = {"1": "peace agreement", "2": "ceasefire",
           "3": "victory for side A (government)",
           "4": "victory for side B (non-state side)",
           "5": "low activity", "6": "actor ceases to exist"}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached: {dest}", file=sys.stderr)
        return
    print(f"downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "latent-atlas-fixture"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as fh:
        fh.write(resp.read())


def no_commas(s: str) -> str:
    return s.replace(",", ";").replace("\n", " ").strip()


def load_brd_deaths(raw_dir: Path) -> dict[tuple[str, int], int]:
    """Return {(conflict_id, year): bd_best} from the BRD conflict dataset."""
    zip_path = raw_dir / "ucdp-brd-conf-261.zip"
    fetch(BRD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            rows = list(csv.DictReader((line.decode("utf-8")
                                        for line in fh.readlines())))
    deaths: dict[tuple[str, int], int] = {}
    for r in rows:
        best = int(r["bd_best"] or 0)
        if best < 0:  # UCDP missing-data sentinel
            continue
        key = (r["conflict_id"], int(r["year"]))
        deaths[key] = deaths.get(key, 0) + best
    return deaths


def load_episodes(ct_path: Path) -> list[dict]:
    """One record per conflict episode (c_epid).

    The CT file is conflict-year level and coded as-of each year: mid-episode
    rows have empty c_ep_endyear/c_epterm/c_ependdate because the episode had
    not ended yet at that time, and actor identities can change mid-episode
    (rebel renames). The definitive episode record is therefore the row of
    the episode's FINAL year (max `year`); for episodes still open at the
    data cut that is the latest in-window year. We also collect the distinct
    side names across all of the episode's rows as aliases.
    """
    rows = list(csv.DictReader(ct_path.open(newline="", encoding="utf-8")))
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ep[r["c_epid"]].append(r)
    episodes = []
    for epid, rs in by_ep.items():
        last = max(rs, key=lambda r: int(r["year"]))
        last["__side_names"] = sorted({r["side_a"] for r in rs}
                                      | {r["side_b"] for r in rs})
        episodes.append(last)
    return episodes


def build_row(ep: dict, deaths: dict[tuple[str, int], int]) -> dict:
    cid = ep["conflict_id"]
    epno = ep["c_epno"]
    start = int(ep["c_ep_startyear"])
    ongoing = ep["c_epterm"] in ("0", "")
    end = "" if ongoing else ep["c_ep_endyear"]
    dur = int(ep["c_ep_durcount"])

    regions = ";".join(REGION[c.strip()] for c in ep["region"].split(",")
                       if c.strip() in REGION)

    # deaths summed over BRD years inside the episode window
    end_i = start if ongoing else int(end)
    total = sum(deaths.get((cid, y), 0)
                for y in range(start, end_i + 1))

    outcome_txt = "ongoing (no termination coded)" if ongoing \
        else OUTCOME.get(ep["c_outcome"], f"outcome {ep['c_outcome']}")
    toc2 = TYPE_OF_CONFLICT2.get(ep["type_of_conflict2"],
                                 TYPE_OF_CONFLICT.get(ep["type_of_conflict"],
                                                      "unknown type"))

    notes = (
        f"Source: UCDP Conflict Termination Dataset v4-2024 "
        f"(Kreutz, JPR 47(2) 2010; Uppsala Conflict Data Program, "
        f"attribution required); retrieved {RETRIEVED} from ucdp.uu.se. "
        f"Region: {regions}. "
    )
    if total > 0:
        notes += (f"Battle-deaths estimate {total} (UCDP BRD v26.1 bd_best "
                  f"summed over episode years; BRD v26.1 covers 1989 onward "
                  f"and is one version newer than the CT base). ")
    else:
        notes += ("No battle-deaths estimate (UCDP BRD v26.1 covers 1989 "
                  "onward; this episode's conflict-years are absent from BRD "
                  "or have zero recorded deaths). ")
    notes += f"Outcome: {outcome_txt}."
    if ongoing:
        notes += (" end_year empty = episode ongoing per UCDP coding "
                  "(activity never fell below the 25 battle-deaths/year "
                  "threshold), not necessarily politically unresolved.")

    # Aliases: individual actor names observed across the episode that are
    # not part of the final canonical sides (fields may bundle several
    # actors separated by ', ' or ';', e.g. after rebel mergers).
    def actors(s: str) -> list[str]:
        return [a.strip() for a in re.split(r"[;,]", s) if a.strip()]

    canonical_actors = set(actors(ep["side_a"])) | set(actors(ep["side_b"]))
    seen: list[str] = []
    for n in ep["__side_names"]:
        for a in actors(n):
            if a not in canonical_actors and a not in seen:
                seen.append(a)
    aliases = "|".join(no_commas(a) for a in seen[:6])

    return {
        "entity_id": f"ucdp:{cid}-{epno}",
        "entity_type": "event",
        "canonical_name": no_commas(f"{ep['side_a']} vs {ep['side_b']}"),
        "aliases": aliases,
        "description": no_commas(
            f"UCDP armed conflict episode {epno} of conflict {cid} in "
            f"{ep['location']}; {toc2}; {dur} active year(s); "
            f"outcome: {outcome_txt}"),
        "relation": "ongoing",
        "start_year": start,
        "end_year": end,
        "confidence": "high" if start >= 1946 else "medium",
        "notes": no_commas(notes),
    }


SPOT_CHECK_IDS = {"ucdp:235-1": "Korean War",
                  "ucdp:249-1": "Vietnam War (South Vietnam vs FNL)",
                  "ucdp:293-1": "Vietnam War (South Vietnam vs North Vietnam)",
                  "ucdp:216-1": "First Indochina War (France vs Viet Minh)",
                  "ucdp:13243-1": "Russia-Ukraine war (2022)",
                  "ucdp:324-2": "Iran-Iraq War (1980 episode)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve()
    root = here.parent.parent
    ap.add_argument("--out", default=str(root / "fixtures"
                                         / "curated_conflicts_ucdp.csv"))
    ap.add_argument("--raw-dir", default=str(root / ".tmp-ucdp"),
                    help="gitignored dir for raw downloads (kept out of git)")
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--spot-check", action="store_true",
                    help="print the famous-conflict spot-check table")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ct_path = raw_dir / "UCDPConflictTerminationDataset_v4_2024_Conflict.csv"
    fetch(CT_URL, ct_path)

    episodes = load_episodes(ct_path)
    deaths = load_brd_deaths(raw_dir)

    rows = [build_row(ep, deaths) for ep in episodes]
    # Prefer the deadliest, then longest, conflicts (fixture cap ordering).
    rows.sort(key=lambda r: (
        -sum(deaths.get((r["entity_id"].split(":")[1].rsplit("-", 1)[0], y), 0)
             for y in range(int(r["start_year"]),
                            (int(r["end_year"]) if r["end_year"]
                             else int(r["start_year"])) + 1)),
        -(int(r["end_year"]) - int(r["start_year"])) if r["end_year"] else -10**9,
        r["entity_id"]))
    dropped = rows[args.max_rows:]
    rows = rows[:args.max_rows]

    # ---- self-validation ----
    ids = [r["entity_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate entity_ids"
    assert all(r["entity_type"] == "event" for r in rows)
    assert all(r["relation"] == "ongoing" for r in rows)
    assert all(r["confidence"] in {"high", "medium", "low"} for r in rows)
    for r in rows:
        assert isinstance(r["start_year"], int) and r["start_year"] >= 1946, r
        assert r["end_year"] == "" or int(r["end_year"]) >= r["start_year"], r
        assert r["entity_id"].startswith("ucdp:"), r
        assert r["canonical_name"] and r["notes"], r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    conf = Counter(r["confidence"] for r in rows)
    ongoing = sum(1 for r in rows if r["end_year"] == "")
    with_deaths = sum(1 for r in rows if "Battle-deaths estimate" in r["notes"])
    print(f"episodes={len(rows)} dropped_over_cap={len(dropped)} "
          f"ongoing(open end_year)={ongoing} with_deaths_estimate={with_deaths} "
          f"confidence: high={conf['high']} medium={conf['medium']} "
          f"low={conf['low']}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)

    if args.spot_check:
        by_id = {r["entity_id"]: r for r in rows}
        print("\nSPOT CHECK (UCDP row vs Wikipedia expectations):",
              file=sys.stderr)
        for eid, label in SPOT_CHECK_IDS.items():
            r = by_id.get(eid)
            if r is None:
                print(f"  {label:48s} ABSENT", file=sys.stderr)
                continue
            print(f"  {label:48s} {eid:16s} "
                  f"{r['start_year']} -> {r['end_year'] or 'ONGOING'}  "
                  f"{r['description'].split(';')[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
