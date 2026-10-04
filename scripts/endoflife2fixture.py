#!/usr/bin/env python3
"""Convert endoflife.date API data into a Latent Atlas curated CSV fixture.

Source: https://endoflife.date — community-maintained EOL dates for software,
hardware and services, published under CC BY-SA 4.0 (see
https://endoflife.date/terms/). Requires attribution: cite "endoflife.date"
and link back (done in every row's notes field).

API used (auth-free, no key):
  GET https://endoflife.date/api/all.json        -> list of product slugs
  GET https://endoflife.date/api/<slug>.json     -> cycles for one product

Each cycle entry looks like:
  {"cycle": "3.13", "releaseDate": "2024-10-07", "eol": "2029-10-31",
   "latest": "...", "latestReleaseDate": "...", "lts": false, "support": "..."}
releaseDate / eol are ISO dates ("YYYY-MM-DD"); eol may be `true` (still
supported, no announced EOL) and either field may be absent. All dates here
are CE, so the astronomical-year conversion is the identity; years are taken
from the leading 4 digits of the ISO date.

One row is emitted per product-cycle that has a known releaseDate:
  entity_id     eol:<product-slug>-<cycle-slug>      (lower-case, de-duped)
  relation      active
  end_year      EOL year, or EMPTY when the cycle is still supported
  confidence    high (dates come straight from the source JSON)

Raw downloads are stored under a temp dir OUTSIDE the repo by default
(--raw-dir), never committed. Regenerate with:

  python3 scripts/endoflife2fixture.py \
      --out fixtures/curated_software_endoflife.csv

Rate limiting: the upstream API is hosted on a CDN and comfortably absorbs
one request per product; we add a small delay between requests anyway.
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

API_BASE = "https://endoflife.date/api"
RETRIEVED = "2026-10-04"  # fixture retrieval date; update when regenerating
DEFAULT_OUT = "fixtures/curated_software_endoflife.csv"
MAX_ROWS = 1000

# Products most useful as LLM temporal probes, in priority order: their cycles
# are emitted first, so they always survive the --max-rows cap.
PRIORITY = [
    "python", "nodejs", "ubuntu", "windows", "windows-server", "postgresql",
    "mysql", "mariadb", "kubernetes", "docker-engine", "nginx", "openssl",
    "go", "rust", "java", "php", "ruby", "rails", "perl", "lua", "julia",
    "redis", "mongodb", "sqlite", "deno", "bun", "dotnet", "dotnetfx",
    "react", "vue", "angular", "django", "laravel", "spring-boot",
    "wordpress", "drupal", "joomla", "firefox", "chrome", "internet-explorer",
    "android", "ios", "macos", "linux", "debian", "fedora", "centos", "rhel",
    "alpine-linux", "freebsd", "openbsd", "oracle-jdk", "eclipse-temurin",
    "apache-http-server", "apache-kafka", "apache-cassandra", "rabbitmq",
    "gitlab", "jenkins", "terraform", "ansible", "puppet", "prometheus",
    "grafana", "electron", "qt", "libreoffice", "virtualbox", "powershell",
    "visual-studio", "office", "msexchange", "mssqlserver", "bitcoin-core",
    "haproxy", "traefik", "envoy", "istio", "podman", "containerd",
    "kotlin", "scala", "ghc", "erlang", "elixir", "tomcat", "maven",
    "gradle", "openssl",
]

HEADER = [
    "entity_id", "entity_type", "canonical_name", "aliases", "description",
    "relation", "start_year", "end_year", "confidence", "notes",
]


def fetch(url: str, dest: Path) -> None:
    """Download url to dest, retrying a couple of times on failure.

    Existing non-empty files are reused so regeneration from a kept raw dir
    does not re-hit the API.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "latent-atlas-fixture/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as err:  # noqa: BLE001 - report whatever broke
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def slugify_cycle(cycle: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(cycle).lower()).strip("-")
    return s or "x"


def year_of(value):
    """Leading 4-digit year of an ISO date string, else None."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^(\d{4})-\d{2}-\d{2}$", value)
    return int(m.group(1)) if m else None


def build_rows(raw_dir: Path, priority: list[str], max_rows: int):
    slugs_path = raw_dir / "all.json"
    fetch(f"{API_BASE}/all.json", slugs_path)
    all_slugs = json.loads(slugs_path.read_text())
    known = set(all_slugs)

    order = [s for s in dict.fromkeys(priority) if s in known]
    order += [s for s in all_slugs if s not in set(order)]

    rows = []
    seen_ids = set()
    dropped = []
    for slug in order:
        if len(rows) >= max_rows:
            break
        prod_path = raw_dir / f"{slug}.json"
        fetch(f"{API_BASE}/{slug}.json", prod_path)
        cycles = json.loads(prod_path.read_text())
        for c in cycles:
            if len(rows) >= max_rows:
                return rows, dropped
            if not isinstance(c, dict):
                continue
            cycle = c.get("cycle")
            release_year = year_of(c.get("releaseDate"))
            if cycle is None or release_year is None:
                dropped.append((slug, cycle, "no releaseDate"))
                continue
            entity_id = f"eol:{slug}-{slugify_cycle(cycle)}"
            if entity_id in seen_ids:
                dropped.append((slug, cycle, "duplicate entity_id after slugify"))
                continue
            eol_year = year_of(c.get("eol"))
            still_supported = c.get("eol") is True or eol_year is None
            seen_ids.add(entity_id)
            eol_note = (
                f"EOL {c['eol']}" if eol_year is not None
                else "still supported (no announced EOL)" if c.get("eol") is True
                else "EOL date unknown"
            )
            label = str(c.get("releaseLabel") or cycle)
            rows.append({
                "entity_id": entity_id,
                "entity_type": "technology",
                "canonical_name": f"{slug.replace('-', ' ').title()} {label}",
                "aliases": "",
                "description": (
                    f"Release cycle {label} of {slug.replace('-', ' ')}, "
                    "tracked by endoflife.date"
                ),
                "relation": "active",
                "start_year": release_year,
                "end_year": eol_year if eol_year is not None else "",
                "confidence": "high",
                "notes": (
                    f"Source: endoflife.date API ({API_BASE}/{slug}.json), "
                    f"product {slug}, cycle {cycle}; {eol_note}; "
                    f"retrieved {RETRIEVED}. Data CC BY-SA 4.0, "
                    "attribution: endoflife.date."
                ),
            })
        time.sleep(0.2)
    return rows, dropped


def write_csv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=None,
                    help="dir for raw API downloads (default: fresh temp dir)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(tempfile.mkdtemp(prefix="eol-raw-"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, dropped = build_rows(raw_dir, PRIORITY, args.max_rows)
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out} (raw data in {raw_dir})")
    if dropped:
        print(f"dropped {len(dropped)} cycles without a known release date:")
        for slug, cycle, why in dropped[:20]:
            print(f"  {slug} / {cycle}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
