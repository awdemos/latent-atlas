//! Wikidata SPARQL source: query builders, response parsing, resumable fetch.
//! All entity types normalize to one raw shape: ?item ?itemLabel
//! ?itemDescription ?start ?startPrecision ?end ?endPrecision.
//! Wikidata timeValues use astronomical year numbering; we keep the signed
//! integer year verbatim and preserve the raw literal for provenance.

use crate::store::read_ndjson;
use crate::types::EntityType;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// WDQS SPARQL endpoint used when no override is configured.
pub const DEFAULT_ENDPOINT: &str = "https://query.wikidata.org/sparql";
const USER_AGENT: &str =
    "latent-atlas/0.1 (https://vibecodingagency.com; research dataset builder)";

/// One raw entity row as fetched from Wikidata: the normalized shape shared
/// by all entity types (?item ?itemLabel ?itemDescription ?start
/// ?startPrecision ?end ?endPrecision) plus the retrieval date. Date fields
/// keep the raw time literal verbatim for provenance; missing optional
/// values stay `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawRow {
    pub item: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub start_precision: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub end_precision: Option<String>,
    pub retrieved_at: String,
}

fn select_query(pattern: &str, offset: usize, limit: usize) -> String {
    format!(
        "SELECT ?item ?itemLabel ?itemDescription ?start ?startPrecision ?end ?endPrecision WHERE {{\n  {pattern}\n  SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}\n}} ORDER BY ?item\nLIMIT {limit} OFFSET {offset}"
    )
}

fn interval_pattern(instance: &str, p_start: &str, p_end: Option<&str>) -> String {
    let mut s = format!(
        "?item wdt:P31/wdt:P279* {instance}; wdt:{p_start} ?start.\n  ?item p:{p_start} ?s1. ?s1 psv:{p_start} ?v1. ?v1 wikibase:timePrecision ?startPrecision."
    );
    if let Some(pe) = p_end {
        s.push_str(&format!(
            "\n  OPTIONAL {{ ?item wdt:{pe} ?end.\n    ?item p:{pe} ?s2. ?s2 psv:{pe} ?v2. ?v2 wikibase:timePrecision ?endPrecision. }}"
        ));
    }
    s
}

/// Build the paged SELECT query for one entity type, with the type's
/// interval properties (birth/death, inception/dissolution, etc.) bound to
/// the normalized raw shape. Results are ordered by ?item so OFFSET paging
/// is stable across resume.
pub fn entity_query(entity_type: EntityType, offset: usize, limit: usize) -> String {
    let pattern = match entity_type {
        EntityType::Person => interval_pattern("wd:Q5", "P569", Some("P570")),
        EntityType::Event => interval_pattern("wd:Q1656682", "P580", Some("P582")),
        EntityType::Polity => interval_pattern("wd:Q3024240", "P571", Some("P576")),
        EntityType::Organization => interval_pattern("wd:Q43229", "P571", Some("P576")),
        EntityType::Work => interval_pattern("wd:Q7725634", "P577", None),
        EntityType::Technology => interval_pattern("wd:Q11016", "P571", None),
    };
    select_query(&pattern, offset, limit)
}

/// Parse the signed integer year from a Wikidata time literal.
pub fn parse_time_year(lit: &str) -> Option<i32> {
    let t = lit.trim();
    let (sign, rest) = match t.strip_prefix('-') {
        Some(r) => (-1i32, r),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<i32>().ok().map(|y| sign * y)
}

fn bound<'a>(b: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    b.get(key)?.get("value")?.as_str()
}

/// Path of the raw-source manifest next to a raw file:
/// `<raw dir>/source_manifest.json`.
fn source_manifest_path(raw_path: &Path) -> PathBuf {
    raw_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("source_manifest.json")
}

/// Reads the raw-source manifest, treating a missing file as empty.
/// Errors name the manifest path for both read and parse failures.
fn read_source_manifest(path: &Path) -> anyhow::Result<serde_json::Value> {
    if path.exists() {
        serde_json::from_str(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))
    } else {
        Ok(serde_json::json!({"sources": []}))
    }
}

/// Replaces the entry for `entry["entity_type"]` in the manifest's `sources`
/// array (there is exactly one entry per type). Errors when the manifest is
/// not an object with a `sources` array.
fn merge_into_manifest(
    manifest: &mut serde_json::Value,
    entry: serde_json::Value,
    path: &Path,
) -> anyhow::Result<()> {
    let sources = manifest
        .get_mut("sources")
        .and_then(|s| s.as_array_mut())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{}: malformed manifest, expected object with a \"sources\" array",
                path.display()
            )
        })?;
    sources.retain(|s| s["entity_type"] != entry["entity_type"]);
    sources.push(entry);
    Ok(())
}

fn write_source_manifest(path: &Path, manifest: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(manifest)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Map a SPARQL JSON response to raw rows. Bindings without an `item` value
/// are dropped; every kept row is stamped with `retrieved_at`. Returns an
/// empty vec when the response has no `results.bindings` array — this is
/// `fetch_all`'s termination signal.
pub fn parse_bindings(json: &serde_json::Value, retrieved_at: &str) -> Vec<RawRow> {
    let Some(rows) = json
        .get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
    else {
        return vec![];
    };
    rows.iter()
        .filter_map(|b| {
            Some(RawRow {
                item: bound(b, "item")?.to_string(),
                label: bound(b, "itemLabel").unwrap_or_default().to_string(),
                description: bound(b, "itemDescription").unwrap_or_default().to_string(),
                start: bound(b, "start").map(str::to_string),
                start_precision: bound(b, "startPrecision").map(str::to_string),
                end: bound(b, "end").map(str::to_string),
                end_precision: bound(b, "endPrecision").map(str::to_string),
                retrieved_at: retrieved_at.to_string(),
            })
        })
        .collect()
}

/// Rate-limited WDQS client: one HTTP client, an endpoint, and the minimum
/// delay between requests (and backoff unit on retries).
pub struct SparqlClient {
    http: reqwest::Client,
    endpoint: String,
    min_delay: Duration,
}

impl Default for SparqlClient {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlClient {
    /// Client targeting [`DEFAULT_ENDPOINT`], polite ~1 request/1.1s pacing.
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            min_delay: Duration::from_millis(1100),
        }
    }

    /// GET one page of results as SPARQL JSON, retrying transient failures
    /// and non-2xx statuses with linear backoff; gives up after 4 attempts.
    pub async fn fetch_page(&self, query: &str) -> anyhow::Result<serde_json::Value> {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let result = self
                .http
                .get(&self.endpoint)
                .query(&[("query", query), ("format", "json")])
                .header(reqwest::header::USER_AGENT, USER_AGENT)
                .header(reqwest::header::ACCEPT, "application/sparql-results+json")
                .send()
                .await;
            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(resp.json().await?);
                    }
                    if attempt >= 4 {
                        anyhow::bail!("SPARQL HTTP {status} after {attempt} attempts");
                    }
                    tokio::time::sleep(self.min_delay * attempt * 5).await;
                }
                Err(e) => {
                    if attempt >= 4 {
                        return Err(e.into());
                    }
                    tokio::time::sleep(self.min_delay * attempt * 2).await;
                }
            }
        }
    }

    /// Fetch up to `limit` rows into `raw_path` (NDJSON). Resumable: the
    /// existing line count becomes the OFFSET of the next page.
    /// One file per entity type: offsets derive from line counts, so
    /// `raw_path` must contain only rows of `entity_type`.
    pub async fn fetch_all(
        &self,
        entity_type: EntityType,
        limit: usize,
        raw_path: &Path,
    ) -> anyhow::Result<usize> {
        // An interrupted prior fetch may have left a torn final line: drop it
        // before counting, or the count silently skips it while the append
        // welds the first new row onto its bytes (a malformed interior line
        // that then hard-errors on every read).
        crate::store::truncate_partial_tail(raw_path)?;
        let existing = if raw_path.exists() {
            read_ndjson::<RawRow>(raw_path)?.len()
        } else {
            0
        };
        if existing >= limit {
            return Ok(0);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(raw_path)
            .with_context(|| format!("appending to {}", raw_path.display()))?;
        let retrieved_at = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let mut written = 0usize;
        while existing + written < limit {
            let offset = existing + written;
            let take = 1000usize.min(limit - offset);
            let json = self
                .fetch_page(&entity_query(entity_type, offset, take))
                .await?;
            let rows = parse_bindings(&json, &retrieved_at);
            if rows.is_empty() {
                break;
            }
            let n = rows.len();
            for row in &rows {
                serde_json::to_writer(&mut file, row)
                    .with_context(|| format!("appending to {}", raw_path.display()))?;
                file.write_all(b"\n")
                    .with_context(|| format!("appending to {}", raw_path.display()))?;
            }
            written += n;
            if n < take {
                break; // source exhausted
            }
            tokio::time::sleep(self.min_delay).await;
        }
        // update the raw-source manifest
        let manifest_path = source_manifest_path(raw_path);
        let mut manifest = read_source_manifest(&manifest_path)?;
        let entry = serde_json::json!({
            "entity_type": entity_type.as_str(),
            "file": raw_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("{}: no file name", raw_path.display()))?
                .to_string_lossy(),
            "rows_total": existing + written,
            "last_fetch": retrieved_at,
        });
        merge_into_manifest(&mut manifest, entry, &manifest_path)?;
        write_source_manifest(&manifest_path, &manifest)?;
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityType;

    #[test]
    fn query_uses_right_properties_per_type() {
        let q = entity_query(EntityType::Person, 0, 100);
        assert!(q.contains("wdt:P569"), "people need birth date: {q}");
        assert!(q.contains("LIMIT 100"), "{q}");
        assert!(q.contains("OFFSET 0"), "{q}");
        assert!(entity_query(EntityType::Event, 0, 10).contains("wdt:P580"));
        assert!(entity_query(EntityType::Polity, 0, 10).contains("wdt:P571"));
        assert!(entity_query(EntityType::Organization, 0, 10).contains("wdt:P571"));
        assert!(entity_query(EntityType::Work, 0, 10).contains("wdt:P577"));
        assert!(entity_query(EntityType::Person, 2000, 1000).contains("OFFSET 2000"));
    }

    #[test]
    fn parse_time_year_handles_signed_literals() {
        assert_eq!(parse_time_year("-0099-07-13T00:00:00Z"), Some(-99));
        assert_eq!(parse_time_year("+2026-01-01T00:00:00Z"), Some(2026));
        assert_eq!(parse_time_year("1453-05-29T00:00:00Z"), Some(1453));
        assert_eq!(parse_time_year("garbage"), None);
        assert_eq!(parse_time_year(""), None);
    }

    #[test]
    fn parse_bindings_maps_sparql_json() {
        let json = serde_json::json!({"results": {"bindings": [{
            "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1048"},
            "itemLabel": {"type": "literal", "value": "Julius Caesar"},
            "itemDescription": {"type": "literal", "value": "Roman general"},
            "start": {"type": "literal", "value": "-0099-07-13T00:00:00Z"},
            "startPrecision": {"type": "literal", "value": "11"}
        }]}});
        let rows = parse_bindings(&json, "2026-10-02");
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.item, "http://www.wikidata.org/entity/Q1048");
        assert_eq!(r.label, "Julius Caesar");
        assert_eq!(r.start.as_deref(), Some("-0099-07-13T00:00:00Z"));
        assert_eq!(r.start_precision.as_deref(), Some("11"));
        assert_eq!(r.end, None);
        assert_eq!(r.retrieved_at, "2026-10-02");
    }

    #[test]
    fn parse_bindings_empty_response_is_empty_vec() {
        // Empty result set is fetch_all's termination signal; pin it.
        assert!(parse_bindings(&serde_json::json!({}), "2026-10-02").is_empty());
    }

    #[test]
    fn parse_bindings_drops_binding_without_item() {
        let json = serde_json::json!({"results": {"bindings": [
            {"itemLabel": {"type": "literal", "value": "no item uri here"}},
            {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q5"}}
        ]}});
        let rows = parse_bindings(&json, "2026-10-02");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].item, "http://www.wikidata.org/entity/Q5");
    }

    #[test]
    fn query_work_and_technology_have_no_end_property() {
        assert!(!entity_query(EntityType::Work, 0, 10).contains("OPTIONAL"));
        assert!(!entity_query(EntityType::Technology, 0, 10).contains("OPTIONAL"));
    }

    #[test]
    fn fetch_resume_prologue_truncates_torn_tail_before_counting() {
        // fetch_all resumes by counting existing rows, then appends. Without
        // truncation a torn final line would be skipped by the count but then
        // welded onto the first new row as a malformed interior line. Pin the
        // prologue composition: truncate, count, append, re-read.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("persons_raw.ndjson");
        let row = |item: &str| RawRow {
            item: item.into(),
            label: String::new(),
            description: String::new(),
            start: None,
            start_precision: None,
            end: None,
            end_precision: None,
            retrieved_at: "2026-10-03".into(),
        };
        let mut bytes = serde_json::to_string(&row("wd:Q1")).unwrap();
        bytes.push('\n');
        bytes.push_str("{\"item\":\"wd:Q2\",\"retr"); // interrupted mid-row
        std::fs::write(&path, bytes).unwrap();
        crate::store::truncate_partial_tail(&path).unwrap();
        let existing = read_ndjson::<RawRow>(&path).unwrap().len();
        assert_eq!(existing, 1);
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        serde_json::to_writer(&mut f, &row("wd:Q2")).unwrap();
        f.write_all(b"\n").unwrap();
        drop(f);
        let rows = read_ndjson::<RawRow>(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].item, "wd:Q2");
    }

    #[test]
    fn query_orders_by_item_for_stable_resume() {
        for t in [
            EntityType::Person,
            EntityType::Event,
            EntityType::Polity,
            EntityType::Organization,
            EntityType::Work,
            EntityType::Technology,
        ] {
            let q = entity_query(t, 0, 10);
            assert!(
                q.contains("ORDER BY ?item"),
                "resume needs stable order: {q}"
            );
        }
    }
}
