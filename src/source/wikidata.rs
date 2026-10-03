//! Wikidata SPARQL source: query builders, response parsing, resumable fetch.
//! All entity types normalize to one raw shape: ?item ?itemLabel
//! ?itemDescription ?start ?startPrecision ?end ?endPrecision.
//! Wikidata timeValues use astronomical year numbering; we keep the signed
//! integer year verbatim and preserve the raw literal for provenance.

use crate::store::read_ndjson;
use crate::types::EntityType;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_ENDPOINT: &str = "https://query.wikidata.org/sparql";
const USER_AGENT: &str =
    "latent-atlas/0.1 (https://vibecodingagency.com; research dataset builder)";

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
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            min_delay: Duration::from_millis(1100),
        }
    }

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
    pub async fn fetch_all(
        &self,
        entity_type: EntityType,
        limit: usize,
        raw_path: &Path,
    ) -> anyhow::Result<usize> {
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
            .open(raw_path)?;
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
                serde_json::to_writer(&mut file, row)?;
                file.write_all(b"\n")?;
            }
            written += n;
            if n < take {
                break; // source exhausted
            }
            tokio::time::sleep(self.min_delay).await;
        }
        // update the raw-source manifest
        let manifest_path = raw_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("source_manifest.json");
        let mut manifest: serde_json::Value = if manifest_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?
        } else {
            serde_json::json!({"sources": []})
        };
        let entry = serde_json::json!({
            "entity_type": entity_type.as_str(),
            "file": raw_path.file_name().unwrap().to_string_lossy(),
            "rows_total": existing + written,
            "last_fetch": retrieved_at,
        });
        let sources = manifest["sources"].as_array_mut().unwrap();
        sources.retain(|s| s["entity_type"] != entry["entity_type"]);
        sources.push(entry);
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
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
}
