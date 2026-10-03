//! Raw source records -> canonical entities/intervals/provenance.
//! Precision honesty rule: rows whose temporal evidence is coarser than a
//! decade are dropped rather than given fake yearly intervals.

use crate::source::wikidata::{RawRow, parse_time_year};
use crate::store::{DatasetRoot, read_ndjson};
use crate::types::*;
use anyhow::Context;
use std::collections::BTreeMap;
use std::path::Path;

/// Extract the QID from a Wikidata entity URI
/// (`http://www.wikidata.org/entity/Q1048` -> `Q1048`).
pub fn qid_from_uri(uri: &str) -> Option<&str> {
    uri.rsplit('/').next().filter(|s| s.starts_with('Q'))
}

/// Map a Wikidata timePrecision code (11=day ... 6=millennium) to
/// [`YearPrecision`]; missing/unparseable codes become `Unknown`.
pub fn precision_from_code(code: Option<&str>) -> YearPrecision {
    match code.and_then(|c| c.parse::<u32>().ok()) {
        Some(11) => YearPrecision::Day,
        Some(10) => YearPrecision::Month,
        Some(9) => YearPrecision::Year,
        Some(8) => YearPrecision::Decade,
        Some(7) => YearPrecision::Century,
        Some(6) => YearPrecision::Millennium,
        _ => YearPrecision::Unknown,
    }
}

/// The canonical relation each entity type contributes to the dataset.
pub fn relation_for(t: EntityType) -> Relation {
    match t {
        EntityType::Person => Relation::Alive,
        EntityType::Event => Relation::Ongoing,
        EntityType::Polity => Relation::Exists,
        EntityType::Organization => Relation::Active,
        EntityType::Work | EntityType::Technology => Relation::Available,
    }
}

fn date_basis_for(t: EntityType) -> &'static str {
    match t {
        EntityType::Person => "birth_death",
        EntityType::Event => "start_end",
        EntityType::Polity | EntityType::Organization => "inception_dissolution",
        EntityType::Work | EntityType::Technology => "publication",
    }
}

/// Raw NDJSON file stem for an entity type, matching `fetch`'s output names.
pub fn raw_file_stem(t: EntityType) -> &'static str {
    match t {
        EntityType::Person => "wikidata_people",
        EntityType::Event => "wikidata_events",
        EntityType::Polity => "wikidata_polities",
        EntityType::Organization => "wikidata_organizations",
        EntityType::Work => "wikidata_works",
        EntityType::Technology => "wikidata_technologies",
    }
}

fn provenance(
    entity_id: &str,
    qid: &str,
    field: &str,
    value: serde_json::Value,
    retrieved_at: &str,
    source_name: &str,
) -> Provenance {
    Provenance {
        entity_id: entity_id.to_string(),
        field: field.to_string(),
        value,
        source_name: source_name.to_string(),
        source_id: qid.to_string(),
        retrieved_at: retrieved_at.to_string(),
        source_statement_id: None,
        human_review_status: "unreviewed".to_string(),
    }
}

/// Normalize one raw Wikidata row into an entity, its interval, and
/// per-bound provenance. Returns `None` when the row has no usable start
/// year or either bound's precision is coarser than a decade.
pub fn normalize_row(
    entity_type: EntityType,
    row: &RawRow,
) -> Option<(Entity, Interval, Vec<Provenance>)> {
    let qid = qid_from_uri(&row.item)?.to_string();
    let start_year = row.start.as_deref().and_then(parse_time_year)?;
    let mut end_year = row.end.as_deref().and_then(parse_time_year);
    let start_precision = precision_from_code(row.start_precision.as_deref());
    let mut end_precision = if row.end.is_some() {
        precision_from_code(row.end_precision.as_deref())
    } else {
        YearPrecision::Unknown
    };

    // Honesty rule: evidence coarser than a decade is dropped.
    for (year, prec) in [
        (Some(start_year), start_precision),
        (end_year, end_precision),
    ] {
        if year.is_some() && prec.ambiguity_window() > YearPrecision::Decade.ambiguity_window() {
            return None;
        }
    }
    // Point events: missing end means end == start.
    if entity_type == EntityType::Event && end_year.is_none() {
        end_year = Some(start_year);
        end_precision = start_precision;
    }

    let entity_id = format!("wd:{qid}");
    let entity = Entity {
        entity_id: entity_id.clone(),
        entity_type,
        canonical_name: row.label.clone(),
        aliases: vec![],
        description: row.description.clone(),
        language: "en".into(),
        source_id: format!("wikidata:{qid}"),
        source_url: row.item.clone(),
    };
    let relation = relation_for(entity_type);
    let interval = Interval {
        interval_id: format!("{}:{entity_id}", relation.as_str()),
        entity_id: entity_id.clone(),
        relation,
        start_year: Some(start_year),
        end_year,
        start_precision,
        end_precision,
        start_inclusive: true,
        end_inclusive: true,
        confidence: Confidence::High,
        date_basis: date_basis_for(entity_type).into(),
        source_id: format!("wikidata:{qid}"),
        notes: String::new(),
    };
    let mut prov = vec![provenance(
        &entity_id,
        &qid,
        "start_year",
        serde_json::json!({"year": start_year, "raw": row.start, "precision": row.start_precision}),
        &row.retrieved_at,
        "Wikidata",
    )];
    if let Some(e) = end_year {
        prov.push(provenance(
            &entity_id,
            &qid,
            "end_year",
            serde_json::json!({"year": e, "raw": row.end, "precision": row.end_precision}),
            &row.retrieved_at,
            "Wikidata",
        ));
    }
    Some((entity, interval, prov))
}

/// curated_roman.csv columns:
/// entity_id,entity_type,canonical_name,aliases,description,relation,start_year,end_year,confidence,notes
/// aliases are '|'-separated; empty year = open bound. entity_type, relation,
/// and confidence must all be valid enum strings; invalid values are errors
/// naming the file and line.
pub fn parse_curated_csv(path: &Path) -> anyhow::Result<Vec<(Entity, Interval, Vec<Provenance>)>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("reading curated CSV {}", path.display()))?;
    let retrieved_at = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec.with_context(|| format!("reading record from {}", path.display()))?;
        let pos = rec.position().map(|p| p.line()).unwrap_or(0);
        let get = |i: usize| rec.get(i).unwrap_or("").trim();
        let entity_type = EntityType::from_str(get(1)).ok_or_else(|| {
            anyhow::anyhow!(
                "{}: line {pos}: bad entity_type {:?}",
                path.display(),
                get(1)
            )
        })?;
        let relation = Relation::from_str(get(5)).ok_or_else(|| {
            anyhow::anyhow!("{}: line {pos}: bad relation {:?}", path.display(), get(5))
        })?;
        let year = |i: usize, col: &str| -> anyhow::Result<Option<i32>> {
            if get(i).is_empty() {
                Ok(None)
            } else {
                get(i).parse::<i32>().map(Some).map_err(|e| {
                    anyhow::anyhow!(
                        "{}: line {pos}: bad {col} {:?}: {e}",
                        path.display(),
                        get(i)
                    )
                })
            }
        };
        let entity_id = get(0).to_string();
        let entity = Entity {
            entity_id: entity_id.clone(),
            entity_type,
            canonical_name: get(2).to_string(),
            aliases: get(3)
                .split('|')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            description: get(4).to_string(),
            language: "en".into(),
            source_id: format!("curated:{entity_id}"),
            source_url: String::new(),
        };
        let interval = Interval {
            interval_id: format!("{}:{entity_id}", relation.as_str()),
            entity_id: entity_id.clone(),
            relation,
            start_year: year(6, "start_year")?,
            end_year: year(7, "end_year")?,
            start_precision: YearPrecision::Year,
            end_precision: YearPrecision::Year,
            start_inclusive: true,
            end_inclusive: true,
            confidence: Confidence::from_str(get(8)).ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: line {pos}: bad confidence {:?}",
                    path.display(),
                    get(8)
                )
            })?,
            date_basis: "curated".into(),
            source_id: format!("curated:{entity_id}"),
            notes: get(9).to_string(),
        };
        let prov = vec![provenance(
            &entity_id,
            &entity_id,
            "interval",
            serde_json::json!({"start": interval.start_year, "end": interval.end_year}),
            &retrieved_at,
            "curated",
        )];
        out.push((entity, interval, prov));
    }
    Ok(out)
}

/// Merge all available raw sources + the curated CSV into canonical tables.
/// A Wikidata entity can carry multiple date statements producing duplicate
/// interval_ids; v0 keeps the first deterministically to preserve the
/// one-interval-per-id invariant downstream joins rely on.
pub fn normalize_all(
    root: &DatasetRoot,
) -> anyhow::Result<(Vec<Entity>, Vec<Interval>, Vec<Provenance>)> {
    let mut entities: BTreeMap<String, Entity> = BTreeMap::new();
    let mut intervals: BTreeMap<String, Interval> = BTreeMap::new();
    let mut provenance = Vec::new();
    let mut push = |e: Entity, iv: Interval, prov: Vec<Provenance>| {
        entities.entry(e.entity_id.clone()).or_insert(e);
        intervals.entry(iv.interval_id.clone()).or_insert(iv);
        provenance.extend(prov);
    };
    for t in [
        EntityType::Person,
        EntityType::Event,
        EntityType::Polity,
        EntityType::Organization,
        EntityType::Work,
        EntityType::Technology,
    ] {
        let path = root.raw_file(&format!("{}.jsonl", raw_file_stem(t)));
        if !path.exists() {
            continue;
        }
        for row in read_ndjson::<RawRow>(&path)? {
            if let Some((e, iv, p)) = normalize_row(t, &row) {
                push(e, iv, p);
            }
        }
    }
    let curated = root.raw_file("curated_roman.csv");
    if curated.exists() {
        for (e, iv, p) in parse_curated_csv(&curated)? {
            push(e, iv, p);
        }
    }
    Ok((
        entities.into_values().collect(),
        intervals.into_values().collect(),
        provenance,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    // types come via `super::*`; a second glob here trips unused_imports.
    use crate::source::wikidata::RawRow;

    fn raw(
        qid_url: &str,
        start: Option<&str>,
        sp: Option<&str>,
        end: Option<&str>,
        ep: Option<&str>,
    ) -> RawRow {
        RawRow {
            item: qid_url.into(),
            label: "X".into(),
            description: "d".into(),
            start: start.map(Into::into),
            start_precision: sp.map(Into::into),
            end: end.map(Into::into),
            end_precision: ep.map(Into::into),
            retrieved_at: "2026-10-02".into(),
        }
    }

    #[test]
    fn person_maps_to_alive_interval() {
        let row = raw(
            "http://www.wikidata.org/entity/Q1048",
            Some("-0099-07-13T00:00:00Z"),
            Some("11"),
            Some("-0043-03-15T00:00:00Z"),
            Some("9"),
        );
        let (e, iv, prov) = normalize_row(EntityType::Person, &row).unwrap();
        assert_eq!(e.entity_id, "wd:Q1048");
        assert_eq!(iv.relation, Relation::Alive);
        assert_eq!(iv.start_year, Some(-99));
        assert_eq!(iv.end_year, Some(-43));
        assert_eq!(iv.start_precision, YearPrecision::Day);
        assert_eq!(iv.date_basis, "birth_death");
        assert_eq!(prov.len(), 2);
    }

    #[test]
    fn event_without_end_becomes_point_interval() {
        let row = raw(
            "http://www.wikidata.org/entity/Q1",
            Some("1066-10-14T00:00:00Z"),
            Some("11"),
            None,
            None,
        );
        let (_, iv, _) = normalize_row(EntityType::Event, &row).unwrap();
        assert_eq!(iv.end_year, Some(1066));
        assert_eq!(iv.relation, Relation::Ongoing);
    }

    #[test]
    fn century_precision_rows_are_dropped() {
        let row = raw(
            "http://www.wikidata.org/entity/Q2",
            Some("0200-01-01T00:00:00Z"),
            Some("7"),
            None,
            None,
        );
        assert!(normalize_row(EntityType::Polity, &row).is_none());
    }

    #[test]
    fn work_maps_to_open_ended_available() {
        let row = raw(
            "http://www.wikidata.org/entity/Q60272",
            Some("-0018-01-01T00:00:00Z"),
            Some("9"),
            None,
            None,
        );
        let (_, iv, _) = normalize_row(EntityType::Work, &row).unwrap();
        assert_eq!(iv.relation, Relation::Available);
        assert_eq!(iv.end_year, None);
        assert_eq!(iv.label_at(100), GoldLabel::Yes);
    }

    #[test]
    fn curated_csv_parses() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("curated_roman.csv");
        std::fs::write(
            &path,
            concat!(
                "entity_id,entity_type,canonical_name,aliases,description,relation,start_year,end_year,confidence,notes\n",
                "curated:roman_republic,polity,Roman Republic,Res Publica,Roman state from 509 BCE,exists,-508,-26,high,Conventional dates\n",
                "curated:aeneid,work,Aeneid,,Epic by Virgil,available,-18,,medium,Open-ended\n",
            ),
        )
        .unwrap();
        let rows = parse_curated_csv(&path).unwrap();
        assert_eq!(rows.len(), 2);
        let (e, iv, prov) = &rows[0];
        assert_eq!(e.aliases, vec!["Res Publica"]);
        assert_eq!(iv.start_year, Some(-508));
        assert_eq!(iv.end_year, Some(-26));
        assert_eq!(prov[0].source_name, "curated");
        assert_eq!(rows[1].1.end_year, None);
    }

    #[test]
    fn unknown_precision_rows_are_dropped() {
        // No precision code -> Unknown -> ambiguity window i32::MAX > Decade.
        let row = raw(
            "http://www.wikidata.org/entity/Q3",
            Some("1066-10-14T00:00:00Z"),
            None,
            None,
            None,
        );
        assert!(normalize_row(EntityType::Event, &row).is_none());
    }

    #[test]
    fn decade_precision_is_kept() {
        let row = raw(
            "http://www.wikidata.org/entity/Q4",
            Some("1066-10-14T00:00:00Z"),
            Some("8"),
            None,
            None,
        );
        let (_, iv, _) = normalize_row(EntityType::Event, &row).unwrap();
        assert_eq!(iv.start_precision, YearPrecision::Decade);
    }

    #[test]
    fn coarse_end_precision_drops_row() {
        let row = raw(
            "http://www.wikidata.org/entity/Q5",
            Some("1066-10-14T00:00:00Z"),
            Some("9"),
            Some("1067-01-01T00:00:00Z"),
            Some("7"),
        );
        assert!(normalize_row(EntityType::Event, &row).is_none());
    }

    #[test]
    fn bad_confidence_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("curated_roman.csv");
        std::fs::write(
            &path,
            concat!(
                "entity_id,entity_type,canonical_name,aliases,description,relation,start_year,end_year,confidence,notes\n",
                "curated:x,polity,X,,d,exists,-508,-26,hihg,typo\n",
            ),
        )
        .unwrap();
        let err = parse_curated_csv(&path).unwrap_err();
        assert!(err.to_string().contains("bad confidence"), "{err}");
    }

    #[test]
    fn duplicate_interval_ids_dedup_first_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let root = DatasetRoot::new(tmp.path());
        root.init().unwrap();
        let line = |birth: &str| {
            serde_json::json!({
                "item": "http://www.wikidata.org/entity/Q1048",
                "label": "X",
                "description": "d",
                "start": birth,
                "start_precision": "9",
                "retrieved_at": "2026-10-02",
            })
        };
        std::fs::write(
            root.raw_file("wikidata_people.jsonl"),
            format!(
                "{}\n{}\n",
                line("-0099-07-13T00:00:00Z"),
                line("-0098-07-13T00:00:00Z")
            ),
        )
        .unwrap();
        let (entities, intervals, _) = normalize_all(&root).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(intervals.len(), 1);
        // read_ndjson preserves file order; the first row's year survives.
        assert_eq!(intervals[0].start_year, Some(-99));
    }
}
