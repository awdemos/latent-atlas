//! Parquet persistence for the canonical layer.
//! Enums are stored as UTF-8 (as_str/from_str); Vec<String> aliases and the
//! provenance `value` are stored as JSON strings in UTF-8 columns.

use crate::types::{Confidence, Entity, EntityType, Interval, Provenance, Relation, YearPrecision};
use arrow::array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn strs(it: impl Iterator<Item = String>) -> ArrayRef {
    Arc::new(StringArray::from(it.collect::<Vec<_>>()))
}

fn write_batch(path: &Path, batch: RecordBatch) -> anyhow::Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn read_batches(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    Ok(reader.collect::<Result<Vec<_>, _>>()?)
}

fn col_str(batch: &RecordBatch, i: usize) -> Vec<String> {
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("utf8 column");
    (0..a.len()).map(|r| a.value(r).to_string()).collect()
}

fn col_opt_str(batch: &RecordBatch, i: usize) -> Vec<Option<String>> {
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("utf8 column");
    (0..a.len())
        .map(|r| {
            if a.is_null(r) {
                None
            } else {
                Some(a.value(r).to_string())
            }
        })
        .collect()
}

fn col_opt_i32(batch: &RecordBatch, i: usize) -> Vec<Option<i32>> {
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("i32 column");
    (0..a.len())
        .map(|r| if a.is_null(r) { None } else { Some(a.value(r)) })
        .collect()
}

fn col_bool(batch: &RecordBatch, i: usize) -> Vec<bool> {
    let a = batch
        .column(i)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("bool column");
    (0..a.len()).map(|r| a.value(r)).collect()
}

fn parse_enum<T>(name: &str, s: &str, f: impl Fn(&str) -> Option<T>) -> anyhow::Result<T> {
    f(s).ok_or_else(|| anyhow::anyhow!("bad {name} value: {s:?}"))
}

pub fn write_entities(path: &Path, rows: &[Entity]) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        utf8("entity_id"),
        utf8("entity_type"),
        utf8("canonical_name"),
        utf8("aliases"),
        utf8("description"),
        utf8("language"),
        utf8("source_id"),
        utf8("source_url"),
    ]));
    let cols: Vec<ArrayRef> = vec![
        strs(rows.iter().map(|e| e.entity_id.clone())),
        strs(rows.iter().map(|e| e.entity_type.as_str().to_string())),
        strs(rows.iter().map(|e| e.canonical_name.clone())),
        strs(
            rows.iter()
                .map(|e| serde_json::to_string(&e.aliases).unwrap()),
        ),
        strs(rows.iter().map(|e| e.description.clone())),
        strs(rows.iter().map(|e| e.language.clone())),
        strs(rows.iter().map(|e| e.source_id.clone())),
        strs(rows.iter().map(|e| e.source_url.clone())),
    ];
    write_batch(path, RecordBatch::try_new(schema, cols)?)
}

// Column-major decode: the row index r walks all 8 column vectors at once,
// so the range loop is clearer than any single-slice iteration.
#[allow(clippy::needless_range_loop)]
pub fn read_entities(path: &Path) -> anyhow::Result<Vec<Entity>> {
    let mut out = Vec::new();
    for batch in read_batches(path)? {
        let c: Vec<Vec<String>> = (0..8).map(|i| col_str(&batch, i)).collect();
        for r in 0..batch.num_rows() {
            out.push(Entity {
                entity_id: c[0][r].clone(),
                entity_type: parse_enum("entity_type", &c[1][r], EntityType::from_str)?,
                canonical_name: c[2][r].clone(),
                aliases: serde_json::from_str(&c[3][r])?,
                description: c[4][r].clone(),
                language: c[5][r].clone(),
                source_id: c[6][r].clone(),
                source_url: c[7][r].clone(),
            });
        }
    }
    Ok(out)
}

pub fn write_intervals(path: &Path, rows: &[Interval]) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        utf8("interval_id"),
        utf8("entity_id"),
        utf8("relation"),
        Field::new("start_year", DataType::Int32, true),
        Field::new("end_year", DataType::Int32, true),
        utf8("start_precision"),
        utf8("end_precision"),
        Field::new("start_inclusive", DataType::Boolean, false),
        Field::new("end_inclusive", DataType::Boolean, false),
        utf8("confidence"),
        utf8("date_basis"),
        utf8("source_id"),
        utf8("notes"),
    ]));
    let cols: Vec<ArrayRef> = vec![
        strs(rows.iter().map(|i| i.interval_id.clone())),
        strs(rows.iter().map(|i| i.entity_id.clone())),
        strs(rows.iter().map(|i| i.relation.as_str().to_string())),
        Arc::new(Int32Array::from(
            rows.iter().map(|i| i.start_year).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|i| i.end_year).collect::<Vec<_>>(),
        )),
        strs(rows.iter().map(|i| i.start_precision.as_str().to_string())),
        strs(rows.iter().map(|i| i.end_precision.as_str().to_string())),
        Arc::new(BooleanArray::from(
            rows.iter().map(|i| i.start_inclusive).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|i| i.end_inclusive).collect::<Vec<_>>(),
        )),
        strs(rows.iter().map(|i| i.confidence.as_str().to_string())),
        strs(rows.iter().map(|i| i.date_basis.clone())),
        strs(rows.iter().map(|i| i.source_id.clone())),
        strs(rows.iter().map(|i| i.notes.clone())),
    ];
    write_batch(path, RecordBatch::try_new(schema, cols)?)
}

pub fn read_intervals(path: &Path) -> anyhow::Result<Vec<Interval>> {
    let mut out = Vec::new();
    for batch in read_batches(path)? {
        let starts = col_opt_i32(&batch, 3);
        let ends = col_opt_i32(&batch, 4);
        let inc_s = col_bool(&batch, 7);
        let inc_e = col_bool(&batch, 8);
        let c: Vec<Vec<String>> = [0, 1, 2, 5, 6, 9, 10, 11, 12]
            .map(|i| col_str(&batch, i))
            .into_iter()
            .collect();
        for r in 0..batch.num_rows() {
            out.push(Interval {
                interval_id: c[0][r].clone(),
                entity_id: c[1][r].clone(),
                relation: parse_enum("relation", &c[2][r], Relation::from_str)?,
                start_year: starts[r],
                end_year: ends[r],
                start_precision: parse_enum("start_precision", &c[3][r], YearPrecision::from_str)?,
                end_precision: parse_enum("end_precision", &c[4][r], YearPrecision::from_str)?,
                start_inclusive: inc_s[r],
                end_inclusive: inc_e[r],
                confidence: parse_enum("confidence", &c[5][r], Confidence::from_str)?,
                date_basis: c[6][r].clone(),
                source_id: c[7][r].clone(),
                notes: c[8][r].clone(),
            });
        }
    }
    Ok(out)
}

pub fn write_provenance(path: &Path, rows: &[Provenance]) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        utf8("entity_id"),
        utf8("field"),
        utf8("value"),
        utf8("source_name"),
        utf8("source_id"),
        utf8("retrieved_at"),
        Field::new("source_statement_id", DataType::Utf8, true),
        utf8("human_review_status"),
    ]));
    let cols: Vec<ArrayRef> = vec![
        strs(rows.iter().map(|p| p.entity_id.clone())),
        strs(rows.iter().map(|p| p.field.clone())),
        strs(rows.iter().map(|p| p.value.to_string())),
        strs(rows.iter().map(|p| p.source_name.clone())),
        strs(rows.iter().map(|p| p.source_id.clone())),
        strs(rows.iter().map(|p| p.retrieved_at.clone())),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|p| p.source_statement_id.as_deref())
                .collect::<Vec<_>>(),
        )),
        strs(rows.iter().map(|p| p.human_review_status.clone())),
    ];
    write_batch(path, RecordBatch::try_new(schema, cols)?)
}

pub fn read_provenance(path: &Path) -> anyhow::Result<Vec<Provenance>> {
    let mut out = Vec::new();
    for batch in read_batches(path)? {
        let stmt = col_opt_str(&batch, 6);
        let c: Vec<Vec<String>> = [0, 1, 2, 3, 4, 5, 7]
            .map(|i| col_str(&batch, i))
            .into_iter()
            .collect();
        for r in 0..batch.num_rows() {
            out.push(Provenance {
                entity_id: c[0][r].clone(),
                field: c[1][r].clone(),
                value: serde_json::from_str(&c[2][r])?,
                source_name: c[3][r].clone(),
                source_id: c[4][r].clone(),
                retrieved_at: c[5][r].clone(),
                source_statement_id: stmt[r].clone(),
                human_review_status: c[6][r].clone(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tests_helpers::*;

    #[test]
    fn entities_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("entities.parquet");
        let rows = vec![caesar_entity(), aeneid_entity()];
        write_entities(&path, &rows).unwrap();
        assert_eq!(read_entities(&path).unwrap(), rows);
    }

    #[test]
    fn intervals_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("intervals.parquet");
        let rows = vec![caesar_interval(), aeneid_interval()];
        write_intervals(&path, &rows).unwrap();
        assert_eq!(read_intervals(&path).unwrap(), rows);
    }

    #[test]
    fn provenance_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("provenance.parquet");
        let rows = vec![caesar_provenance()];
        write_provenance(&path, &rows).unwrap();
        assert_eq!(read_provenance(&path).unwrap(), rows);
    }

    #[test]
    fn empty_table_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.parquet");
        write_entities(&path, &[]).unwrap();
        assert_eq!(read_entities(&path).unwrap(), vec![]);
    }
}
