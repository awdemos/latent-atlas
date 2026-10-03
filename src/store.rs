//! Dataset-root layout and NDJSON I/O.

use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct DatasetRoot {
    pub path: PathBuf,
}

impl DatasetRoot {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
    pub fn init(&self) -> std::io::Result<()> {
        for d in [
            self.raw(),
            self.canonical(),
            self.splits(),
            self.generated(),
            self.runs(),
        ] {
            fs::create_dir_all(d)?;
        }
        Ok(())
    }
    pub fn raw(&self) -> PathBuf {
        self.path.join("raw")
    }
    pub fn canonical(&self) -> PathBuf {
        self.path.join("canonical")
    }
    pub fn splits(&self) -> PathBuf {
        self.path.join("splits")
    }
    pub fn generated(&self) -> PathBuf {
        self.path.join("generated")
    }
    pub fn runs(&self) -> PathBuf {
        self.path.join("runs")
    }
    pub fn run_dir(&self, model: &str) -> PathBuf {
        self.runs().join(model)
    }
    pub fn raw_file(&self, name: &str) -> PathBuf {
        self.raw().join(name)
    }
}

pub fn write_ndjson<T: Serialize>(path: &Path, rows: &[T]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    for row in rows {
        serde_json::to_writer(&mut w, row)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

pub fn read_ndjson<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), i + 1))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let root = DatasetRoot::new(tmp.path());
        root.init().unwrap();
        for d in ["raw", "canonical", "splits", "generated", "runs"] {
            assert!(tmp.path().join(d).is_dir(), "missing {d}");
        }
        assert_eq!(root.run_dir("gpt-x"), tmp.path().join("runs/gpt-x"));
    }

    #[test]
    fn ndjson_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rows.ndjson");
        let rows = vec![serde_json::json!({"a": 1}), serde_json::json!({"a": 2})];
        write_ndjson(&path, &rows).unwrap();
        let back: Vec<serde_json::Value> = read_ndjson(&path).unwrap();
        assert_eq!(back, rows);
    }

    #[test]
    fn read_ndjson_reports_line_number() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.ndjson");
        std::fs::write(&path, "{\"a\":1}\nnot json\n").unwrap();
        let err = read_ndjson::<serde_json::Value>(&path)
            .unwrap_err()
            .to_string();
        assert!(err.contains(":2:"), "error should name line 2: {err}");
    }
}
