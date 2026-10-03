//! Dataset-root layout and NDJSON I/O.

use anyhow::Context;
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Root directory of a dataset, with the standard layout:
/// `raw/`, `canonical/`, `splits/`, `generated/`, `runs/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetRoot {
    pub path: PathBuf,
}

impl DatasetRoot {
    /// Wraps a path as a dataset root without touching the filesystem.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
    /// Creates the five layout directories (and any missing ancestors).
    pub fn init(&self) -> anyhow::Result<()> {
        for d in [
            self.raw(),
            self.canonical(),
            self.splits(),
            self.generated(),
            self.runs(),
        ] {
            fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
        }
        Ok(())
    }
    /// Path of the `raw/` directory; does not create it.
    pub fn raw(&self) -> PathBuf {
        self.path.join("raw")
    }
    /// Path of the `canonical/` directory; does not create it.
    pub fn canonical(&self) -> PathBuf {
        self.path.join("canonical")
    }
    /// Path of the `splits/` directory; does not create it.
    pub fn splits(&self) -> PathBuf {
        self.path.join("splits")
    }
    /// Path of the `generated/` directory; does not create it.
    pub fn generated(&self) -> PathBuf {
        self.path.join("generated")
    }
    /// Path of the `runs/` directory; does not create it.
    pub fn runs(&self) -> PathBuf {
        self.path.join("runs")
    }
    /// Path of the run directory for `model`; does not create it.
    pub fn run_dir(&self, model: &str) -> PathBuf {
        self.runs().join(model)
    }
    /// Path of the raw file `name`; does not create it.
    pub fn raw_file(&self, name: &str) -> PathBuf {
        self.raw().join(name)
    }
}

/// Writes `rows` as NDJSON: one trailing-newline-terminated JSON value per
/// line. Parent directories of `path` are created automatically.
pub fn write_ndjson<T: Serialize>(path: &Path, rows: &[T]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut w =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    for row in rows {
        serde_json::to_writer(&mut w, row)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}

/// Reads an NDJSON file. Blank and whitespace-only lines are skipped; parse
/// errors carry `path:line:` context naming the 1-based offending line.
pub fn read_ndjson<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let f = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line.with_context(|| format!("reading {}", path.display()))?;
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
    fn write_ndjson_creates_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nested/deep/rows.ndjson");
        let rows = vec![serde_json::json!({"a": 1})];
        write_ndjson(&path, &rows).unwrap();
        let back: Vec<serde_json::Value> = read_ndjson(&path).unwrap();
        assert_eq!(back, rows);
    }

    #[test]
    fn read_ndjson_skips_blank_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sparse.ndjson");
        std::fs::write(&path, "{\"a\":1}\n\n   \n{\"a\":2}\n").unwrap();
        let back: Vec<serde_json::Value> = read_ndjson(&path).unwrap();
        assert_eq!(
            back,
            vec![serde_json::json!({"a": 1}), serde_json::json!({"a": 2})]
        );
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
