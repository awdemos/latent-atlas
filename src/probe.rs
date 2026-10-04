//! Probe runner: examples in, denormalized responses out.
//! Resumable by example_id; bounded concurrency; append-only output.

use crate::model::ModelClient;
use crate::store::{ensure_parent, read_ndjson, truncate_partial_tail};
use crate::types::{Example, Response};
use anyhow::Context;
use futures_util::StreamExt;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Score every example not already in `out_path` and append denormalized
/// responses as NDJSON; returns the number of rows written this run.
///
/// - Output order is nondeterministic (`buffer_unordered` completion order).
/// - `concurrency` must be >= 1 (validated up front; 0 would hang forever).
/// - Resume dedups on `example_id` only: pointing this at another model's
///   response file will skip everything.
/// - A kill mid-flush can leave a torn final line; resume truncates the
///   partial tail before appending, so torn lines never become interior.
pub async fn run_probe(
    client: &dyn ModelClient,
    examples: Vec<Example>,
    out_path: &Path,
    concurrency: usize,
) -> anyhow::Result<usize> {
    anyhow::ensure!(
        concurrency >= 1,
        "concurrency must be >= 1, got {concurrency}"
    );
    truncate_partial_tail(out_path)?;
    let done: HashSet<String> = if out_path.exists() {
        read_ndjson::<Response>(out_path)?
            .into_iter()
            .map(|r| r.example.example_id)
            .collect()
    } else {
        HashSet::new()
    };
    let todo: Vec<Example> = examples
        .into_iter()
        .filter(|e| !done.contains(&e.example_id))
        .collect();
    if todo.is_empty() {
        return Ok(0);
    }
    ensure_parent(out_path)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .with_context(|| format!("opening {}", out_path.display()))?;
    let mut file = std::io::BufWriter::new(file);
    let model = client.name().to_string();
    let mut written = 0usize;

    let mut stream = futures_util::stream::iter(todo.into_iter().map(|ex| async move {
        let start = Instant::now();
        let score = client.score(&ex).await;
        (ex, score, start.elapsed().as_millis() as u64)
    }))
    .buffer_unordered(concurrency);

    while let Some((ex, score, latency_ms)) = stream.next().await {
        let score = score
            .with_context(|| format!("scoring {} ({} written so far)", ex.example_id, written))?;
        let resp = Response {
            example: ex,
            model: model.clone(),
            logit_diff: score.logit_diff,
            p_yes: score.p_yes,
            top_logprobs: score.top_logprobs,
            latency_ms,
        };
        serde_json::to_writer(&mut file, &resp)
            .with_context(|| format!("appending to {}", out_path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("appending to {}", out_path.display()))?;
        written += 1;
        if written.is_multiple_of(100) {
            file.flush()
                .with_context(|| format!("flushing {}", out_path.display()))?;
        }
    }
    file.flush()
        .with_context(|| format!("flushing {}", out_path.display()))?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockClient;
    use crate::querygen::{GenConfig, gen_eval};
    use crate::store::read_ndjson;
    use crate::types::Response;
    use crate::types::tests_helpers::{aeneid_interval, caesar_interval};

    fn examples() -> Vec<crate::types::Example> {
        let cfg = GenConfig::default();
        let mut v = gen_eval(&caesar_interval(), "Julius Caesar", &[], &cfg);
        v.extend(gen_eval(&aeneid_interval(), "Aeneid", &[], &cfg));
        v
    }

    #[tokio::test]
    async fn probe_writes_denormalized_responses() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("responses.ndjson");
        let client = MockClient::new("mock-v1");
        let n = run_probe(&client, examples(), &out, 8).await.unwrap();
        assert_eq!(n, 16);
        let rows = read_ndjson::<Response>(&out).unwrap();
        assert_eq!(rows.len(), 16);
        assert!(rows.iter().all(|r| r.model == "mock-v1"));
        assert!(rows.iter().all(|r| r.p_yes > 0.0 && r.p_yes < 1.0));
        assert!(
            rows.iter()
                .any(|r| r.example.subject_name == "Julius Caesar")
        );
    }

    #[tokio::test]
    async fn probe_resumes_without_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("responses.ndjson");
        let client = MockClient::new("mock-v1");
        run_probe(&client, examples(), &out, 4).await.unwrap();
        let n = run_probe(&client, examples(), &out, 4).await.unwrap();
        assert_eq!(n, 0);
        assert_eq!(read_ndjson::<Response>(&out).unwrap().len(), 16);
    }

    #[tokio::test]
    async fn probe_resume_repairs_torn_tail() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("responses.ndjson");
        let client = MockClient::new("mock-v1");
        run_probe(&client, examples(), &out, 4).await.unwrap();
        // simulate a kill mid-flush: a partial line without a trailing newline
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&out).unwrap();
        f.write_all(b"{\"partial\":tru").unwrap();
        drop(f);
        let n = run_probe(&client, examples(), &out, 4).await.unwrap();
        assert_eq!(n, 0);
        assert_eq!(read_ndjson::<Response>(&out).unwrap().len(), 16);
    }

    #[tokio::test]
    async fn probe_rejects_zero_concurrency() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("responses.ndjson");
        let client = MockClient::new("mock-v1");
        let err = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_probe(&client, examples(), &out, 0),
        )
        .await
        .expect("must not hang")
        .unwrap_err();
        assert!(
            err.to_string().contains("concurrency"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn probe_resumes_mid_run() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("responses.ndjson");
        let client = MockClient::new("mock-v1");
        let all = examples();
        let n = run_probe(&client, all[..6].to_vec(), &out, 4)
            .await
            .unwrap();
        assert_eq!(n, 6);
        let n = run_probe(&client, all, &out, 4).await.unwrap();
        assert_eq!(n, 10);
        let rows = read_ndjson::<Response>(&out).unwrap();
        assert_eq!(rows.len(), 16);
        let ids: HashSet<&str> = rows.iter().map(|r| r.example.example_id.as_str()).collect();
        assert_eq!(ids.len(), 16);
    }
}
