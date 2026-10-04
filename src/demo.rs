//! End-to-end demo pipeline over the bundled fixture.

use crate::metrics::MetricsReport;
use crate::model::{MockClient, ModelClient, OpenAiClient};
use crate::types::{Entity, Interval, Relation, Response};
use crate::{DatasetRoot, metrics, parquet_io, probe, querygen, render, store};
use anyhow::Context;
use querygen::{GenConfig, GenMode};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Read every `*.ndjson` response file under a run directory; bails when the
/// directory holds none. Files are read in sorted-name order and responses
/// are deduped by `example_id` (first occurrence wins): a re-probe can append
/// a second file containing examples already scored, and counting them twice
/// would skew every metric.
pub fn read_run_responses(run: &Path) -> anyhow::Result<Vec<Response>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(run)
        .with_context(|| format!("reading {}", run.display()))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for path in paths {
        if path.extension().and_then(|e| e.to_str()) == Some("ndjson") {
            for r in store::read_ndjson::<Response>(&path)? {
                if seen.insert(r.example.example_id.clone()) {
                    out.push(r);
                }
            }
        }
    }
    if out.is_empty() {
        anyhow::bail!("no responses under {}", run.display());
    }
    Ok(out)
}

/// Run the whole pipeline over [`crate::fixtures::demo_entities`]: canonical
/// parquet, eval+sweep queries per relation, probe (mock or OpenAI-compatible
/// per `mock`), metrics, and score maps. Returns the report and the rendered
/// artifact paths.
pub async fn run_demo(
    root: &DatasetRoot,
    mock: bool,
) -> anyhow::Result<(MetricsReport, Vec<PathBuf>)> {
    root.init()?;
    let pairs = crate::fixtures::demo_entities();
    let entities: Vec<Entity> = pairs.iter().map(|(e, _)| e.clone()).collect();
    let intervals: Vec<Interval> = pairs.iter().map(|(_, i)| i.clone()).collect();
    parquet_io::write_entities(&root.canonical().join("entities.parquet"), &entities)?;
    parquet_io::write_intervals(&root.canonical().join("intervals.parquet"), &intervals)?;

    let cfg = GenConfig::default();
    let relations: BTreeSet<Relation> = intervals.iter().map(|i| i.relation).collect();
    let mut inputs = Vec::new();
    for r in relations {
        let ivs: Vec<Interval> = intervals
            .iter()
            .filter(|i| i.relation == r)
            .cloned()
            .collect();
        for (mode, suffix) in [(GenMode::Eval, ""), (GenMode::Sweep, "_sweep")] {
            let examples = querygen::generate(&entities, &ivs, &cfg, mode)?;
            let path = root
                .generated()
                .join(format!("{}_yesno{suffix}.ndjson", r.as_str()));
            store::write_ndjson(&path, &examples)?;
            inputs.push(path);
        }
    }

    let client: Box<dyn ModelClient> = if mock {
        Box::new(MockClient::new("mock-atlas-v1"))
    } else {
        Box::new(OpenAiClient::from_env(None)?)
    };
    for input in &inputs {
        let examples = store::read_ndjson(input)?;
        let stem = input.file_stem().unwrap().to_string_lossy();
        let out = root
            .run_dir(client.name())
            .join(format!("{stem}.responses.ndjson"));
        probe::run_probe(client.as_ref(), examples, &out, 16).await?;
    }

    let run = root.run_dir(client.name());
    let responses = read_run_responses(&run)?;
    let report = metrics::compute_report(&responses, cfg.world_end);
    let metrics_path = run.join("metrics.json");
    std::fs::write(&metrics_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing {}", metrics_path.display()))?;
    let written = render::render_run(&responses, &run.join("score_maps"), 25)?;
    Ok((report, written))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::querygen::{GenConfig, gen_eval};
    use crate::types::tests_helpers::caesar_interval;

    fn response(p_yes: f64) -> Response {
        let example = gen_eval(
            &caesar_interval(),
            "Julius Caesar",
            &[],
            &GenConfig::default(),
        )
        .into_iter()
        .next()
        .unwrap();
        Response {
            p_yes,
            logit_diff: 0.0,
            top_logprobs: vec![],
            model: "m".into(),
            latency_ms: 1,
            example,
        }
    }

    #[test]
    fn read_run_responses_dedups_across_files() {
        // A re-probe appends a second file to the run dir; an example probed
        // twice (e.g. partial wipe + redo) must be counted once. First
        // occurrence wins per sorted-file order.
        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("runs/mock");
        std::fs::create_dir_all(&run).unwrap();
        let r = response(0.9);
        store::write_ndjson(&run.join("a.responses.ndjson"), std::slice::from_ref(&r)).unwrap();
        store::write_ndjson(&run.join("b.responses.ndjson"), &[r]).unwrap();
        let rows = read_run_responses(&run).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].p_yes, 0.9); // first file wins
    }

    #[test]
    fn read_run_responses_bails_without_ndjson() {
        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("runs/empty");
        std::fs::create_dir_all(&run).unwrap();
        assert!(read_run_responses(&run).is_err());
    }
}
