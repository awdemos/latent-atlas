//! End-to-end demo pipeline over the bundled fixture.

use crate::metrics::MetricsReport;
use crate::model::{MockClient, ModelClient, OpenAiClient};
use crate::types::{Entity, Interval, Relation};
use crate::{DatasetRoot, metrics, parquet_io, probe, querygen, render, store};
use anyhow::Context;
use querygen::{GenConfig, GenMode};
use std::collections::BTreeSet;
use std::path::PathBuf;

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
    let responses = store::read_run_responses(&run)?;
    let report = metrics::compute_report(&responses, cfg.world_end);
    let metrics_path = run.join("metrics.json");
    std::fs::write(&metrics_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing {}", metrics_path.display()))?;
    let written = render::render_run(&responses, &run.join("score_maps"), 25, cfg.world_end)?;
    Ok((report, written))
}
