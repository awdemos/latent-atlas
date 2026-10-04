//! End-to-end demo pipeline over the bundled fixture.

use crate::metrics::MetricsReport;
use crate::model::{MockClient, ModelClient, OpenAiClient};
use crate::types::{Entity, Interval, Relation};
use crate::{DatasetRoot, metrics, parquet_io, probe, querygen, render, store};
use anyhow::Context;
use querygen::{GenConfig, GenMode};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Builds the model client for a run, shared by the CLI Probe/Demo commands
/// and [`run_demo`]: a deterministic `MockClient` when `mock` (named
/// `mock_name`, overridable via `model`), else the OpenAI-compatible client
/// from env/CLI flags.
pub fn select_client(
    model: Option<String>,
    mock: bool,
    mock_name: &str,
) -> anyhow::Result<Box<dyn ModelClient>> {
    if mock {
        Ok(Box::new(MockClient::new(
            model.unwrap_or_else(|| mock_name.into()),
        )))
    } else {
        Ok(Box::new(OpenAiClient::from_env(model)?))
    }
}

/// Generates examples for one relation and writes them to
/// `generated/{relation}_yesno{suffix}.ndjson` (suffix `_sweep` in Sweep
/// mode), creating parent directories. Uses `GenConfig::default()`, like
/// both generation call sites. Returns `None` without writing anything when
/// no intervals carry `relation` (callers skip); otherwise the written path
/// and the example count.
pub fn generate_for_relation(
    root: &DatasetRoot,
    entities: &[Entity],
    intervals: &[Interval],
    relation: Relation,
    mode: GenMode,
) -> anyhow::Result<Option<(PathBuf, usize)>> {
    let ivs: Vec<Interval> = intervals
        .iter()
        .filter(|i| i.relation == relation)
        .cloned()
        .collect();
    if ivs.is_empty() {
        return Ok(None);
    }
    let examples = querygen::generate(entities, &ivs, &GenConfig::default(), mode)?;
    let suffix = if mode == GenMode::Sweep { "_sweep" } else { "" };
    let path = root
        .generated()
        .join(format!("{}_yesno{suffix}.ndjson", relation.as_str()));
    store::write_ndjson(&path, &examples)?;
    Ok(Some((path, examples.len())))
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
        for mode in [GenMode::Eval, GenMode::Sweep] {
            if let Some((path, _)) = generate_for_relation(root, &entities, &intervals, r, mode)? {
                inputs.push(path);
            }
        }
    }

    let client = select_client(None, mock, "mock-atlas-v1")?;
    for input in &inputs {
        let examples = store::read_ndjson(input)?;
        let stem = input
            .file_stem()
            .ok_or_else(|| anyhow::anyhow!("{}: no file stem", input.display()))?
            .to_string_lossy();
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
