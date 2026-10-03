use clap::{Parser, Subcommand};
use latent_atlas::model::{MockClient, ModelClient, OpenAiClient};
use latent_atlas::querygen::{GenConfig, GenMode};
use latent_atlas::types::{Entity, EntityType, Example, Interval, Relation};
use latent_atlas::{DatasetRoot, normalize, parquet_io, probe, querygen, render, splits, store};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "atlas",
    version,
    about = "Latent Atlas — temporal-interval behavioral cartography"
)]
struct Cli {
    /// Dataset root directory
    #[arg(long, default_value = "atlas-data", global = true)]
    data: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fetch raw facts from Wikidata into raw/ (resumable)
    Fetch {
        #[arg(long)]
        entity_type: String,
        #[arg(long)]
        limit: usize,
    },
    /// Normalize raw/ into canonical/*.parquet
    Normalize,
    /// Generate evaluation queries into generated/
    Generate {
        #[arg(long)]
        relation: Option<String>,
        #[arg(long, default_value = "eval")]
        mode: String,
    },
    /// Write entity-level split tables into splits/
    Split,
    /// Probe a model over a generated query file
    Probe {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        mock: bool,
        #[arg(long, default_value_t = 16)]
        concurrency: usize,
    },
    /// Score a run directory into metrics.json
    Score {
        #[arg(long)]
        run: PathBuf,
    },
    /// Render score maps for a run directory
    Render {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value_t = 25)]
        curves: usize,
    },
    /// End-to-end demo on the bundled fixture
    Demo {
        #[arg(long)]
        mock: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let root = DatasetRoot::new(&cli.data);
    root.init()?;
    match cli.cmd {
        Cmd::Fetch { entity_type, limit } => {
            let t = EntityType::from_str(&entity_type)
                .ok_or_else(|| anyhow::anyhow!("bad entity_type {entity_type:?}"))?;
            let client = latent_atlas::source::wikidata::SparqlClient::new();
            let path = root.raw_file(&format!("{}.jsonl", normalize::raw_file_stem(t)));
            let n = client.fetch_all(t, limit, &path).await?;
            println!("wrote {n} new rows to {}", path.display());
        }
        Cmd::Normalize => {
            let (entities, intervals, provenance) = normalize::normalize_all(&root)?;
            parquet_io::write_entities(&root.canonical().join("entities.parquet"), &entities)?;
            parquet_io::write_intervals(&root.canonical().join("intervals.parquet"), &intervals)?;
            parquet_io::write_provenance(
                &root.canonical().join("provenance.parquet"),
                &provenance,
            )?;
            println!(
                "{} entities, {} intervals, {} provenance rows",
                entities.len(),
                intervals.len(),
                provenance.len()
            );
        }
        Cmd::Generate { relation, mode } => {
            let entities = parquet_io::read_entities(&root.canonical().join("entities.parquet"))?;
            let intervals =
                parquet_io::read_intervals(&root.canonical().join("intervals.parquet"))?;
            let mode = match mode.as_str() {
                "eval" => GenMode::Eval,
                "sweep" => GenMode::Sweep,
                other => anyhow::bail!("mode must be eval|sweep, got {other:?}"),
            };
            let relations: Vec<Relation> = match &relation {
                Some(r) => vec![
                    Relation::from_str(r).ok_or_else(|| anyhow::anyhow!("bad relation {r:?}"))?,
                ],
                None => vec![
                    Relation::Alive,
                    Relation::Ongoing,
                    Relation::Exists,
                    Relation::Active,
                    Relation::Available,
                ],
            };
            for r in relations {
                let ivs: Vec<Interval> = intervals
                    .iter()
                    .filter(|i| i.relation == r)
                    .cloned()
                    .collect();
                if ivs.is_empty() {
                    continue;
                }
                let examples = querygen::generate(&entities, &ivs, &GenConfig::default(), mode);
                let suffix = if mode == GenMode::Sweep { "_sweep" } else { "" };
                let path = root
                    .generated()
                    .join(format!("{}_yesno{suffix}.ndjson", r.as_str()));
                store::write_ndjson(&path, &examples)?;
                println!(
                    "{}: {} examples -> {}",
                    r.as_str(),
                    examples.len(),
                    path.display()
                );
            }
        }
        Cmd::Split => {
            let entities = parquet_io::read_entities(&root.canonical().join("entities.parquet"))?;
            for split in ["train", "dev", "test"] {
                let rows: Vec<Entity> = entities
                    .iter()
                    .filter(|e| splits::entity_split(&e.entity_id) == split)
                    .cloned()
                    .collect();
                parquet_io::write_entities(
                    &root.splits().join(format!("{split}_entities.parquet")),
                    &rows,
                )?;
                println!("{split}: {} entities", rows.len());
            }
        }
        Cmd::Probe {
            input,
            model,
            mock,
            concurrency,
        } => {
            let examples: Vec<Example> = store::read_ndjson(&input)?;
            let client: Box<dyn ModelClient> = if mock {
                Box::new(MockClient::new(model.unwrap_or_else(|| "mock".into())))
            } else {
                Box::new(OpenAiClient::from_env(model)?)
            };
            let stem = input.file_stem().unwrap().to_string_lossy();
            let out = root
                .run_dir(client.name())
                .join(format!("{stem}.responses.ndjson"));
            let n = probe::run_probe(client.as_ref(), examples, &out, concurrency).await?;
            println!("wrote {n} responses to {}", out.display());
        }
        Cmd::Score { run } => {
            let responses = latent_atlas::demo::read_run_responses(&run)?;
            let report = latent_atlas::metrics::compute_report(&responses, 2026);
            let path = run.join("metrics.json");
            std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
            println!(
                "accuracy {:.3}  auroc {:?}  brier {:.3}  iou {:.3}",
                report.overall.accuracy,
                report.overall.auroc,
                report.overall.brier,
                report.mean_interval_iou
            );
            println!("wrote {}", path.display());
        }
        Cmd::Render { run, curves } => {
            let responses = latent_atlas::demo::read_run_responses(&run)?;
            let written = render::render_run(&responses, &run.join("score_maps"), curves)?;
            for p in written {
                println!("wrote {}", p.display());
            }
        }
        Cmd::Demo { mock } => {
            let (report, written) = latent_atlas::demo::run_demo(&root, mock).await?;
            println!("\nDemo complete.");
            println!("  accuracy: {:.3}", report.overall.accuracy);
            println!("  interval IoU: {:.3}", report.mean_interval_iou);
            println!("  artifacts:");
            for p in written {
                println!("    {}", p.display());
            }
        }
    }
    Ok(())
}
