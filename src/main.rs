use clap::{Parser, Subcommand, ValueEnum};
use latent_atlas::model::{MockClient, ModelClient, OpenAiClient};
use latent_atlas::querygen::{GenConfig, GenMode};
use latent_atlas::types::{Entity, EntityType, Example, Interval, Relation, Split};
use latent_atlas::{DatasetRoot, normalize, parquet_io, probe, querygen, render, splits, store};
use std::path::PathBuf;

#[derive(Parser, Debug)]
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

/// Query generation mode: stratified eval dates or the dense per-year sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Eval,
    Sweep,
}

/// Relations the Generate command can restrict to; mirrors
/// [`latent_atlas::types::Relation`] at the CLI boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RelationArg {
    Alive,
    Ongoing,
    Exists,
    Active,
    Available,
}

impl RelationArg {
    fn relation(self) -> Relation {
        match self {
            RelationArg::Alive => Relation::Alive,
            RelationArg::Ongoing => Relation::Ongoing,
            RelationArg::Exists => Relation::Exists,
            RelationArg::Active => Relation::Active,
            RelationArg::Available => Relation::Available,
        }
    }
}

#[derive(Subcommand, Debug)]
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
        relation: Option<RelationArg>,
        #[arg(long, default_value = "eval")]
        mode: Mode,
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
            let mode = match mode {
                Mode::Eval => GenMode::Eval,
                Mode::Sweep => GenMode::Sweep,
            };
            let relations: Vec<Relation> = match &relation {
                Some(r) => vec![r.relation()],
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
                let examples = querygen::generate(&entities, &ivs, &GenConfig::default(), mode)?;
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
            for split in [Split::Train, Split::Dev, Split::Test] {
                let rows: Vec<Entity> = entities
                    .iter()
                    .filter(|e| splits::entity_split(&e.entity_id) == split)
                    .cloned()
                    .collect();
                parquet_io::write_entities(
                    &root
                        .splits()
                        .join(format!("{}_entities.parquet", split.as_str())),
                    &rows,
                )?;
                println!("{}: {} entities", split.as_str(), rows.len());
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
            let responses = store::read_run_responses(&run)?;
            let report =
                latent_atlas::metrics::compute_report(&responses, GenConfig::default().world_end);
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
            let responses = store::read_run_responses(&run)?;
            let written = render::render_run(
                &responses,
                &run.join("score_maps"),
                curves,
                GenConfig::default().world_end,
            )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    #[test]
    fn parses_fetch() {
        let cli =
            Cli::try_parse_from(["atlas", "fetch", "--entity-type", "person", "--limit", "10"])
                .unwrap();
        assert!(matches!(cli.cmd, Cmd::Fetch { limit: 10, .. }));
    }

    #[test]
    fn parses_normalize() {
        assert!(matches!(
            Cli::try_parse_from(["atlas", "normalize"]).unwrap().cmd,
            Cmd::Normalize
        ));
    }

    #[test]
    fn parses_generate_defaults_to_eval_mode() {
        let cli = Cli::try_parse_from(["atlas", "generate"]).unwrap();
        assert!(matches!(
            cli.cmd,
            Cmd::Generate {
                relation: None,
                mode: Mode::Eval
            }
        ));
    }

    #[test]
    fn parses_generate_with_relation_and_sweep_mode() {
        let cli = Cli::try_parse_from([
            "atlas",
            "generate",
            "--relation",
            "alive",
            "--mode",
            "sweep",
        ])
        .unwrap();
        assert!(matches!(
            cli.cmd,
            Cmd::Generate {
                relation: Some(RelationArg::Alive),
                mode: Mode::Sweep
            }
        ));
    }

    #[test]
    fn parses_split() {
        assert!(matches!(
            Cli::try_parse_from(["atlas", "split"]).unwrap().cmd,
            Cmd::Split
        ));
    }

    #[test]
    fn parses_probe() {
        let cli =
            Cli::try_parse_from(["atlas", "probe", "--input", "gen.ndjson", "--mock"]).unwrap();
        assert!(matches!(
            cli.cmd,
            Cmd::Probe {
                concurrency: 16,
                mock: true,
                ..
            }
        ));
    }

    #[test]
    fn parses_score() {
        let cli = Cli::try_parse_from(["atlas", "score", "--run", "runs/m"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Score { .. }));
    }

    #[test]
    fn parses_render() {
        let cli = Cli::try_parse_from(["atlas", "render", "--run", "runs/m"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Render { curves: 25, .. }));
    }

    #[test]
    fn parses_demo() {
        let cli = Cli::try_parse_from(["atlas", "demo", "--mock"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Demo { mock: true }));
    }

    #[test]
    fn rejects_bad_mode_at_parse_time() {
        let err = Cli::try_parse_from(["atlas", "generate", "--mode", "bogus"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn rejects_bad_relation_at_parse_time() {
        let err = Cli::try_parse_from(["atlas", "generate", "--relation", "bogus"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn rejects_missing_required_args() {
        let err = Cli::try_parse_from(["atlas", "fetch"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
        let err = Cli::try_parse_from(["atlas", "score"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }
}
