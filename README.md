# Latent Atlas

**Behavioral cartography for language models** — a product from
[vibecodingagency](https://vibecodingagency.com/). MIT licensed.

An LLM's factual knowledge is a continuous, fuzzy internal field, not a lookup
table.

The inspiration: [Karpathy's "land or water?" eval](https://x.com/karpathy/status/2105909609487872075) —
ask an LLM "Land or Water?" at thousands of latitude/longitude coordinates,
plot the token probabilities, and the continents emerge from text alone.

Latent Atlas generalizes the technique to **any kind of factual data**: pick a
claim type and its controlled input dimensions, probe `P(claim | inputs)` with
one-token queries, and map where the model is confident, fuzzy, or wrong. The
first use case is **factual historical events** — it probes
`P(entity was alive/extant/active | entity, year)` across thousands of queries
and reconstructs the model's *implicit timeline of history*: era boundaries,
anachronism blindness, calibration, and all.

What you get is not an accuracy score. It's a map: which centuries the model
locates confidently, where its interval edges blur (measured in *years* of
drift), and how its confidence compares to reality.

## What it can do

- **Probe any OpenAI-compatible model** (vLLM, Ollama, OpenAI, Together,
  Fireworks, …) with logprobs — one token per query, so a 7,500-query sweep
  costs almost nothing and finishes in minutes on a consumer GPU.
- **Map a model's temporal world model** as a PNG heatmap (entity × year
  probability field), per-entity probability curves, a boundary-error
  histogram, and a calibration plot.
- **Score it properly**: accuracy, AUROC, log loss, Brier score,
  **boundary error** (the temporal analogue of coastline error, in years),
  interval IoU, and temporal smoothness — stratified by date band
  (interior / near-boundary / far / era-confusable) and by relation
  (alive / ongoing / exists / available).
- **Build benchmarks from Wikidata** at any scale (people, events, polities,
  works, technologies — resumable fetches), or hand-curate your own entity
  set as CSV.
- **Run fully offline** against a deterministic mock model for CI, testing,
  and pipeline development.

Real run, 15-entity fixture, local Ollama models (eval sets, 96 queries):

| model | accuracy | AUROC | Brier | interval IoU |
|---|---|---|---|---|
| nimble (9B, Bespoke Labs) | 0.817 | 0.916 | 0.126 | 0.596 |
| ternary-bonsai (1.7B) | 0.729 | 0.763 | 0.210 | 0.476 |

The maps tell the richer story: nimble answers far dates perfectly but its
interval *edges* are near chance — boundary error ±181 years on start dates
vs ±77 on end dates. It knows Caesar lived in the middle of his life, not
exactly when he was born.

## Install

Requires Rust 1.88+.

```bash
# From source (recommended for now)
git clone <this repo> && cd latent-atlas
cargo install --path .

# Or run without installing
cargo run --release -- --help
```

The binary is named `atlas`. There are no runtime dependencies beyond your
model endpoint; everything it writes lands in `./atlas-data/` (gitignored).

## Quickstart

**Tier 0 — zero setup, no network, no API key** (runs the full pipeline on a
bundled fixture against a deterministic mock):

```bash
atlas demo --mock
```

Outputs land in `atlas-data/runs/mock-atlas-v1/score_maps/`: `heatmap.png`,
`curve_<name>_<relation>_<id>.svg` per entity, `boundary_error.svg`,
`calibration.svg`, plus `metrics.json`.

**Tier 1 — a real local model via Ollama:**

```bash
ollama pull llama3.1:8b            # any instruct model
export ATLAS_BASE_URL="http://localhost:11434/v1"
export ATLAS_API_KEY="ollama"      # any non-empty string works for Ollama
export ATLAS_MODEL="llama3.1:8b"
atlas probe --input atlas-data/generated/alive_yesno_sweep.ndjson
atlas score  --run "atlas-data/runs/llama3.1:8b"
atlas render --run "atlas-data/runs/llama3.1:8b"
```

(After `demo --mock` you already have the generated query files; otherwise run
`atlas demo --mock` once, or build your own dataset — see "The full pipeline".)

**Tier 2 — cloud APIs:**

```bash
export ATLAS_BASE_URL="https://api.openai.com/v1"
export ATLAS_API_KEY="sk-..."
export ATLAS_MODEL="gpt-4o-mini"
atlas probe --input atlas-data/generated/alive_yesno_sweep.ndjson --concurrency 32
```

## CLI reference

Global flag: `--data <dir>` (default `atlas-data`) — the dataset root.

| command | what it does |
|---|---|
| `demo [--mock]` | End-to-end run on the bundled fixture: canonical parquet → queries → probe → metrics → score maps. |
| `fetch --entity-type <person\|event\|polity\|organization\|work\|technology> --limit <n>` | Fetch facts from Wikidata (SPARQL) into `raw/`. Resumable: re-running continues where it stopped. |
| `normalize` | Parse `raw/` (Wikidata NDJSON + curated CSVs) into `canonical/*.parquet` (entities, intervals, provenance). |
| `generate [--relation <r>] [--mode eval\|sweep]` | Build Yes/No query files into `generated/`. `eval` = stratified 8-per-entity benchmark; `sweep` = dense year grid for maps. |
| `split` | Write entity-disjoint train/dev/test split tables into `splits/`. |
| `probe --input <file> [--model <name>] [--mock] [--concurrency 16]` | Score every query against a model; appends `*.responses.ndjson` into `runs/<model>/`. Resumable and concurrent. |
| `score --run <dir>` | Compute `metrics.json` for a run directory. |
| `render --run <dir> [--curves 25]` | Render score maps into `<run>/score_maps/`. |

### Configuration (environment variables)

| variable | required | meaning |
|---|---|---|
| `ATLAS_MODEL` | yes (or `--model`) | Model name to send to the endpoint. |
| `ATLAS_BASE_URL` | no | OpenAI-compatible base URL. Default `https://api.openai.com/v1`. |
| `ATLAS_API_KEY` | no | Bearer token. Empty is allowed — keyless local endpoints (Ollama, vLLM) need none. |
| `ATLAS_ASSISTANT_PREFILL` | see FAQ | Text to pre-fill as the assistant turn (e.g. `Answer:`). Required for reasoning/thinking models. |

## FAQ

**My model errors with "Yes/No absent from top_logprobs".**
The model opens with a think-token instead of an answer (Qwen3.5, nimble,
DeepSeek-R1-style reasoning models), pushing Yes/No out of the top-20. Fix:

```bash
export ATLAS_ASSISTANT_PREFILL="Answer:"
```

This pre-fills the assistant turn so the first generated token *is* the
answer. Verified on Ollama's nimble (first token becomes ` Yes` at p≈0.92).
Note the prefill conditions the probe on "answer mode", which slightly
changes the measured distribution — say so when publishing results.

**Which Ollama version do I need?** 0.35+ for recent library models; any
recent version works for the probing protocol itself. Ollama's OpenAI shim
supports the `logprobs`/`top_logprobs` fields Latent Atlas requires.

**How much does a run cost?** One completion token per query, plus the prompt.
The bundled fixture's full sweep is ~7,500 queries — a few cents on cloud
APIs, minutes on a local GPU. Wikidata-scale benchmarks are linear in entity
count.

**My run got interrupted (killed, network drop, timeout).** Everything is
resumable. `probe` dedups by query ID and skips finished rows; `fetch`
resumes at the last complete page. Torn partial lines left by a kill are
repaired automatically on resume.

**How do I add my own entities?** Drop a CSV into `atlas-data/raw/` (see
`fixtures/curated_roman.csv` for the exact schema: entity id, type, name,
relation, start/end year in astronomical numbering, confidence), then
`atlas normalize && atlas generate --mode sweep && atlas probe ...`. Years are
astronomical (`1 BCE = 0`, `100 BCE = -99`); open-ended intervals leave the
end year empty.

**What do the metrics mean?**
- *Boundary error*: where the model's P(Yes) crosses 0.5 vs the true interval
  edge, in years. The single most revealing number for "does it know *when*".
- *Interval IoU*: overlap between the model's confident interval and truth.
- *Smoothness*: average year-to-year probability jump — noisy maps mean an
  unstable representation, not just uncertainty.
- Band breakdowns: `interior`/`far_*` probe broad knowledge;
  `near_before`/`near_after` probe boundary precision; `era_confusable` dates
  are when a *different* famous entity of the same type was active — it tests
  entity discrimination, not just chronology.

**How large a benchmark should I build?** The 15-entity fixture demonstrates
the method; meaningful claims start around a few hundred entities, and the
dataset is designed to scale to tens of thousands via `fetch`. Keep date
precision honest: the pipeline carries `year`/`decade`/`century`/`approximate`
markers through to labels instead of faking exact years.

**Is this interpretability?** It's behavioral cartography: a systematic probe
of model *outputs* along controlled input dimensions. It does not read
activations, and a sharp coastline does not mean the model "stores a map" —
it means the behavior is spatially coherent. The maps are evidence about
behavior, not photographs of internal representations.

## For AI agents

If you are an AI agent (coding assistant, autonomous researcher) working in
this repo, here is everything you need.

**What this repo is**: a Rust workspace (lib `latent_atlas` + binary `atlas`)
implementing a temporal-interval probing pipeline. 17 modules in `src/`,
one integration test in `tests/`.

**Setup**: `cargo build --release` (Rust 1.88+). No services, no databases,
no API keys needed for tests or the mock demo.

**The 5 commands to run an experiment end-to-end**:

```bash
cargo run --release -- demo --mock                                    # pipeline smoke test, real artifacts
export ATLAS_BASE_URL=... ATLAS_API_KEY=... ATLAS_MODEL=...           # real model
cargo run --release -- probe --input atlas-data/generated/exists_yesno_sweep.ndjson
cargo run --release -- score  --run "atlas-data/runs/$ATLAS_MODEL"
cargo run --release -- render --run "atlas-data/runs/$ATLAS_MODEL"
```

**Conventions you must follow**:

- `atlas-data/` is generated state — NEVER commit it (already gitignored).
- Run all three gates before committing: `cargo fmt --check`,
  `cargo test`, `cargo clippy --all-targets -- -D warnings`. All must pass.
- Years are astronomical integers internally (`1 BCE = 0`); display strings
  (`100 BCE`) are produced by `year::display_year` only at prompt/output time.
- Public items carry doc comments; I/O errors use `anyhow::Context` naming
  the path; no panics on user-supplied data.
- Generation is deterministic (seeded RNG, pinned split hashes) — do not
  introduce unseeded randomness.

**Programmatic use** (Rust API sketch):

```rust
use latent_atlas::{DatasetRoot, model::{ModelClient, OpenAiClient}, probe, metrics, render};

let client = OpenAiClient::from_env(None)?;          // honors ATLAS_* env
let examples: Vec<latent_atlas::types::Example> =
    latent_atlas::store::read_ndjson(&input_path)?;
probe::run_probe(&client, examples, &out_path, 16).await?;
let responses = latent_atlas::store::read_run_responses(&run_dir)?;
let report = metrics::compute_report(&responses, 2026);
render::render_run(&responses, &run_dir.join("score_maps"), 25, 2026)?;
```

Implement `model::ModelClient` (one method: `score(&Example)`) to plug in any
backend; `model::MockClient` shows the contract and powers the test suite.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test        # 118 tests: unit + 12 CLI parse tests + end-to-end pipeline
```

## License

MIT — see [LICENSE](LICENSE). Dataset derived from Wikidata (CC0) plus
hand-curated additions.
