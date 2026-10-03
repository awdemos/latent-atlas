# Latent Atlas

**Behavioral cartography for language models** — a product from
[vibecodingagency](https://vibecodingagency.com/).

An LLM's factual knowledge is a continuous, fuzzy internal field, not a lookup
table. The original "land or water?" experiment turned token logits into a map
of a model's implicit Earth. Latent Atlas does the same for **time**: it
probes `P(entity was alive/extant/active | entity, year)` across thousands of
(entity, year) queries and reconstructs the model's implicit timeline of
human history — era boundaries, anachronism blindness, calibration, and all.

## Quickstart (no network, no API key)

```bash
cargo run --release -- demo --mock
```

This runs the full pipeline on a bundled fixture (Caesar, the Roman Republic,
the Aeneid, the printing press, …) against a deterministic mock model and
writes score maps to `atlas-data/runs/mock-atlas-v1/score_maps/`:

- `heatmap.png` — entity × year probability field ("civilization bands")
- `curve_<entity>.svg` — per-entity P(Yes) curves with the true interval shaded
- `boundary_error.svg` — how far the model's era boundaries drift, in years
- `calibration.svg` — predicted vs empirical probabilities

## Pipeline

```bash
atlas fetch --entity-type people --limit 10000   # Wikidata -> raw/ (resumable)
atlas normalize                                  # raw/ -> canonical/*.parquet
atlas generate --mode eval                       # canonical -> generated/*.ndjson
atlas generate --mode sweep                      # dense year axis for rendering
atlas split                                      # entity-level split tables
atlas probe --input generated/alive_yesno.ndjson --model gpt-4o-mini
atlas score --run atlas-data/runs/gpt-4o-mini
atlas render --run atlas-data/runs/gpt-4o-mini
```

Model endpoint configuration (any OpenAI-compatible API with logprobs —
vLLM, OpenAI, Together, Fireworks, Ollama's OpenAI shim):

```bash
export ATLAS_BASE_URL="http://localhost:8000/v1"
export ATLAS_API_KEY="..."
export ATLAS_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Use `--mock` on `probe` to dry-run without a model.

## Roman showcase panel

Render the atlas over the curated Roman subset only (curated entity IDs carry
the `curated:` prefix):

```bash
grep 'curated:' atlas-data/generated/exists_yesno_sweep.ndjson > atlas-data/generated/roman_showcase.ndjson
atlas probe --input atlas-data/generated/roman_showcase.ndjson --mock
atlas render --run atlas-data/runs/mock
```

## Design

Three strictly separated dataset layers (see
`docs/superpowers/specs/2026-10-02-latent-atlas-design.md`):

1. **Source facts** (`raw/`) — Wikidata rows + curated CSVs, with provenance.
2. **Semantic intervals** (`canonical/`) — `alive(Caesar) = [-99, -43]` in
   astronomical years (`1 BCE = 0`; `100 BCE = -99`), with honest precision.
3. **Evaluation queries** (`generated/`) — stratified Yes/No prompts:
   interior, near-boundary, far, and era-confusable dates; entity-disjoint
   splits plus a held-out prompt template per relation.

Metrics are never accuracy alone: AUROC, log loss, Brier score, **boundary
error** (the temporal analogue of coastline error), interval IoU, and
temporal smoothness.

## Development

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
