//! Latent Atlas — temporal-interval benchmark pipeline.
//! See docs/superpowers/specs/2026-10-02-latent-atlas-design.md.

pub mod metrics;
pub mod model;
pub mod normalize;
pub mod parquet_io;
pub mod probe;
pub mod querygen;
pub mod render;
pub mod source;
pub mod splits;
pub mod store;
pub mod types;
pub mod year;

pub use store::DatasetRoot;
