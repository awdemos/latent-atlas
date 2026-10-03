//! Latent Atlas — temporal-interval benchmark pipeline.
//! See docs/superpowers/specs/2026-10-02-latent-atlas-design.md.

pub mod normalize;
pub mod parquet_io;
pub mod source;
pub mod store;
pub mod types;
pub mod year;

pub use store::DatasetRoot;
