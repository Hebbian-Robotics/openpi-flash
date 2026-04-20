//! Transport library for openpi-flash: QUIC + Arrow IPC + image preprocessing
//! + action-chunk caching + server-timing injection.
//!
//! The binary at `src/main.rs` is the deployed transport process; smaller
//! helper binaries (cross-language test harnesses, benchmark utilities) live
//! under `src/bin/` and reuse the codec modules exposed here.

pub mod arrow_codec;
pub mod chunk_cache;
pub mod image_preprocess;
pub mod local_format;
pub mod metadata;
