//! Cross-language codec test harness.
//!
//! Reads a local-frame byte stream from stdin, pushes it through
//! ``local_format::decode_local_frame`` → ``arrow_codec::encode_arrow_ipc`` →
//! ``arrow_codec::decode_arrow_ipc`` → ``local_format::encode_local_frame``,
//! and writes the result to stdout. A successful round-trip should produce
//! output that is byte-for-byte identical to the Python ``pack_local_frame``
//! output for the same payload (modulo deterministic field ordering).
//!
//! Optionally, if the env var ``OPENPI_ECHO_INJECT_TIMING`` is set to a
//! comma-separated `infer_ms[,prev_total_ms]` (e.g. `12.5,140.0` or just
//! `12.5`), the harness injects that into the frame's `server_timing`
//! before re-encoding. This lets the Python test suite verify the same
//! injection path used by the server-mode sidecar.
//!
//! Intended only for tests: the Python ``test_arrow_wire`` suite spawns this
//! binary and feeds it sample frames to validate cross-language compatibility.

use std::io::{Read, Write};

use anyhow::{Context, Result};
use openpi_flash_transport::{
    arrow_codec, image_preprocess,
    local_format::{self, ServerTiming},
    metadata,
};

fn main() -> Result<()> {
    let mut input_buffer = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input_buffer)
        .context("Failed to read local frame bytes from stdin")?;

    let mut frame = local_format::decode_local_frame(&input_buffer)
        .context("Rust failed to decode Python-produced local frame")?;

    // Apply image preprocessing FIRST (matches the client sidecar's request
    // path: resize before Arrow-encoding for the wire).
    if let Some(image_specs_metadata_bytes) = read_image_specs_metadata_env()? {
        let specs = metadata::parse_image_specs(&image_specs_metadata_bytes)
            .context("Failed to parse OPENPI_ECHO_IMAGE_SPECS_METADATA")?;
        for index in 0..frame.arrays.len() {
            let path = frame.arrays[index].path.clone();
            if let Some(spec) = specs.iter().find(|spec| spec.path == path) {
                if let Some(replacement) =
                    image_preprocess::maybe_preprocess(&frame.arrays[index], spec)
                        .with_context(|| format!("Failed to preprocess image {path:?}"))?
                {
                    frame.arrays[index] = replacement;
                }
            }
        }
    }

    let arrow_ipc_bytes = arrow_codec::encode_arrow_ipc(&frame)
        .context("Rust failed to encode Arrow IPC from local frame")?;
    let mut frame_after_arrow = arrow_codec::decode_arrow_ipc(&arrow_ipc_bytes)
        .context("Rust failed to decode its own Arrow IPC output")?;

    if let Some(timing) = parse_inject_timing_env()? {
        local_format::inject_server_timing(&mut frame_after_arrow, timing)
            .context("Failed to inject server_timing in echo harness")?;
    }

    let echoed_bytes = local_format::encode_local_frame(&frame_after_arrow)
        .context("Rust failed to re-encode local frame")?;

    std::io::stdout()
        .write_all(&echoed_bytes)
        .context("Failed to write echoed bytes to stdout")?;
    Ok(())
}

/// Read a Python-supplied msgpack metadata blob from a file path so the
/// echo harness can exercise the same `parse_image_specs` path the client
/// sidecar uses. Path is taken from `OPENPI_ECHO_IMAGE_SPECS_METADATA`.
fn read_image_specs_metadata_env() -> Result<Option<Vec<u8>>> {
    let Ok(path) = std::env::var("OPENPI_ECHO_IMAGE_SPECS_METADATA") else {
        return Ok(None);
    };
    let bytes = std::fs::read(&path)
        .with_context(|| format!("Failed to read image_specs metadata from {path}"))?;
    Ok(Some(bytes))
}

fn parse_inject_timing_env() -> Result<Option<ServerTiming>> {
    let Ok(value) = std::env::var("OPENPI_ECHO_INJECT_TIMING") else {
        return Ok(None);
    };
    let parts: Vec<&str> = value.split(',').collect();
    let infer_ms: f64 = parts
        .first()
        .ok_or_else(|| anyhow::anyhow!("OPENPI_ECHO_INJECT_TIMING is empty"))?
        .trim()
        .parse()
        .context("OPENPI_ECHO_INJECT_TIMING infer_ms must be a float")?;
    let prev_total_ms = match parts.get(1) {
        Some(raw) if !raw.trim().is_empty() => Some(
            raw.trim()
                .parse::<f64>()
                .context("OPENPI_ECHO_INJECT_TIMING prev_total_ms must be a float")?,
        ),
        _ => None,
    };
    Ok(Some(ServerTiming {
        infer_ms,
        prev_total_ms,
    }))
}
