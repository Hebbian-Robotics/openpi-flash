//! Cross-language metadata parse test harness.
//!
//! Reads a server metadata msgpack blob from stdin, runs it through the
//! same `parse_image_specs` + `parse_action_horizon` entry points the
//! client sidecar uses at handshake, and writes the decoded result as
//! JSON to stdout.
//!
//! The Python test suite (`tests/test_metadata_python_emit.py`) exercises
//! this by packing metadata via `msgpack_numpy.Packer()` — the actual
//! production path — and asserting this binary decodes to the same
//! values. That catches any byte-encoding mismatch between Python's
//! msgpack emission and Rust's `rmp_serde` deserialization.

use std::io::{Read, Write};
use std::num::NonZeroUsize;

use anyhow::{Context, Result};
use openpi_flash_transport::image_preprocess::ImageDtype;
use openpi_flash_transport::metadata::{parse_action_horizon, parse_image_specs};

fn main() -> Result<()> {
    let mut input_buffer = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input_buffer)
        .context("Failed to read metadata msgpack bytes from stdin")?;

    let specs = parse_image_specs(&input_buffer)
        .context("parse_image_specs failed on Python-emitted metadata")?;
    let horizon = parse_action_horizon(&input_buffer)
        .context("parse_action_horizon failed on Python-emitted metadata")?;

    // Emit as a small JSON structure the Python test can compare against.
    // `action_horizon` is lowered from `NonZeroUsize` to `usize` here because
    // the Python test just compares numeric values.
    let output = DecodedMetadata {
        image_specs: specs
            .iter()
            .map(|spec| DecodedImageSpec {
                path: spec.path.clone(),
                target_shape: spec.target_shape,
                dtype: match spec.target_dtype {
                    ImageDtype::Uint8 => "uint8",
                    ImageDtype::Float32 => "float32",
                },
            })
            .collect(),
        action_horizon: horizon.map(NonZeroUsize::get),
    };
    let json = serde_json::to_vec(&output).context("Failed to serialize decoded metadata")?;
    std::io::stdout()
        .write_all(&json)
        .context("Failed to write decoded metadata to stdout")?;
    Ok(())
}

#[derive(serde::Serialize)]
struct DecodedMetadata {
    image_specs: Vec<DecodedImageSpec>,
    action_horizon: Option<usize>,
}

#[derive(serde::Serialize)]
struct DecodedImageSpec {
    path: Vec<String>,
    target_shape: [u32; 3],
    dtype: &'static str,
}
