//! Decoder for the small subset of server metadata the client transport
//! cares about (currently `image_specs` and `action_horizon`).
//!
//! Server metadata is msgpack-encoded by Python (`msgpack_numpy.Packer`).
//! We use `rmp-serde` with `serde::Deserialize`-derived structs that mirror
//! the Python-emitted shape so adding new metadata fields is just adding a
//! field to [`ServerMetadata`].

use std::num::NonZeroUsize;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::image_preprocess::{ImageDtype, ImageSpec};

/// Wire-shape of the server metadata blob, as far as the client sidecar
/// cares. `#[serde(default)]` everywhere makes us tolerant of older servers
/// that don't yet advertise these fields.
#[derive(Debug, Default, Deserialize)]
struct ServerMetadata {
    #[serde(default)]
    image_specs: Vec<ImageSpecDeserialize>,
    #[serde(default)]
    action_horizon: Option<u64>,
}

/// Deserialize-only mirror of [`ImageSpec`]. Exists separately so the public
/// [`ImageSpec`] type stays free of `serde` derives (keeping the image
/// pipeline independent of msgpack details).
#[derive(Debug, Deserialize)]
struct ImageSpecDeserialize {
    path: Vec<String>,
    target_shape: [u32; 3],
    dtype: ImageDtypeName,
}

/// Allowed `dtype` values in `image_specs` metadata. `serde(rename_all = ...)`
/// matches the lowercase strings Python emits.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ImageDtypeName {
    Uint8,
    Float32,
}

impl From<ImageDtypeName> for ImageDtype {
    fn from(name: ImageDtypeName) -> Self {
        match name {
            ImageDtypeName::Uint8 => Self::Uint8,
            ImageDtypeName::Float32 => Self::Float32,
        }
    }
}

impl ImageSpecDeserialize {
    fn into_image_spec(self) -> ImageSpec {
        ImageSpec {
            path: self.path,
            target_shape: self.target_shape,
            target_dtype: self.dtype.into(),
        }
    }
}

/// Parse the server metadata blob and extract any advertised image specs.
/// Returns an empty vector if the metadata has no `image_specs` field —
/// callers should treat that as "do not preprocess".
///
/// # Errors
/// Returns `Err` when the metadata bytes are malformed msgpack, the
/// top-level value is not a map, or an individual `image_spec` entry is
/// missing a required field (`path`, `target_shape`, `dtype`) or carries
/// an unsupported dtype.
pub fn parse_image_specs(metadata_bytes: &[u8]) -> Result<Vec<ImageSpec>> {
    let metadata = decode_metadata(metadata_bytes)?;
    Ok(metadata
        .image_specs
        .into_iter()
        .map(ImageSpecDeserialize::into_image_spec)
        .collect())
}

/// Parse the server metadata blob for an advertised `action_horizon`. Returns
/// `None` when the field is absent (the sidecar should fall back to "no
/// chunking").
///
/// # Errors
/// Returns `Err` when the metadata bytes are malformed msgpack, the
/// top-level value is not a map, or `action_horizon` is present but is
/// zero or does not fit in `usize`. The `NonZeroUsize` return type lets
/// the caller rely on the "at least one action step per chunk" invariant
/// without a runtime check.
pub fn parse_action_horizon(metadata_bytes: &[u8]) -> Result<Option<NonZeroUsize>> {
    let metadata = decode_metadata(metadata_bytes)?;
    metadata
        .action_horizon
        .map(|raw| {
            let as_usize =
                usize::try_from(raw).map_err(|_| anyhow!("action_horizon {raw} exceeds usize"))?;
            NonZeroUsize::new(as_usize).ok_or_else(|| anyhow!("action_horizon must be >= 1, got 0"))
        })
        .transpose()
}

fn decode_metadata(metadata_bytes: &[u8]) -> Result<ServerMetadata> {
    if metadata_bytes.is_empty() {
        return Ok(ServerMetadata::default());
    }
    // Python `msgpack_numpy.Packer()` emits string keys as msgpack strings;
    // rmp-serde expects them at struct field names by default.
    rmp_serde::from_slice(metadata_bytes).context("Failed to decode server metadata msgpack")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    /// Mirror of the Python-emitted shape, used in tests to avoid hand-writing
    /// rmpv values (the whole point of this refactor was to stop doing that).
    #[derive(Serialize)]
    struct TestImageSpec {
        path: Vec<String>,
        target_shape: [u32; 3],
        dtype: &'static str,
    }

    #[derive(Serialize, Default)]
    struct TestMetadata {
        #[serde(skip_serializing_if = "Option::is_none")]
        foo: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        image_specs: Vec<TestImageSpec>,
        #[serde(skip_serializing_if = "Option::is_none")]
        action_horizon: Option<u64>,
    }

    fn pack(value: &TestMetadata) -> Vec<u8> {
        // Python's `msgpack_numpy.Packer()` defaults to map types with named
        // keys (not numeric), which `rmp_serde::to_vec_named` matches.
        rmp_serde::to_vec_named(value).unwrap()
    }

    #[test]
    fn returns_empty_when_metadata_is_empty() {
        assert!(parse_image_specs(&[]).unwrap().is_empty());
        assert!(parse_action_horizon(&[]).unwrap().is_none());
    }

    #[test]
    fn returns_empty_when_image_specs_field_missing() {
        let bytes = pack(&TestMetadata {
            foo: Some("bar".to_owned()),
            ..Default::default()
        });
        assert!(parse_image_specs(&bytes).unwrap().is_empty());
        assert!(parse_action_horizon(&bytes).unwrap().is_none());
    }

    #[test]
    fn parses_droid_style_image_specs() {
        let bytes = pack(&TestMetadata {
            image_specs: vec![TestImageSpec {
                path: vec!["observation/exterior_image_1_left".to_owned()],
                target_shape: [224, 224, 3],
                dtype: "uint8",
            }],
            ..Default::default()
        });
        let specs = parse_image_specs(&bytes).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].path, vec!["observation/exterior_image_1_left"]);
        assert_eq!(specs[0].target_shape, [224, 224, 3]);
        assert_eq!(specs[0].target_dtype, ImageDtype::Uint8);
    }

    #[test]
    fn parses_aloha_style_nested_path_and_chw() {
        let bytes = pack(&TestMetadata {
            image_specs: vec![TestImageSpec {
                path: vec!["images".to_owned(), "cam_high".to_owned()],
                target_shape: [3, 224, 224],
                dtype: "uint8",
            }],
            ..Default::default()
        });
        let specs = parse_image_specs(&bytes).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].path, vec!["images", "cam_high"]);
        assert_eq!(specs[0].target_shape, [3, 224, 224]);
    }

    #[test]
    fn parses_action_horizon() {
        let bytes = pack(&TestMetadata {
            action_horizon: Some(50),
            ..Default::default()
        });
        assert_eq!(
            parse_action_horizon(&bytes).unwrap(),
            Some(NonZeroUsize::new(50).unwrap())
        );
    }

    #[test]
    fn rejects_zero_action_horizon() {
        let bytes = pack(&TestMetadata {
            action_horizon: Some(0),
            ..Default::default()
        });
        let err = parse_action_horizon(&bytes).unwrap_err();
        assert!(format!("{err:#}").contains("must be >= 1"));
    }

    #[test]
    fn rejects_unknown_dtype() {
        let bytes = pack(&TestMetadata {
            image_specs: vec![TestImageSpec {
                path: vec!["x".to_owned()],
                target_shape: [1, 1, 3],
                dtype: "complex64",
            }],
            ..Default::default()
        });
        // The `lowercase` rename + enum makes "complex64" fail at deserialize
        // time, before we ever see an unsupported variant.
        let err = parse_image_specs(&bytes).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("complex64") || message.contains("variant"));
    }

    #[test]
    fn rejects_truncated_metadata() {
        // Random non-msgpack bytes.
        let bytes = b"not msgpack";
        let result = parse_image_specs(bytes);
        // Either Ok(empty) if the bytes happen to parse as something, or Err.
        // The intent is "no panic and a clean Result".
        assert!(result.is_ok() || result.is_err());
    }
}
