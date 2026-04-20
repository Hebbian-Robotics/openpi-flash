//! Custom binary framing spoken between Python and openpi-flash-transport
//! over the local Unix socket.
//!
//! See `docs/arrow-wire.md` for the spec. The format is deliberately thin
//! so the Python side can produce it with `struct.pack` + `ndarray.tobytes()`
//! without pulling in a serialization framework.

use anyhow::{anyhow, bail, Context, Result};
use byteorder::{BigEndian, ByteOrder};

/// Numpy dtype codes mirrored from `docs/arrow-wire.md`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtypeCode {
    Uint8 = 0x01,
    Int8 = 0x02,
    Uint16 = 0x03,
    Int16 = 0x04,
    Uint32 = 0x05,
    Int32 = 0x06,
    Uint64 = 0x07,
    Int64 = 0x08,
    Float16 = 0x09,
    Float32 = 0x0A,
    Float64 = 0x0B,
    Bool = 0x0C,
}

impl DtypeCode {
    /// Decode a wire byte into a dtype variant.
    ///
    /// # Errors
    /// Returns `Err` when the byte does not correspond to any known dtype.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::Uint8),
            0x02 => Ok(Self::Int8),
            0x03 => Ok(Self::Uint16),
            0x04 => Ok(Self::Int16),
            0x05 => Ok(Self::Uint32),
            0x06 => Ok(Self::Int32),
            0x07 => Ok(Self::Uint64),
            0x08 => Ok(Self::Int64),
            0x09 => Ok(Self::Float16),
            0x0A => Ok(Self::Float32),
            0x0B => Ok(Self::Float64),
            0x0C => Ok(Self::Bool),
            _ => bail!("Unknown local dtype code: {value:#x}"),
        }
    }

    #[must_use]
    pub fn element_size_bytes(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

/// A single named numpy array carried in a local frame.
#[derive(Clone, Debug)]
pub struct LocalArray {
    /// Nested dict path (e.g. `["images", "cam_high"]`). Never empty.
    pub path: Vec<String>,
    pub dtype: DtypeCode,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

/// A whole observation or action payload decoded from a local frame.
#[derive(Clone, Debug)]
pub struct LocalFrame {
    pub schema_id: String,
    pub arrays: Vec<LocalArray>,
    pub scalar_json: Vec<u8>,
}

/// Server-measured timing the server-side transport injects into action
/// responses on the Arrow path. `prev_total_ms` is `None` on the first
/// response of a connection (no previous request to compare against).
#[derive(Clone, Copy, Debug)]
pub struct ServerTiming {
    pub infer_ms: f64,
    pub prev_total_ms: Option<f64>,
}

/// Merge a `server_timing` entry into the frame's scalar JSON object.
/// Existing values for `infer_ms` / `prev_total_ms` are overwritten;
/// other keys (including a Python-supplied `policy_timing`) are preserved.
///
/// # Errors
/// Returns `Err` when the existing `scalar_json` is not valid UTF-8 JSON,
/// or when it decodes to something other than a JSON object.
pub fn inject_server_timing(frame: &mut LocalFrame, timing: ServerTiming) -> Result<()> {
    let mut scalars: serde_json::Value = if frame.scalar_json.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_slice(&frame.scalar_json).context("scalar_json is not valid JSON")?
    };
    let scalars_object = scalars
        .as_object_mut()
        .ok_or_else(|| anyhow!("scalar_json is not a JSON object"))?;

    let mut server_timing = scalars_object
        .remove("server_timing")
        .and_then(|value| match value {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default();
    server_timing.insert(
        "infer_ms".to_owned(),
        serde_json::Value::from(timing.infer_ms),
    );
    if let Some(prev_total_ms) = timing.prev_total_ms {
        server_timing.insert(
            "prev_total_ms".to_owned(),
            serde_json::Value::from(prev_total_ms),
        );
    }
    scalars_object.insert(
        "server_timing".to_owned(),
        serde_json::Value::Object(server_timing),
    );

    frame.scalar_json = serde_json::to_vec(&scalars).context("Failed to re-encode scalar_json")?;
    Ok(())
}

/// Parse a local frame body. `body` is the bytes after the outer length
/// prefix and the 1-byte request/response type.
///
/// # Errors
/// Returns `Err` when the body is truncated, the schema id / path components
/// aren't valid UTF-8, an array's declared shape and data length disagree,
/// or any reserved integer field exceeds its declared width.
pub fn decode_local_frame(body: &[u8]) -> Result<LocalFrame> {
    let mut cursor = Cursor::new(body);

    let schema_id_len = cursor.read_u8()?;
    let schema_id_bytes = cursor.read_slice(usize::from(schema_id_len))?;
    let schema_id = std::str::from_utf8(schema_id_bytes)
        .context("schema_id is not valid UTF-8")?
        .to_owned();

    let num_arrays = cursor.read_u16_be()?;
    let mut arrays = Vec::with_capacity(usize::from(num_arrays));

    for _ in 0..num_arrays {
        let path_depth = cursor.read_u8()?;
        if path_depth == 0 {
            bail!("Array path_depth must be >= 1");
        }
        let mut path = Vec::with_capacity(usize::from(path_depth));
        for _ in 0..path_depth {
            let component_len = cursor.read_u8()?;
            let component_bytes = cursor.read_slice(usize::from(component_len))?;
            let component = std::str::from_utf8(component_bytes)
                .context("path component is not valid UTF-8")?
                .to_owned();
            path.push(component);
        }

        let dtype = DtypeCode::from_u8(cursor.read_u8()?)?;
        let ndim = cursor.read_u8()?;
        let mut shape = Vec::with_capacity(usize::from(ndim));
        for _ in 0..ndim {
            shape.push(cursor.read_u32_be()?);
        }

        let data_len = cursor.read_u64_be()?;
        let data = cursor
            .read_slice(
                usize::try_from(data_len)
                    .map_err(|_| anyhow!("data_len {data_len} exceeds usize"))?,
            )?
            .to_vec();

        let element_count: u64 = shape.iter().map(|&d| u64::from(d)).product();
        let expected_bytes = element_count
            .checked_mul(dtype.element_size_bytes() as u64)
            .ok_or_else(|| anyhow!("Shape * element_size overflows u64"))?;
        if expected_bytes != data_len {
            bail!(
                "Array payload size mismatch for {path:?}: expected {expected_bytes} bytes (shape={shape:?}, dtype={dtype:?}), got {data_len}"
            );
        }

        arrays.push(LocalArray {
            path,
            dtype,
            shape,
            data,
        });
    }

    let scalar_json_len = cursor.read_u32_be()?;
    let scalar_json = cursor
        .read_slice(
            usize::try_from(scalar_json_len)
                .map_err(|_| anyhow!("scalar_json_len {scalar_json_len} exceeds usize"))?,
        )?
        .to_vec();

    cursor.ensure_exhausted()?;

    Ok(LocalFrame {
        schema_id,
        arrays,
        scalar_json,
    })
}

/// Encode a local frame body. Output is the bytes that should follow the
/// outer length prefix and type byte.
///
/// # Errors
/// Returns `Err` when any of the declared length fields (schema id, path
/// component, array count, shape dim count, array data length) would
/// overflow the wire-format integer type that holds it.
pub fn encode_local_frame(frame: &LocalFrame) -> Result<Vec<u8>> {
    let schema_id_bytes = frame.schema_id.as_bytes();
    let schema_id_len = u8::try_from(schema_id_bytes.len()).map_err(|_| {
        anyhow!(
            "schema_id too long ({} bytes, max 255)",
            schema_id_bytes.len()
        )
    })?;
    let num_arrays = u16::try_from(frame.arrays.len())
        .map_err(|_| anyhow!("too many arrays ({}, max 65535)", frame.arrays.len()))?;
    let scalar_json_len = u32::try_from(frame.scalar_json.len())
        .map_err(|_| anyhow!("scalar_json too long ({} bytes)", frame.scalar_json.len()))?;

    let mut estimated_capacity = 1 + schema_id_bytes.len() + 2 + 4 + frame.scalar_json.len();
    for array in &frame.arrays {
        estimated_capacity += 1; // path_depth
        for component in &array.path {
            estimated_capacity += 1 + component.len();
        }
        estimated_capacity += 1 + 1 + array.shape.len() * 4 + 8 + array.data.len();
    }

    let mut out = Vec::with_capacity(estimated_capacity);
    out.push(schema_id_len);
    out.extend_from_slice(schema_id_bytes);
    out.extend_from_slice(&num_arrays.to_be_bytes());

    for array in &frame.arrays {
        let path_depth = u8::try_from(array.path.len())
            .map_err(|_| anyhow!("path too deep ({}, max 255)", array.path.len()))?;
        if path_depth == 0 {
            bail!("Array path cannot be empty");
        }
        out.push(path_depth);
        for component in &array.path {
            let component_bytes = component.as_bytes();
            let component_len = u8::try_from(component_bytes.len()).map_err(|_| {
                anyhow!(
                    "path component too long ({} bytes, max 255)",
                    component_bytes.len()
                )
            })?;
            out.push(component_len);
            out.extend_from_slice(component_bytes);
        }

        out.push(array.dtype as u8);
        let ndim = u8::try_from(array.shape.len())
            .map_err(|_| anyhow!("shape too long ({} dims, max 255)", array.shape.len()))?;
        out.push(ndim);
        for &dim in &array.shape {
            out.extend_from_slice(&dim.to_be_bytes());
        }

        let data_len = u64::try_from(array.data.len())
            .map_err(|_| anyhow!("array data too long ({} bytes)", array.data.len()))?;
        out.extend_from_slice(&data_len.to_be_bytes());
        out.extend_from_slice(&array.data);
    }

    out.extend_from_slice(&scalar_json_len.to_be_bytes());
    out.extend_from_slice(&frame.scalar_json);

    Ok(out)
}

struct Cursor<'a> {
    buf: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.offset
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            bail!(
                "Unexpected end of local frame: need {} bytes, have {}",
                len,
                self.remaining()
            );
        }
        let slice = &self.buf[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_slice(1)?[0])
    }

    fn read_u16_be(&mut self) -> Result<u16> {
        Ok(BigEndian::read_u16(self.read_slice(2)?))
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        Ok(BigEndian::read_u32(self.read_slice(4)?))
    }

    fn read_u64_be(&mut self) -> Result<u64> {
        Ok(BigEndian::read_u64(self.read_slice(8)?))
    }

    fn ensure_exhausted(&self) -> Result<()> {
        if self.remaining() != 0 {
            bail!("Trailing {} bytes in local frame", self.remaining());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> LocalFrame {
        LocalFrame {
            schema_id: "droid".to_owned(),
            arrays: vec![
                LocalArray {
                    path: vec!["observation/joint_position".to_owned()],
                    dtype: DtypeCode::Float64,
                    shape: vec![7],
                    data: (0..7_u32)
                        .flat_map(|i| f64::from(i).to_le_bytes())
                        .collect(),
                },
                LocalArray {
                    path: vec!["images".to_owned(), "cam_high".to_owned()],
                    dtype: DtypeCode::Uint8,
                    shape: vec![2, 2, 3],
                    data: (0..12_u8).collect(),
                },
            ],
            scalar_json: br#"{"prompt":"do something"}"#.to_vec(),
        }
    }

    #[test]
    fn round_trip_encode_decode() {
        let frame = sample_frame();
        let encoded = encode_local_frame(&frame).expect("encode");
        let decoded = decode_local_frame(&encoded).expect("decode");

        assert_eq!(decoded.schema_id, frame.schema_id);
        assert_eq!(decoded.scalar_json, frame.scalar_json);
        assert_eq!(decoded.arrays.len(), frame.arrays.len());
        for (a, b) in decoded.arrays.iter().zip(frame.arrays.iter()) {
            assert_eq!(a.path, b.path);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn rejects_shape_vs_data_size_mismatch() {
        // Craft a frame where data_len agrees with the data slice but the
        // declared shape does not. Decoder must catch the shape mismatch.
        let mut crafted = Vec::new();
        crafted.push(5); // schema_id_len
        crafted.extend_from_slice(b"droid");
        crafted.extend_from_slice(&1_u16.to_be_bytes());
        crafted.push(1); // path_depth
        crafted.push(u8::try_from("observation/joint_position".len()).unwrap());
        crafted.extend_from_slice(b"observation/joint_position");
        crafted.push(DtypeCode::Float64 as u8);
        crafted.push(1); // ndim
        crafted.extend_from_slice(&7_u32.to_be_bytes()); // shape says 7 f64 = 56 bytes
        crafted.extend_from_slice(&55_u64.to_be_bytes()); // but data is 55 bytes
        crafted.extend(std::iter::repeat_n(0_u8, 55));
        crafted.extend_from_slice(&0_u32.to_be_bytes()); // scalar_json_len

        let err = decode_local_frame(&crafted).expect_err("size mismatch should fail");
        let msg = format!("{err:?}");
        assert!(msg.contains("mismatch"), "unexpected error: {msg}");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let frame = sample_frame();
        let mut encoded = encode_local_frame(&frame).expect("encode");
        encoded.push(0);
        assert!(decode_local_frame(&encoded).is_err());
    }

    #[test]
    fn rejects_empty_path() {
        let mut crafted = Vec::new();
        crafted.push(5);
        crafted.extend_from_slice(b"droid");
        crafted.extend_from_slice(&1_u16.to_be_bytes());
        crafted.push(0); // path_depth = 0 (invalid)
        assert!(decode_local_frame(&crafted).is_err());
    }

    #[test]
    fn injects_server_timing_into_empty_scalars() {
        let mut frame = LocalFrame {
            schema_id: "droid".to_owned(),
            arrays: vec![],
            scalar_json: Vec::new(),
        };
        inject_server_timing(
            &mut frame,
            ServerTiming {
                infer_ms: 12.5,
                prev_total_ms: Some(140.0),
            },
        )
        .unwrap();
        let scalars: serde_json::Value = serde_json::from_slice(&frame.scalar_json).unwrap();
        assert_eq!(scalars["server_timing"]["infer_ms"], 12.5);
        assert_eq!(scalars["server_timing"]["prev_total_ms"], 140.0);
    }

    #[test]
    fn injects_server_timing_preserves_policy_timing() {
        let mut frame = LocalFrame {
            schema_id: "droid".to_owned(),
            arrays: vec![],
            scalar_json: br#"{"policy_timing":{"infer_ms":7.0}}"#.to_vec(),
        };
        inject_server_timing(
            &mut frame,
            ServerTiming {
                infer_ms: 22.0,
                prev_total_ms: None,
            },
        )
        .unwrap();
        let scalars: serde_json::Value = serde_json::from_slice(&frame.scalar_json).unwrap();
        assert_eq!(scalars["policy_timing"]["infer_ms"], 7.0);
        assert_eq!(scalars["server_timing"]["infer_ms"], 22.0);
        assert!(scalars["server_timing"].get("prev_total_ms").is_none());
    }
}
