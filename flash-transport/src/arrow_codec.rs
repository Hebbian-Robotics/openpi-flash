//! Convert between [`LocalFrame`] payloads and Arrow IPC Streaming Format
//! bytes used on the QUIC wire.
//!
//! Each inference payload becomes a single Arrow IPC stream containing one
//! [`RecordBatch`] with one row. Every named numpy array becomes one column
//! of type `Binary` whose row value is the raw tensor bytes. The original
//! numpy dtype, shape, and nested dict path live in Arrow field metadata.
//! Schema metadata carries the robot `schema_id` and the JSON scalar blob.
//!
//! The wire format is Rust-sidecar-to-Rust-sidecar only, so we deliberately
//! skip the cost of per-dtype typed arrays and keep one `Binary` column per
//! numpy array.

use std::io::Cursor;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{Array, BinaryArray, RecordBatch, RecordBatchOptions};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::local_format::{DtypeCode, LocalArray, LocalFrame};

const SCHEMA_METADATA_SCHEMA_ID_KEY: &str = "openpi_schema_id";
const SCHEMA_METADATA_SCALAR_JSON_KEY: &str = "openpi_scalar_json";
const FIELD_METADATA_DTYPE_KEY: &str = "openpi_dtype";
const FIELD_METADATA_SHAPE_KEY: &str = "openpi_shape";
const FIELD_METADATA_PATH_KEY: &str = "openpi_path";
const FLATTENED_PATH_SEPARATOR: &str = "\u{001f}";

/// Encode a [`LocalFrame`] as Arrow IPC Streaming Format bytes.
///
/// # Errors
/// Returns `Err` when the Arrow writer fails to serialize the schema or
/// record batch (in practice: non-UTF-8 scalar metadata or an unreachable
/// arrow-ipc internal error).
pub fn encode_arrow_ipc(frame: &LocalFrame) -> Result<Vec<u8>> {
    let schema = build_schema(frame)?;
    let schema_ref = Arc::new(schema);

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(frame.arrays.len());
    for array in &frame.arrays {
        let binary_array = BinaryArray::from(vec![array.data.as_slice()]);
        columns.push(Arc::new(binary_array));
    }

    // Planner (subtask_only) responses carry no numpy arrays — every value
    // lives in scalar_json. Arrow needs an explicit row count in that case
    // because it normally infers num_rows from the first column. We always
    // emit one logical row per inference, matching the decoder's invariant.
    let options = RecordBatchOptions::new().with_row_count(Some(1));
    let record_batch = RecordBatch::try_new_with_options(schema_ref.clone(), columns, &options)
        .context("Failed to assemble Arrow RecordBatch")?;

    let mut buffer: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, schema_ref.as_ref())
            .context("Failed to construct Arrow StreamWriter")?;
        writer
            .write(&record_batch)
            .context("Failed to write Arrow RecordBatch")?;
        writer.finish().context("Failed to finalize Arrow stream")?;
    }
    Ok(buffer)
}

/// Decode Arrow IPC Streaming Format bytes back into a [`LocalFrame`].
///
/// # Errors
/// Returns `Err` when the Arrow stream is malformed, contains zero or more
/// than one record batch, or is missing the schema / field metadata keys
/// this codec relies on (`openpi_schema_id`, `openpi_scalar_json`,
/// `openpi_dtype`, `openpi_shape`, `openpi_path`).
pub fn decode_arrow_ipc(bytes: &[u8]) -> Result<LocalFrame> {
    let cursor = Cursor::new(bytes);
    let mut reader =
        StreamReader::try_new(cursor, None).context("Failed to construct Arrow StreamReader")?;

    let schema_ref = reader.schema();
    let batch = reader
        .next()
        .ok_or_else(|| anyhow!("Arrow stream contained no RecordBatch"))?
        .context("Failed to read Arrow RecordBatch")?;
    if reader.next().is_some() {
        bail!("Arrow stream contained more than one RecordBatch");
    }
    if batch.num_rows() != 1 {
        bail!(
            "Expected exactly 1 row in Arrow RecordBatch, got {}",
            batch.num_rows()
        );
    }

    let schema_metadata = schema_ref.metadata();
    let schema_id = schema_metadata
        .get(SCHEMA_METADATA_SCHEMA_ID_KEY)
        .cloned()
        .ok_or_else(|| anyhow!("Arrow schema missing {SCHEMA_METADATA_SCHEMA_ID_KEY}"))?;
    let scalar_json = schema_metadata
        .get(SCHEMA_METADATA_SCALAR_JSON_KEY)
        .cloned()
        .unwrap_or_default()
        .into_bytes();

    let mut arrays = Vec::with_capacity(batch.num_columns());
    for (column_index, field) in schema_ref.fields().iter().enumerate() {
        let field_metadata = field.metadata();

        let dtype_str = field_metadata
            .get(FIELD_METADATA_DTYPE_KEY)
            .ok_or_else(|| anyhow!("Field {} missing {FIELD_METADATA_DTYPE_KEY}", field.name()))?;
        let dtype = parse_dtype_name(dtype_str)?;

        let shape_json = field_metadata
            .get(FIELD_METADATA_SHAPE_KEY)
            .ok_or_else(|| anyhow!("Field {} missing {FIELD_METADATA_SHAPE_KEY}", field.name()))?;
        let shape: Vec<u32> = serde_json::from_str(shape_json).with_context(|| {
            format!(
                "Failed to parse {FIELD_METADATA_SHAPE_KEY} for field {}",
                field.name()
            )
        })?;

        let path_json = field_metadata
            .get(FIELD_METADATA_PATH_KEY)
            .ok_or_else(|| anyhow!("Field {} missing {FIELD_METADATA_PATH_KEY}", field.name()))?;
        let path: Vec<String> = serde_json::from_str(path_json).with_context(|| {
            format!(
                "Failed to parse {FIELD_METADATA_PATH_KEY} for field {}",
                field.name()
            )
        })?;
        if path.is_empty() {
            bail!("Field {} has empty path metadata", field.name());
        }

        let binary_array = batch
            .column(column_index)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| anyhow!("Field {} is not a BinaryArray as expected", field.name()))?;
        let data = binary_array.value(0).to_vec();

        arrays.push(LocalArray {
            path,
            dtype,
            shape,
            data,
        });
    }

    Ok(LocalFrame {
        schema_id,
        arrays,
        scalar_json,
    })
}

fn build_schema(frame: &LocalFrame) -> Result<Schema> {
    let mut fields = Vec::with_capacity(frame.arrays.len());
    for array in &frame.arrays {
        if array.path.is_empty() {
            bail!("Cannot build Arrow field for array with empty path");
        }
        let flattened_name = array.path.join(FLATTENED_PATH_SEPARATOR);

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            FIELD_METADATA_DTYPE_KEY.to_owned(),
            dtype_name(array.dtype).to_owned(),
        );
        metadata.insert(
            FIELD_METADATA_SHAPE_KEY.to_owned(),
            serde_json::to_string(&array.shape).context("Failed to serialize shape metadata")?,
        );
        metadata.insert(
            FIELD_METADATA_PATH_KEY.to_owned(),
            serde_json::to_string(&array.path).context("Failed to serialize path metadata")?,
        );

        let field = Field::new(flattened_name, DataType::Binary, false).with_metadata(metadata);
        fields.push(field);
    }

    let mut schema_metadata = std::collections::HashMap::new();
    schema_metadata.insert(
        SCHEMA_METADATA_SCHEMA_ID_KEY.to_owned(),
        frame.schema_id.clone(),
    );
    let scalar_json_str =
        std::str::from_utf8(&frame.scalar_json).context("scalar_json is not valid UTF-8")?;
    schema_metadata.insert(
        SCHEMA_METADATA_SCALAR_JSON_KEY.to_owned(),
        scalar_json_str.to_owned(),
    );

    Ok(Schema::new_with_metadata(fields, schema_metadata))
}

fn dtype_name(dtype: DtypeCode) -> &'static str {
    match dtype {
        DtypeCode::Uint8 => "uint8",
        DtypeCode::Int8 => "int8",
        DtypeCode::Uint16 => "uint16",
        DtypeCode::Int16 => "int16",
        DtypeCode::Uint32 => "uint32",
        DtypeCode::Int32 => "int32",
        DtypeCode::Uint64 => "uint64",
        DtypeCode::Int64 => "int64",
        DtypeCode::Float16 => "float16",
        DtypeCode::Float32 => "float32",
        DtypeCode::Float64 => "float64",
        DtypeCode::Bool => "bool",
    }
}

fn parse_dtype_name(name: &str) -> Result<DtypeCode> {
    match name {
        "uint8" => Ok(DtypeCode::Uint8),
        "int8" => Ok(DtypeCode::Int8),
        "uint16" => Ok(DtypeCode::Uint16),
        "int16" => Ok(DtypeCode::Int16),
        "uint32" => Ok(DtypeCode::Uint32),
        "int32" => Ok(DtypeCode::Int32),
        "uint64" => Ok(DtypeCode::Uint64),
        "int64" => Ok(DtypeCode::Int64),
        "float16" => Ok(DtypeCode::Float16),
        "float32" => Ok(DtypeCode::Float32),
        "float64" => Ok(DtypeCode::Float64),
        "bool" => Ok(DtypeCode::Bool),
        _ => bail!("Unknown openpi_dtype: {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec as prop_vec;
    use proptest::prelude::*;

    /// Generate an arbitrary `DtypeCode`.
    fn arb_dtype() -> impl Strategy<Value = DtypeCode> {
        prop_oneof![
            Just(DtypeCode::Uint8),
            Just(DtypeCode::Int8),
            Just(DtypeCode::Uint16),
            Just(DtypeCode::Int16),
            Just(DtypeCode::Uint32),
            Just(DtypeCode::Int32),
            Just(DtypeCode::Uint64),
            Just(DtypeCode::Int64),
            Just(DtypeCode::Float16),
            Just(DtypeCode::Float32),
            Just(DtypeCode::Float64),
            Just(DtypeCode::Bool),
        ]
    }

    /// Generate a `LocalArray`. Path has 1..=3 ASCII components (the codec
    /// rejects empty paths on both sides). `data` length is arbitrary because
    /// the Arrow codec stores raw bytes — shape/dtype consistency is a
    /// `local_format` concern, not this codec's.
    fn arb_local_array() -> impl Strategy<Value = LocalArray> {
        (
            prop_vec("[a-zA-Z0-9_]{1,12}", 1..=3),
            arb_dtype(),
            prop_vec(0_u32..=6, 0..=4),
            prop_vec(any::<u8>(), 0..=64),
        )
            .prop_map(|(path, dtype, shape, data)| LocalArray {
                path,
                dtype,
                shape,
                data,
            })
    }

    /// Generate a `LocalFrame`. `arrays` can be empty (this is the planner
    /// case) and `scalar_json` must be UTF-8 since the codec rejects non-UTF-8
    /// scalar blobs.
    fn arb_local_frame() -> impl Strategy<Value = LocalFrame> {
        (
            "[a-zA-Z0-9_]{0,20}",
            prop_vec(arb_local_array(), 0..=4),
            ".{0,64}",
        )
            .prop_map(|(schema_id, arrays, scalar_json)| LocalFrame {
                schema_id,
                arrays,
                scalar_json: scalar_json.into_bytes(),
            })
    }

    proptest! {
        #[test]
        fn round_trip_any_frame(frame in arb_local_frame()) {
            let encoded = encode_arrow_ipc(&frame).expect("encode should succeed");
            let decoded = decode_arrow_ipc(&encoded).expect("decode should succeed");
            prop_assert_eq!(decoded, frame);
        }
    }

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
    fn round_trip_arrow_ipc() {
        let frame = sample_frame();
        let encoded = encode_arrow_ipc(&frame).expect("encode");
        let decoded = decode_arrow_ipc(&encoded).expect("decode");

        assert_eq!(decoded.schema_id, frame.schema_id);
        assert_eq!(decoded.scalar_json, frame.scalar_json);
        assert_eq!(decoded.arrays.len(), frame.arrays.len());
        for (decoded_array, original_array) in decoded.arrays.iter().zip(frame.arrays.iter()) {
            assert_eq!(decoded_array.path, original_array.path);
            assert_eq!(decoded_array.dtype, original_array.dtype);
            assert_eq!(decoded_array.shape, original_array.shape);
            assert_eq!(decoded_array.data, original_array.data);
        }
    }

    #[test]
    fn round_trip_zero_array_frame() {
        // Planner subtask_only responses carry zero numpy arrays — everything
        // lives in scalar_json. The codec must still produce a valid single-row
        // RecordBatch so the response can flow over QUIC.
        let frame = LocalFrame {
            schema_id: "planner".to_owned(),
            arrays: vec![],
            scalar_json: br#"{"subtask":{"text":"pick up cup","ms":442.3}}"#.to_vec(),
        };
        let encoded = encode_arrow_ipc(&frame).expect("encode");
        let decoded = decode_arrow_ipc(&encoded).expect("decode");
        assert_eq!(decoded.schema_id, frame.schema_id);
        assert_eq!(decoded.scalar_json, frame.scalar_json);
        assert!(decoded.arrays.is_empty());
    }

    #[test]
    fn preserves_nested_paths() {
        let frame = LocalFrame {
            schema_id: "aloha".to_owned(),
            arrays: vec![
                LocalArray {
                    path: vec!["images".to_owned(), "cam_high".to_owned()],
                    dtype: DtypeCode::Uint8,
                    shape: vec![3, 2, 2],
                    data: (0..12_u8).collect(),
                },
                LocalArray {
                    path: vec!["state".to_owned()],
                    dtype: DtypeCode::Float32,
                    shape: vec![14],
                    data: vec![0_u8; 14 * 4],
                },
            ],
            scalar_json: b"{}".to_vec(),
        };
        let bytes = encode_arrow_ipc(&frame).expect("encode");
        let decoded = decode_arrow_ipc(&bytes).expect("decode");
        assert_eq!(decoded.arrays[0].path, vec!["images", "cam_high"]);
        assert_eq!(decoded.arrays[1].path, vec!["state"]);
    }
}
