//! Image resize and dtype conversion on the client side of openpi-flash-transport.
//!
//! Mirrors `openpi_client.image_tools.resize_with_pad` (BILINEAR resize with
//! aspect-preserving zero-padding) and `convert_to_uint8` (float [0, 1] →
//! uint8 [0, 255]). Driven by `image_specs` advertised in the server's
//! handshake metadata so customers can stop calling these in Python.
//!
//! Supports HWC (height, width, 3) and CHW (3, height, width) layouts; the
//! `target_shape` in the server-advertised spec determines which one we emit.
//! Resizing is always done in HWC internally because that's what
//! `fast_image_resize` operates on.

use anyhow::{anyhow, bail, Context, Result};
use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use ndarray::{s, Array3, ArrayView3};

use crate::local_format::{DtypeCode, LocalArray};

/// Subset of numpy dtypes the image preprocessing pipeline accepts. Narrower
/// than `DtypeCode` on purpose: `ImageSpec` only carries values for which
/// resize and dtype conversion are implemented, so bad specs are rejected at
/// construction (e.g. in `metadata::parse_image_specs`) rather than at
/// request time inside `maybe_preprocess`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageDtype {
    Uint8,
    Float32,
}

impl ImageDtype {
    /// Lift a general `DtypeCode` into `ImageDtype` when the image pipeline
    /// can handle it. Returns `None` for dtypes we can't process.
    #[must_use]
    pub fn from_dtype_code(code: DtypeCode) -> Option<Self> {
        match code {
            DtypeCode::Uint8 => Some(Self::Uint8),
            DtypeCode::Float32 => Some(Self::Float32),
            _ => None,
        }
    }

    #[must_use]
    pub fn to_dtype_code(self) -> DtypeCode {
        match self {
            Self::Uint8 => DtypeCode::Uint8,
            Self::Float32 => DtypeCode::Float32,
        }
    }
}

/// Per-field image preprocessing rule advertised by the server in metadata.
#[derive(Clone, Debug)]
pub struct ImageSpec {
    /// Nested dict path for the field, matching `LocalArray::path`.
    pub path: Vec<String>,
    /// Desired shape after preprocessing. Either HWC `[H, W, 3]` or CHW
    /// `[3, H, W]`. The leading dimension is interpreted as channels when
    /// it's exactly 3 and the trailing dim is not.
    pub target_shape: [u32; 3],
    /// Desired dtype after preprocessing. Narrowed to `ImageDtype` so specs
    /// can't advertise a dtype we don't know how to produce (e.g. `int64`).
    pub target_dtype: ImageDtype,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageLayout {
    Hwc,
    Chw,
}

impl ImageLayout {
    fn detect(shape: [u32; 3]) -> Result<Self> {
        let [a, _b, c] = shape;
        // Prefer HWC when the trailing dim is the channel count, fall back to
        // CHW. Both ALOHA (CHW) and DROID (HWC) examples exist in openpi.
        // 3x*x3 is ambiguous; we treat it as HWC because DROID is more common.
        match (a == 3, c == 3) {
            (_, true) => Ok(Self::Hwc),
            (true, false) => Ok(Self::Chw),
            (false, false) => bail!("Image shape {shape:?} has no 3-channel axis"),
        }
    }
}

/// Preprocess an image array to match the spec's target shape and dtype.
///
/// Returns `Ok(None)` when the array already matches the spec (no work
/// needed). Otherwise returns a fresh `LocalArray` with the new shape, dtype,
/// and reshaped data buffer.
///
/// # Errors
/// Returns `Err` when the array is not 3-D, when its source dtype isn't one
/// the image pipeline supports (`Uint8`, `Float32`), or when the source or
/// target shape has no 3-channel axis to interpret as HWC/CHW.
pub fn maybe_preprocess(array: &LocalArray, spec: &ImageSpec) -> Result<Option<LocalArray>> {
    if array.shape.len() != 3 {
        bail!(
            "Image preprocess for {:?}: expected 3-D array, got shape {:?}",
            spec.path,
            array.shape
        );
    }
    let source_shape = [array.shape[0], array.shape[1], array.shape[2]];
    let source_dtype = ImageDtype::from_dtype_code(array.dtype).ok_or_else(|| {
        anyhow!(
            "Unsupported source dtype {:?} for image {:?}",
            array.dtype,
            spec.path
        )
    })?;

    if source_shape == spec.target_shape && source_dtype == spec.target_dtype {
        return Ok(None);
    }

    let source_layout = ImageLayout::detect(source_shape)?;
    let target_layout = ImageLayout::detect(spec.target_shape)?;

    let (source_height, source_width) = match source_layout {
        ImageLayout::Hwc => (source_shape[0], source_shape[1]),
        ImageLayout::Chw => (source_shape[1], source_shape[2]),
    };
    let (target_height, target_width) = match target_layout {
        ImageLayout::Hwc => (spec.target_shape[0], spec.target_shape[1]),
        ImageLayout::Chw => (spec.target_shape[1], spec.target_shape[2]),
    };

    let hwc_uint8 = to_hwc_uint8(&array.data, source_dtype, source_layout, source_shape)?;

    let resized_hwc_array = if (source_height, source_width) == (target_height, target_width) {
        ArrayView3::<u8>::from_shape(
            (source_height as usize, source_width as usize, 3),
            &hwc_uint8,
        )
        .context("Failed to view HWC buffer as 3-D array")?
        .to_owned()
    } else {
        resize_with_pad_bilinear_hwc(
            &hwc_uint8,
            source_height,
            source_width,
            target_height,
            target_width,
        )
        .context("BILINEAR resize-with-pad failed")?
    };

    let final_data = match target_layout {
        ImageLayout::Hwc => array_to_vec(resized_hwc_array.view()),
        ImageLayout::Chw => array_to_vec(resized_hwc_array.permuted_axes([2, 0, 1]).view()),
    };

    Ok(Some(LocalArray {
        path: array.path.clone(),
        dtype: spec.target_dtype.to_dtype_code(),
        shape: spec.target_shape.to_vec(),
        data: final_data,
    }))
}

/// Aspect-ratio preserving resize with zero (black) padding to fit a target
/// `(out_height, out_width)`. Mirrors `tf.image.resize_with_pad` /
/// `openpi_client.image_tools.resize_with_pad` semantics.
///
/// Returns an [`Array3<u8>`] of shape `(out_height, out_width, 3)` in HWC
/// standard layout so callers can transpose to CHW with `.permuted_axes`.
fn resize_with_pad_bilinear_hwc(
    hwc: &[u8],
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
) -> Result<Array3<u8>> {
    let expected_len = (in_h as usize) * (in_w as usize) * 3;
    if hwc.len() != expected_len {
        bail!(
            "HWC buffer length mismatch: expected {expected_len}, got {}",
            hwc.len()
        );
    }

    let scale = f64::min(
        f64::from(out_h) / f64::from(in_h),
        f64::from(out_w) / f64::from(in_w),
    );
    let scaled_h = scale_dim_to_u32(in_h, scale);
    let scaled_w = scale_dim_to_u32(in_w, scale);

    let src_image = FirImage::from_vec_u8(in_w, in_h, hwc.to_vec(), PixelType::U8x3)
        .map_err(|e| anyhow!("fast_image_resize source image build failed: {e}"))?;
    let mut dst_image = FirImage::new(scaled_w, scaled_h, PixelType::U8x3);

    let mut resizer = Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|e| anyhow!("fast_image_resize resize failed: {e}"))?;
    let resized_bytes = dst_image.into_vec();

    let resized_view =
        ArrayView3::<u8>::from_shape((scaled_h as usize, scaled_w as usize, 3), &resized_bytes)
            .context("Failed to view resized buffer as 3-D array")?;

    let mut padded = Array3::<u8>::zeros((out_h as usize, out_w as usize, 3));
    let pad_top = ((out_h - scaled_h) / 2) as usize;
    let pad_left = ((out_w - scaled_w) / 2) as usize;
    padded
        .slice_mut(s![
            pad_top..pad_top + scaled_h as usize,
            pad_left..pad_left + scaled_w as usize,
            ..
        ])
        .assign(&resized_view);
    Ok(padded)
}

fn to_hwc_uint8(
    data: &[u8],
    dtype: ImageDtype,
    layout: ImageLayout,
    shape: [u32; 3],
) -> Result<Vec<u8>> {
    let (height, width) = match layout {
        ImageLayout::Hwc => (shape[0] as usize, shape[1] as usize),
        ImageLayout::Chw => (shape[1] as usize, shape[2] as usize),
    };
    let pixel_count = height * width * 3;

    let uint8_buffer: Vec<u8> = match dtype {
        ImageDtype::Uint8 => data.to_vec(),
        ImageDtype::Float32 => {
            let expected_bytes = pixel_count * 4;
            if data.len() != expected_bytes {
                bail!(
                    "Float32 image buffer length mismatch: expected {expected_bytes}, got {}",
                    data.len()
                );
            }
            data.chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    unit_float_to_u8(value)
                })
                .collect()
        }
    };

    Ok(match layout {
        ImageLayout::Hwc => uint8_buffer,
        ImageLayout::Chw => {
            // Transpose CHW → HWC by viewing the buffer as a (3, h, w) array
            // and permuting axes; ndarray handles the index math.
            let chw_view = ArrayView3::<u8>::from_shape((3, height, width), &uint8_buffer)
                .context("Failed to interpret CHW buffer as 3-D array")?;
            array_to_vec(chw_view.permuted_axes([1, 2, 0]))
        }
    })
}

/// Scale `source_dim` (a u32) by a scale factor and return the rounded
/// result as a u32, at least 1.
///
/// Aspect-preserving resize produces `scale ∈ (0, 1]`, so the scaled
/// dimension is always within `[0, source_dim]`. Clamping the f64 against
/// `[0.0, u32::MAX as f64]` before casting is what turns a lossy
/// `as u32` into a provably bounded conversion.
fn scale_dim_to_u32(source_dim: u32, scale: f64) -> u32 {
    let scaled = (f64::from(source_dim) * scale).round();
    let clamped = scaled.clamp(0.0, f64::from(u32::MAX));
    // SAFETY of cast: `clamped` is guaranteed non-negative and at most
    // `u32::MAX` by the clamp above.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dim = clamped as u32;
    dim.max(1)
}

/// Convert a unit-range float (clamped to `[0.0, 1.0]`) into a u8 on the
/// 0..=255 scale. Mirrors `openpi_client.image_tools.convert_to_uint8`.
fn unit_float_to_u8(value: f32) -> u8 {
    let scaled_0_255 = (value.clamp(0.0, 1.0) * 255.0).round();
    // SAFETY of cast: clamp guarantees `scaled_0_255 ∈ [0.0, 255.0]`, which
    // fits in u8 without sign or truncation surprises.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let byte = scaled_0_255 as u8;
    byte
}

/// Materialize an [`ArrayView3<u8>`] into a contiguous `Vec<u8>` in standard
/// (row-major) layout, copying through the iterator so a permuted view comes
/// out laid out by its new axis order.
fn array_to_vec<V: ndarray::Data<Elem = u8>>(view: ndarray::ArrayBase<V, ndarray::Ix3>) -> Vec<u8> {
    view.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_hwc(height: u32, width: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut data = Vec::with_capacity((height * width * 3) as usize);
        for _ in 0..(height * width) {
            data.extend_from_slice(&rgb);
        }
        data
    }

    #[test]
    fn no_op_when_already_matching() {
        let array = LocalArray {
            path: vec!["img".to_owned()],
            dtype: DtypeCode::Uint8,
            shape: vec![4, 4, 3],
            data: solid_hwc(4, 4, [1, 2, 3]),
        };
        let spec = ImageSpec {
            path: vec!["img".to_owned()],
            target_shape: [4, 4, 3],
            target_dtype: ImageDtype::Uint8,
        };
        assert!(maybe_preprocess(&array, &spec).unwrap().is_none());
    }

    #[test]
    fn resize_smaller_pads_with_zeros() {
        // A 4x8 HWC solid red input resized to 4x4 should letterbox: the
        // resized region preserves aspect, and the rest is zero-padded.
        let array = LocalArray {
            path: vec!["img".to_owned()],
            dtype: DtypeCode::Uint8,
            shape: vec![4, 8, 3],
            data: solid_hwc(4, 8, [255, 0, 0]),
        };
        let spec = ImageSpec {
            path: vec!["img".to_owned()],
            target_shape: [4, 4, 3],
            target_dtype: ImageDtype::Uint8,
        };
        let result = maybe_preprocess(&array, &spec).unwrap().unwrap();
        assert_eq!(result.shape, vec![4, 4, 3]);
        // After scale=4/8=0.5, scaled = 2x4. Padded into 4x4: top/bottom
        // padding 1 row each, no horizontal padding. Top and bottom rows
        // should be black; middle two rows should be red.
        let row_size = 4 * 3;
        let top_row = &result.data[0..row_size];
        let bottom_row = &result.data[3 * row_size..4 * row_size];
        assert!(top_row.iter().all(|&b| b == 0));
        assert!(bottom_row.iter().all(|&b| b == 0));
        let middle_row = &result.data[row_size..2 * row_size];
        assert!(middle_row.iter().enumerate().all(|(i, &b)| {
            // RGB triplets — index % 3 == 0 should be red, others zero.
            if i % 3 == 0 {
                b == 255
            } else {
                b == 0
            }
        }));
    }

    #[test]
    fn chw_input_emits_chw_output() {
        // 3-channel CHW input, target also CHW, after a no-op resize.
        let mut chw_data = Vec::with_capacity(3 * 2 * 2);
        chw_data.extend_from_slice(&[10, 20, 30, 40]); // R plane
        chw_data.extend_from_slice(&[50, 60, 70, 80]); // G plane
        chw_data.extend_from_slice(&[90, 100, 110, 120]); // B plane
        let array = LocalArray {
            path: vec!["img".to_owned()],
            dtype: DtypeCode::Uint8,
            shape: vec![3, 2, 2],
            data: chw_data.clone(),
        };
        let spec = ImageSpec {
            path: vec!["img".to_owned()],
            target_shape: [3, 2, 2],
            target_dtype: ImageDtype::Uint8,
        };
        // Same shape and dtype — should be a no-op.
        assert!(maybe_preprocess(&array, &spec).unwrap().is_none());
    }

    #[test]
    fn float32_input_converts_to_uint8() {
        // Single 1x1 image with float32 RGB (0.0, 0.5, 1.0) → uint8 (0, 128, 255).
        let mut data = Vec::new();
        data.extend_from_slice(&0.0_f32.to_le_bytes());
        data.extend_from_slice(&0.5_f32.to_le_bytes());
        data.extend_from_slice(&1.0_f32.to_le_bytes());
        let array = LocalArray {
            path: vec!["img".to_owned()],
            dtype: DtypeCode::Float32,
            shape: vec![1, 1, 3],
            data,
        };
        let spec = ImageSpec {
            path: vec!["img".to_owned()],
            target_shape: [1, 1, 3],
            target_dtype: ImageDtype::Uint8,
        };
        let result = maybe_preprocess(&array, &spec).unwrap().unwrap();
        assert_eq!(result.dtype, DtypeCode::Uint8);
        assert_eq!(result.data, vec![0, 128, 255]);
    }
}
