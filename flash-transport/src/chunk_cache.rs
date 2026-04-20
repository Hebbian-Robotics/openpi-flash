//! Action-chunk caching on the client side of openpi-flash-transport.
//!
//! Servers return a full action chunk per inference call (e.g. 10 or 50
//! steps). Today, customers wrap the policy in
//! `openpi_client.action_chunk_broker.ActionChunkBroker` to serve one step
//! per Python `infer()` call. This module replicates that behavior inside
//! the client transport so the customer can drop the wrapper, and so the
//! QUIC path finally has chunking (it never did before).
//!
//! Chunking is opt-in via `OPENPI_OPEN_LOOP_HORIZON` (set by the user) and
//! requires the server to advertise an `action_horizon` in metadata.
//!
//! The cache is modeled as an explicit state machine: a cache is either
//! `Empty` (no chunk cached, caller must fetch) or `Loaded` (chunk cached,
//! some steps remaining). That state is internal — the public API exposes
//! it as [`StepResult`], which has variants for "next step served" and
//! "refresh needed" so callers can't accidentally pop an exhausted cache.

use std::num::NonZeroUsize;

use anyhow::{bail, Result};

use crate::local_format::{LocalArray, LocalFrame};

/// Cache for one inference round-trip's worth of action steps.
///
/// `action_horizon` is the leading dimension of chunked arrays in the
/// server's response. `open_loop_horizon` is how many of those steps we
/// serve before fetching a fresh chunk; capped at `action_horizon`.
/// Both are `NonZeroUsize` so the "at least one step per chunk"
/// invariant is proven at the type level — callers can't pass 0 and the
/// cache never has to defensively clamp.
pub struct ChunkCache {
    cached: Option<CachedChunk>,
    open_loop_horizon: NonZeroUsize,
    action_horizon: NonZeroUsize,
}

/// A loaded chunk paired with the step index we'll serve next. Keeping
/// these together makes it impossible to drift (e.g. a non-zero
/// `next_step_index` without a frame) — the old representation allowed
/// that and relied on a `needs_refresh` boolean to smooth over it.
struct CachedChunk {
    frame: LocalFrame,
    next_step_index: usize,
}

/// Outcome of requesting the next chunk step.
///
/// Encoded as a variant rather than an `Option<LocalFrame>` so it's
/// self-documenting at call sites: `RefreshNeeded` is a recognized
/// control-flow branch, not an ambiguous `None`. Matches Example 2
/// (state variants over implicit boolean combinations) from the style
/// guide.
pub enum StepResult {
    /// A single-step slice of the cached chunk. The caller hands this to
    /// Python directly.
    Served(LocalFrame),
    /// Cache is empty or has run out of steps within the open-loop
    /// horizon. The caller must fetch a fresh chunk from the server, store
    /// it via [`ChunkCache::store`], and request again.
    RefreshNeeded,
}

impl ChunkCache {
    #[must_use]
    pub fn new(action_horizon: NonZeroUsize, open_loop_horizon: NonZeroUsize) -> Self {
        // Can only serve as many steps as the chunk actually contains, so
        // cap open_loop_horizon at action_horizon. `NonZeroUsize::min`
        // preserves the non-zero guarantee because both inputs are >= 1.
        let bounded_open_loop = open_loop_horizon.min(action_horizon);
        Self {
            cached: None,
            open_loop_horizon: bounded_open_loop,
            action_horizon,
        }
    }

    /// Store a freshly fetched chunk; resets the step counter to zero.
    pub fn store(&mut self, frame: LocalFrame) {
        self.cached = Some(CachedChunk {
            frame,
            next_step_index: 0,
        });
    }

    /// Drop any cached chunk and start asking the server for fresh chunks
    /// again. Called from the sidecar's Reset handler.
    pub fn reset(&mut self) {
        self.cached = None;
    }

    /// Return the next step of the cached chunk, or signal that a refresh
    /// is needed. `Err` is reserved for genuine slicing failures (malformed
    /// frame), not for "cache empty" — that's `Ok(StepResult::RefreshNeeded)`.
    ///
    /// # Errors
    /// Returns `Err` when a chunked array in the cached frame has an invalid
    /// shape or a data buffer shorter than its declared shape requires.
    pub fn next_step(&mut self) -> Result<StepResult> {
        let Some(cached) = self.cached.as_mut() else {
            return Ok(StepResult::RefreshNeeded);
        };
        if cached.next_step_index >= self.open_loop_horizon.get() {
            // Exhausted this chunk's open-loop window. Drop the cache so a
            // subsequent `store()` replaces it cleanly.
            self.cached = None;
            return Ok(StepResult::RefreshNeeded);
        }
        let sliced = slice_frame_at_step(
            &cached.frame,
            cached.next_step_index,
            self.action_horizon.get(),
        )?;
        cached.next_step_index += 1;
        Ok(StepResult::Served(sliced))
    }
}

/// Slice a [`LocalFrame`] at a single step. Arrays whose leading dim equals
/// `action_horizon` get sliced to step `step_index` (leading dim removed);
/// other arrays pass through unchanged. Scalar JSON is preserved.
fn slice_frame_at_step(
    frame: &LocalFrame,
    step_index: usize,
    action_horizon: usize,
) -> Result<LocalFrame> {
    let mut sliced_arrays = Vec::with_capacity(frame.arrays.len());
    for array in &frame.arrays {
        // Compare as usize so we don't have to cast action_horizon down to
        // u32 (which would risk truncation on 64-bit platforms). u32 → usize
        // is widening on every target we care about (std does not provide a
        // generic `From<u32> for usize` because 16-bit MCUs exist, but we
        // don't run there).
        let leading_dim = array.shape.first().copied().map(|d| d as usize);
        if leading_dim == Some(action_horizon) {
            sliced_arrays.push(slice_array_at_step(array, step_index)?);
        } else {
            sliced_arrays.push(array.clone());
        }
    }
    Ok(LocalFrame {
        schema_id: frame.schema_id.clone(),
        arrays: sliced_arrays,
        scalar_json: frame.scalar_json.clone(),
    })
}

fn slice_array_at_step(array: &LocalArray, step_index: usize) -> Result<LocalArray> {
    if array.shape.is_empty() {
        bail!(
            "Cannot slice 0-D array {:?} at step {step_index}",
            array.path
        );
    }
    let leading_dim = array.shape[0] as usize;
    if step_index >= leading_dim {
        bail!(
            "step_index {step_index} out of bounds for leading dim {leading_dim} on {:?}",
            array.path
        );
    }

    let element_size = array.dtype.element_size_bytes();
    let bytes_per_step: usize = array
        .shape
        .iter()
        .skip(1)
        .map(|&dim| dim as usize)
        .product::<usize>()
        * element_size;

    let start = step_index * bytes_per_step;
    let end = start + bytes_per_step;
    if end > array.data.len() {
        bail!(
            "Sliced byte range {start}..{end} exceeds data length {} on {:?}",
            array.data.len(),
            array.path
        );
    }

    Ok(LocalArray {
        path: array.path.clone(),
        dtype: array.dtype,
        shape: array.shape[1..].to_vec(),
        data: array.data[start..end].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_format::DtypeCode;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test helper: value must be > 0")
    }

    fn make_action_frame(action_horizon: u16, dof: u16) -> LocalFrame {
        // actions: float32 [horizon, dof] with deterministic content so tests
        // can verify the right step was sliced. Using u16 for indices lets
        // us lift into f32 with `f32::from` (lossless for u16 values) rather
        // than a lossy `as f32` cast.
        let total_elements = usize::from(action_horizon) * usize::from(dof);
        let mut data: Vec<u8> = Vec::with_capacity(total_elements * 4);
        for step in 0..action_horizon {
            for joint in 0..dof {
                let value = f32::from(step) * 100.0 + f32::from(joint);
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        LocalFrame {
            schema_id: "test".to_owned(),
            arrays: vec![LocalArray {
                path: vec!["actions".to_owned()],
                dtype: DtypeCode::Float32,
                shape: vec![u32::from(action_horizon), u32::from(dof)],
                data,
            }],
            scalar_json: br#"{"policy_timing":{"infer_ms":7.5}}"#.to_vec(),
        }
    }

    fn expect_served(result: StepResult) -> LocalFrame {
        match result {
            StepResult::Served(frame) => frame,
            StepResult::RefreshNeeded => {
                panic!("expected StepResult::Served, got RefreshNeeded")
            }
        }
    }

    fn read_action_step(sliced: &LocalFrame) -> Vec<f32> {
        let array = &sliced.arrays[0];
        assert_eq!(array.shape, vec![3]);
        array
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn empty_cache_asks_for_refresh() {
        let mut cache = ChunkCache::new(nz(5), nz(3));
        assert!(matches!(
            cache.next_step().unwrap(),
            StepResult::RefreshNeeded
        ));
    }

    #[test]
    fn serves_chunk_steps_in_order_then_refreshes() {
        let mut cache = ChunkCache::new(nz(5), nz(3));
        cache.store(make_action_frame(5, 3));
        assert_eq!(
            read_action_step(&expect_served(cache.next_step().unwrap())),
            vec![0.0, 1.0, 2.0]
        );
        assert_eq!(
            read_action_step(&expect_served(cache.next_step().unwrap())),
            vec![100.0, 101.0, 102.0]
        );
        assert_eq!(
            read_action_step(&expect_served(cache.next_step().unwrap())),
            vec![200.0, 201.0, 202.0]
        );
        // Hit open_loop_horizon, cache is drained — should ask for a refresh.
        assert!(matches!(
            cache.next_step().unwrap(),
            StepResult::RefreshNeeded
        ));
    }

    #[test]
    fn reset_clears_cache() {
        let mut cache = ChunkCache::new(nz(5), nz(3));
        cache.store(make_action_frame(5, 3));
        let _ = cache.next_step().unwrap();
        cache.reset();
        assert!(matches!(
            cache.next_step().unwrap(),
            StepResult::RefreshNeeded
        ));
    }

    #[test]
    fn open_loop_horizon_capped_at_action_horizon() {
        let cache = ChunkCache::new(nz(5), nz(100));
        assert_eq!(cache.open_loop_horizon, nz(5));
    }

    #[test]
    fn passes_through_arrays_with_non_horizon_leading_dim() {
        let mut frame = make_action_frame(5, 3);
        // Add a metadata array shaped [4] that should NOT be chunked.
        frame.arrays.push(LocalArray {
            path: vec!["metadata".to_owned()],
            dtype: DtypeCode::Uint8,
            shape: vec![4],
            data: vec![10, 20, 30, 40],
        });

        let mut cache = ChunkCache::new(nz(5), nz(5));
        cache.store(frame);
        let sliced = expect_served(cache.next_step().unwrap());

        // actions sliced to step 0
        assert_eq!(sliced.arrays[0].shape, vec![3]);
        // metadata passed through unchanged
        assert_eq!(sliced.arrays[1].shape, vec![4]);
        assert_eq!(sliced.arrays[1].data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn refresh_after_exhaustion_accepts_new_chunk() {
        // After draining a chunk the cache must accept a fresh one and
        // start serving from step 0 again.
        let mut cache = ChunkCache::new(nz(5), nz(2));
        cache.store(make_action_frame(5, 3));
        let _ = cache.next_step().unwrap();
        let _ = cache.next_step().unwrap();
        assert!(matches!(
            cache.next_step().unwrap(),
            StepResult::RefreshNeeded
        ));
        cache.store(make_action_frame(5, 3));
        assert_eq!(
            read_action_step(&expect_served(cache.next_step().unwrap())),
            vec![0.0, 1.0, 2.0]
        );
    }
}
