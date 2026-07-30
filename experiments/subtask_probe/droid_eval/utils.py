"""Shared utilities for the DROID subtask evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import DROID_ACTION_DIM, MODEL_ACTION_DIM, InferenceMode


def load_manifest(samples_dir: Path) -> list[dict[str, Any]]:
    """Load the episode manifest from a samples directory."""
    with (samples_dir / "manifest.json").open() as f:
        return json.load(f)


def load_subtask_records(path: Path) -> list[dict[str, Any]]:
    """Load the raw list of subtask records from a subtasks JSON file.

    Accepts two on-disk shapes:
      * Legacy: a bare JSON list of records.
      * Current: ``{"prompt_format": "...", "results": [...]}`` — written by
        the prompt-format-aware ``generate_subtasks.py``.
    """
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"Unrecognized subtask JSON shape in {path}: expected list or dict with 'results' key"
    )


def load_subtask_index(path: Path) -> dict[tuple[str, int], str]:
    """Load subtask results and index by (episode_id, frame_idx) -> subtask_text."""
    return {
        (entry["episode_id"], entry["frame_idx"]): entry["subtask_text"]
        for entry in load_subtask_records(path)
    }


def load_subtask_entries(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load subtask results and index by (episode_id, frame_idx) -> full entry dict."""
    return {
        (entry["episode_id"], entry["frame_idx"]): entry for entry in load_subtask_records(path)
    }


def build_subtask_observation(
    exterior_image: np.ndarray,
    wrist_image: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Build an observation dict for subtask generation (mode="subtask_only").

    Images are sent as raw uint8 — the server's _normalize_image() handles
    conversion to float32 [-1, 1] and the camera name mapping.
    """
    return {
        "images": {
            "base_0_rgb": exterior_image,
            "left_wrist_0_rgb": wrist_image,
        },
        "state": np.zeros(14, dtype=np.float32),
        "prompt": prompt,
        "mode": "subtask_only",
    }


def build_action_observation(
    exterior_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    prompt: str,
    noise: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build an observation dict for action generation (mode="action_only").

    The server's pi05_droid policy transforms handle normalization,
    tokenization, and image preprocessing internally.

    Args:
        noise: Pre-generated noise tensor for flow matching denoising,
            shape (ACTION_HORIZON, MODEL_ACTION_DIM). When the same noise is
            passed for multiple prompt conditions, actions differ only due to
            the prompt, not random noise.
    """
    joint_position = state[:7]
    gripper_position = state[7:8]

    obs: dict[str, Any] = {
        "observation/exterior_image_1_left": exterior_image,
        "observation/wrist_image_left": wrist_image,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
        "mode": "action_only",
    }
    if noise is not None:
        obs["noise"] = noise
    return obs


def build_warmup_observation(mode: InferenceMode = "action_only") -> dict[str, Any]:
    """Build a dummy observation for server warmup."""
    if mode == "subtask_only":
        return build_subtask_observation(
            exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
            wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
            prompt="warmup",
        )
    return build_action_observation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.zeros(DROID_ACTION_DIM, dtype=np.float32),
        prompt="warmup",
    )


def generate_frame_noise(episode_id: str, frame_idx: int) -> np.ndarray:
    """Generate deterministic noise for flow matching denoising.

    Uses a hash of (episode_id, frame_idx) as the seed so that multiple
    prompt conditions on the same frame get identical noise, making the
    comparison fair.
    """
    from .constants import ACTION_HORIZON

    rng = np.random.RandomState(hash((episode_id, frame_idx)) % (2**31))
    return rng.randn(ACTION_HORIZON, MODEL_ACTION_DIM).astype(np.float32)


def decode_droid_image(img_bytes: bytes | str | np.ndarray) -> np.ndarray:
    """Decode a DROID image from RLDS format.

    Handles both encoded (JPEG bytes) and pre-decoded (ndarray) formats
    that appear in different DROID dataset versions.
    """
    import tensorflow as tf  # ty: ignore[unresolved-import]

    if isinstance(img_bytes, (bytes, str)) or (
        hasattr(img_bytes, "dtype")
        and (img_bytes.dtype == np.object_ or img_bytes.dtype.kind in ("S", "U"))
    ):
        return tf.io.decode_image(img_bytes, expand_animations=False, dtype=tf.uint8).numpy()
    return np.asarray(img_bytes, dtype=np.uint8)
