"""Round-trip tests for the Python local-frame codec."""

from __future__ import annotations

import numpy as np
import pytest

from hosting.local_frame import pack_local_frame, unpack_local_frame


def test_round_trip_droid_observation() -> None:
    observation = {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7).astype(np.float64),
        "observation/gripper_position": np.random.rand(1).astype(np.float64),
        "prompt": "pick up the red block",
    }
    frame = pack_local_frame(observation, schema_id="droid")
    decoded = unpack_local_frame(frame)

    assert decoded["prompt"] == "pick up the red block"
    for key in [
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
    ]:
        np.testing.assert_array_equal(decoded[key], observation[key])


def test_round_trip_aloha_observation_with_nested_images() -> None:
    observation = {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }
    frame = pack_local_frame(observation, schema_id="aloha")
    decoded = unpack_local_frame(frame)

    assert decoded["prompt"] == "do something"
    np.testing.assert_array_equal(decoded["state"], observation["state"])
    for cam in ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]:
        np.testing.assert_array_equal(decoded["images"][cam], observation["images"][cam])


def test_round_trip_action_chunk_with_server_timing() -> None:
    action = {
        "actions": np.random.rand(10, 8).astype(np.float32),
        "server_timing": {"infer_ms": 42.5, "prev_total_ms": 150.0},
        "policy_timing": {"infer_ms": 38.0},
    }
    frame = pack_local_frame(action, schema_id="droid_action")
    decoded = unpack_local_frame(frame)

    np.testing.assert_array_equal(decoded["actions"], action["actions"])
    assert decoded["server_timing"] == action["server_timing"]
    assert decoded["policy_timing"] == action["policy_timing"]


def test_rejects_unsupported_dtype() -> None:
    observation = {"bad": np.array([1 + 2j], dtype=np.complex64)}
    with pytest.raises(ValueError, match="Unsupported numpy dtype"):
        pack_local_frame(observation)


def test_rejects_non_string_key() -> None:
    observation: dict = {1: np.zeros(3)}
    with pytest.raises(TypeError, match="must be str"):
        pack_local_frame(observation)


def test_rejects_trailing_bytes() -> None:
    frame = pack_local_frame({"x": np.zeros(1, dtype=np.uint8)})
    with pytest.raises(ValueError, match="Trailing"):
        unpack_local_frame(frame + b"\x00")


def test_preserves_non_contiguous_arrays() -> None:
    """Non-contiguous arrays must be materialized before encode."""
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    non_contiguous_view = source.T  # Fortran-order view of a C-order array.
    observation = {"state": non_contiguous_view}

    frame = pack_local_frame(observation)
    decoded = unpack_local_frame(frame)

    np.testing.assert_array_equal(decoded["state"], non_contiguous_view)
