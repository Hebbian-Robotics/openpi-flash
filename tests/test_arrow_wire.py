"""Cross-language round-trip tests for the Arrow-wire codec.

The Python side packs a local frame, pipes it through the Rust helper
binary ``openpi-arrow-wire-echo`` (which runs it through
``decode_local_frame`` → ``encode_arrow_ipc`` → ``decode_arrow_ipc`` →
``encode_local_frame``), then the Python side unpacks the echoed bytes and
asserts they match the original payload. This catches any divergence
between the Python and Rust implementations of the binary framing.

The helper binary is built by ``cargo build`` in ``flash-transport/``; tests
are skipped automatically if the binary is missing.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest

from hosting.local_frame import pack_local_frame, unpack_local_frame


def _hosting_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _candidate_echo_binary_paths() -> list[pathlib.Path]:
    root = _hosting_repo_root()
    return [
        root / "flash-transport" / "target" / "debug" / "openpi-arrow-wire-echo",
        root / "flash-transport" / "target" / "release" / "openpi-arrow-wire-echo",
    ]


def _resolve_echo_binary() -> pathlib.Path | None:
    for candidate in _candidate_echo_binary_paths():
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def echo_binary_path() -> pathlib.Path:
    resolved = _resolve_echo_binary()
    if resolved is None:
        searched = "\n".join(f"  - {p}" for p in _candidate_echo_binary_paths())
        pytest.skip(
            "openpi-arrow-wire-echo binary not built. Run `cargo build` in "
            f"flash-transport/ to enable these tests. Searched:\n{searched}"
        )
    return resolved


def _round_trip_through_rust(
    echo_binary_path: pathlib.Path,
    frame_bytes: bytes,
    *,
    inject_timing: tuple[float, float | None] | None = None,
) -> bytes:
    env = None
    if inject_timing is not None:
        infer_ms, prev_total_ms = inject_timing
        env_value = f"{infer_ms}"
        if prev_total_ms is not None:
            env_value += f",{prev_total_ms}"
        import os

        env = {**os.environ, "OPENPI_ECHO_INJECT_TIMING": env_value}
    result = subprocess.run(
        [str(echo_binary_path)],
        input=frame_bytes,
        capture_output=True,
        check=True,
        timeout=30,
        env=env,
    )
    return result.stdout


def _assert_observation_equal(decoded: dict[str, Any], original: dict[str, Any]) -> None:
    for key, value in original.items():
        assert key in decoded, f"missing key {key!r}"
        if isinstance(value, dict):
            _assert_observation_equal(decoded[key], value)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(decoded[key], value)
        else:
            assert decoded[key] == value


def test_cross_language_droid_observation(echo_binary_path: pathlib.Path) -> None:
    observation = {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7).astype(np.float64),
        "observation/gripper_position": np.random.rand(1).astype(np.float64),
        "prompt": "pick up the red block",
    }
    frame_from_python = pack_local_frame(observation, schema_id="droid")
    echoed_bytes = _round_trip_through_rust(echo_binary_path, frame_from_python)
    decoded = unpack_local_frame(echoed_bytes)
    _assert_observation_equal(decoded, observation)


def test_cross_language_aloha_nested_observation(echo_binary_path: pathlib.Path) -> None:
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
    frame_from_python = pack_local_frame(observation, schema_id="aloha")
    echoed_bytes = _round_trip_through_rust(echo_binary_path, frame_from_python)
    decoded = unpack_local_frame(echoed_bytes)
    _assert_observation_equal(decoded, observation)


def test_cross_language_action_chunk(echo_binary_path: pathlib.Path) -> None:
    action = {
        "actions": np.random.rand(10, 8).astype(np.float32),
        "server_timing": {"infer_ms": 42.5, "prev_total_ms": 150.0},
        "policy_timing": {"infer_ms": 38.0},
    }
    frame_from_python = pack_local_frame(action, schema_id="droid_action")
    echoed_bytes = _round_trip_through_rust(echo_binary_path, frame_from_python)
    decoded = unpack_local_frame(echoed_bytes)
    _assert_observation_equal(decoded, action)


def test_server_timing_injection_overrides_existing(echo_binary_path: pathlib.Path) -> None:
    """The openpi-flash-transport server should overwrite any pre-existing
    server_timing fields and preserve policy_timing untouched."""
    action = {
        "actions": np.random.rand(10, 8).astype(np.float32),
        # Stale server_timing the Python backend might have left in place.
        "server_timing": {"infer_ms": 999.0, "prev_total_ms": 999.0},
        "policy_timing": {"infer_ms": 7.5},
    }
    frame_from_python = pack_local_frame(action, schema_id="droid_action")
    echoed_bytes = _round_trip_through_rust(
        echo_binary_path,
        frame_from_python,
        inject_timing=(42.5, 140.0),
    )
    decoded = unpack_local_frame(echoed_bytes)

    np.testing.assert_array_equal(decoded["actions"], action["actions"])
    assert decoded["server_timing"]["infer_ms"] == 42.5
    assert decoded["server_timing"]["prev_total_ms"] == 140.0
    # policy_timing must not be touched by the Rust injection.
    assert decoded["policy_timing"]["infer_ms"] == 7.5


def test_server_timing_injection_first_request_has_no_prev_total(
    echo_binary_path: pathlib.Path,
) -> None:
    """First request of a connection has no prev_total_ms; injecting only
    infer_ms must leave prev_total_ms absent rather than zero."""
    action = {
        "actions": np.random.rand(10, 8).astype(np.float32),
        "policy_timing": {"infer_ms": 7.5},
    }
    frame_from_python = pack_local_frame(action, schema_id="droid_action")
    echoed_bytes = _round_trip_through_rust(
        echo_binary_path,
        frame_from_python,
        inject_timing=(33.0, None),
    )
    decoded = unpack_local_frame(echoed_bytes)

    assert decoded["server_timing"]["infer_ms"] == 33.0
    assert "prev_total_ms" not in decoded["server_timing"]


def _run_with_image_specs(
    echo_binary_path: pathlib.Path,
    frame_bytes: bytes,
    image_specs: list[dict[str, Any]],
    tmp_path: pathlib.Path,
) -> bytes:
    """Run the echo harness with a Python-built msgpack metadata blob containing
    image_specs. The harness applies preprocessing to matching arrays before
    re-encoding."""
    import os

    import msgpack

    metadata_bytes = msgpack.packb({"image_specs": image_specs}, use_bin_type=True)
    metadata_path = tmp_path / "metadata.msgpack"
    metadata_path.write_bytes(metadata_bytes)

    env = {**os.environ, "OPENPI_ECHO_IMAGE_SPECS_METADATA": str(metadata_path)}
    result = subprocess.run(
        [str(echo_binary_path)],
        input=frame_bytes,
        capture_output=True,
        check=True,
        timeout=30,
        env=env,
    )
    return result.stdout


def test_image_preprocess_resizes_droid_image(
    echo_binary_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Python sends a raw 480x640 DROID camera frame; the Rust echo harness
    runs it through image_preprocess (resize-with-pad to 224x224); Python
    verifies the output shape is correct and pixel values are close to the
    PIL reference."""
    from openpi_client import image_tools

    raw_image = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
    observation = {
        "observation/exterior_image_1_left": raw_image,
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "prompt": "test",
    }
    frame_from_python = pack_local_frame(observation, schema_id="droid")

    image_specs = [
        {
            "path": ["observation/exterior_image_1_left"],
            "target_shape": [224, 224, 3],
            "dtype": "uint8",
        }
    ]
    echoed_bytes = _run_with_image_specs(echo_binary_path, frame_from_python, image_specs, tmp_path)
    decoded = unpack_local_frame(echoed_bytes)

    resized = decoded["observation/exterior_image_1_left"]
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8

    # Compare against PIL reference. fast_image_resize's bilinear filter uses
    # the same convolution algorithm as PIL, but with slightly different
    # floating-point rounding behavior. The strong claim we make here is
    # "no pixel differs by more than 1 unit on a 0-255 scale" — i.e. the
    # outputs are bit-near-exact, the differences are at most last-bit
    # rounding noise. Mean diff is bounded loosely just to catch regressions
    # where the algorithm diverges entirely.
    pil_reference = image_tools.resize_with_pad(raw_image, 224, 224)
    abs_diff = np.abs(resized.astype(np.int16) - pil_reference.astype(np.int16))
    assert abs_diff.max() <= 1, (
        f"Max abs diff {abs_diff.max()} exceeds 1; fast_image_resize and PIL diverge"
    )
    assert abs_diff.mean() < 0.5, f"Mean abs diff {abs_diff.mean()} too large"

    # Untouched fields survive the round-trip.
    np.testing.assert_array_equal(
        decoded["observation/joint_position"], observation["observation/joint_position"]
    )
    assert decoded["prompt"] == "test"


def test_image_preprocess_skips_when_already_at_target_shape(
    echo_binary_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """If the customer already resized client-side, the sidecar must no-op."""
    pre_resized = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    observation = {
        "observation/exterior_image_1_left": pre_resized,
        "prompt": "test",
    }
    frame_from_python = pack_local_frame(observation, schema_id="droid")
    image_specs = [
        {
            "path": ["observation/exterior_image_1_left"],
            "target_shape": [224, 224, 3],
            "dtype": "uint8",
        }
    ]
    echoed_bytes = _run_with_image_specs(echo_binary_path, frame_from_python, image_specs, tmp_path)
    decoded = unpack_local_frame(echoed_bytes)
    np.testing.assert_array_equal(decoded["observation/exterior_image_1_left"], pre_resized)
