"""Python-emit ↔ Rust-decode compatibility test for server metadata.

The Python server packs its metadata blob via ``msgpack_numpy.Packer()``
(see `openpi.serving.websocket_policy_server` and our local socket
server). openpi-flash-transport on the client decodes the same bytes via
``rmp_serde::from_slice`` at handshake to recover image_specs and
action_horizon. Those two libraries have subtly different conventions for
map key types, integer widths, and string encoding.

This test pipes real `msgpack_numpy.Packer()` output through the
``openpi-metadata-echo`` test binary and asserts the round-tripped
values match what Python emitted. It catches encoding divergences that
pure-Rust unit tests cannot — those only prove rmp_serde round-trips
itself.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest
from openpi_client import msgpack_numpy


def _hosting_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _candidate_metadata_echo_paths() -> list[pathlib.Path]:
    root = _hosting_repo_root()
    return [
        root / "flash-transport" / "target" / "debug" / "openpi-metadata-echo",
        root / "flash-transport" / "target" / "release" / "openpi-metadata-echo",
    ]


def _resolve_metadata_echo_binary() -> pathlib.Path | None:
    for candidate in _candidate_metadata_echo_paths():
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def metadata_echo_binary_path() -> pathlib.Path:
    resolved = _resolve_metadata_echo_binary()
    if resolved is None:
        searched = "\n".join(f"  - {p}" for p in _candidate_metadata_echo_paths())
        pytest.skip(
            "openpi-metadata-echo binary not built. Run `cargo build` in "
            f"flash-transport/ to enable these tests. Searched:\n{searched}"
        )
    return resolved


def _decode_with_rust(
    metadata_echo_binary_path: pathlib.Path, metadata_bytes: bytes
) -> dict[str, Any]:
    result = subprocess.run(
        [str(metadata_echo_binary_path)],
        input=metadata_bytes,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _pack_via_openpi_msgpack_numpy(metadata: dict[str, Any]) -> bytes:
    """Pack using the same Packer the hosting server uses at handshake.

    ``openpi_client.msgpack_numpy.Packer`` is a partial of ``msgpack.Packer``
    with a numpy-aware default. For plain dicts (no numpy arrays in the
    server metadata) it behaves exactly like stock msgpack.Packer.
    """
    return msgpack_numpy.Packer().pack(metadata)


def test_empty_metadata_decodes_as_empty(metadata_echo_binary_path: pathlib.Path) -> None:
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy({}))
    assert decoded == {"image_specs": [], "action_horizon": None}


def test_droid_image_specs_round_trip(metadata_echo_binary_path: pathlib.Path) -> None:
    metadata = {
        "image_specs": [
            {
                "path": ["observation/exterior_image_1_left"],
                "target_shape": [224, 224, 3],
                "dtype": "uint8",
            },
            {
                "path": ["observation/wrist_image_left"],
                "target_shape": [224, 224, 3],
                "dtype": "uint8",
            },
        ],
        "action_horizon": 10,
    }
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["action_horizon"] == 10
    assert len(decoded["image_specs"]) == 2
    assert decoded["image_specs"][0] == {
        "path": ["observation/exterior_image_1_left"],
        "target_shape": [224, 224, 3],
        "dtype": "uint8",
    }
    assert decoded["image_specs"][1] == {
        "path": ["observation/wrist_image_left"],
        "target_shape": [224, 224, 3],
        "dtype": "uint8",
    }


def test_aloha_nested_path_and_chw_round_trip(metadata_echo_binary_path: pathlib.Path) -> None:
    metadata = {
        "image_specs": [
            {
                "path": ["images", "cam_high"],
                "target_shape": [3, 224, 224],
                "dtype": "uint8",
            },
            {
                "path": ["images", "cam_low"],
                "target_shape": [3, 224, 224],
                "dtype": "uint8",
            },
            {
                "path": ["images", "cam_left_wrist"],
                "target_shape": [3, 224, 224],
                "dtype": "uint8",
            },
            {
                "path": ["images", "cam_right_wrist"],
                "target_shape": [3, 224, 224],
                "dtype": "uint8",
            },
        ],
        "action_horizon": 50,
    }
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["action_horizon"] == 50
    assert len(decoded["image_specs"]) == 4
    assert all(spec["target_shape"] == [3, 224, 224] for spec in decoded["image_specs"])
    assert [spec["path"] for spec in decoded["image_specs"]] == [
        ["images", "cam_high"],
        ["images", "cam_low"],
        ["images", "cam_left_wrist"],
        ["images", "cam_right_wrist"],
    ]


def test_unknown_top_level_fields_are_ignored(
    metadata_echo_binary_path: pathlib.Path,
) -> None:
    """Real server metadata carries extra fields we don't care about
    (model_name, version, etc.). Rust must tolerate them without failing."""
    metadata = {
        "model_name": "pi0.5-base",
        "version": "1.0.0",
        "creator": "openpi",
        "image_specs": [
            {
                "path": ["observation/exterior_image_1_left"],
                "target_shape": [224, 224, 3],
                "dtype": "uint8",
            },
        ],
        "action_horizon": 10,
    }
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["action_horizon"] == 10
    assert len(decoded["image_specs"]) == 1


def test_float32_dtype_round_trips(metadata_echo_binary_path: pathlib.Path) -> None:
    metadata = {
        "image_specs": [
            {
                "path": ["obs"],
                "target_shape": [224, 224, 3],
                "dtype": "float32",
            }
        ],
    }
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["image_specs"][0]["dtype"] == "float32"


def test_missing_action_horizon_decodes_as_none(
    metadata_echo_binary_path: pathlib.Path,
) -> None:
    metadata = {
        "image_specs": [
            {"path": ["obs"], "target_shape": [224, 224, 3], "dtype": "uint8"},
        ],
    }
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["action_horizon"] is None


def test_missing_image_specs_decodes_as_empty(
    metadata_echo_binary_path: pathlib.Path,
) -> None:
    metadata = {"action_horizon": 10}
    decoded = _decode_with_rust(metadata_echo_binary_path, _pack_via_openpi_msgpack_numpy(metadata))
    assert decoded["image_specs"] == []
    assert decoded["action_horizon"] == 10


def test_large_action_horizon_fits(metadata_echo_binary_path: pathlib.Path) -> None:
    """Confirm we handle chunk sizes larger than a u8/u16 boundary (the
    kind msgpack would switch encoding on). 50 fits in u8, 1000 in u16,
    100_000 in u32 — all should round-trip."""
    for horizon in (50, 1000, 100_000):
        decoded = _decode_with_rust(
            metadata_echo_binary_path,
            _pack_via_openpi_msgpack_numpy({"action_horizon": horizon}),
        )
        assert decoded["action_horizon"] == horizon
