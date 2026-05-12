"""Python codec for the LocalFrame binary format used over the local Unix socket.

See ``docs/arrow-wire.md`` for the wire format. Provides the thin writer
and reader used on the Python side; the Rust transport translates these
frames to/from Arrow IPC Streaming Format for the QUIC wire. Mirrors
``flash-transport/src/local_format.rs``.

The format intentionally avoids any serialization framework so encoding is
roughly ``ndarray.tobytes()`` + a handful of ``struct.pack`` calls. On the
decode side, ``np.frombuffer`` is used so tensor data is a view over the
received bytes rather than a fresh copy.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from typing import Any, Final

import numpy as np

# Mirrors the Rust `DtypeCode` enum in `flash-transport/src/local_format.rs`.
DTYPE_CODE_UINT8: Final[int] = 0x01
DTYPE_CODE_INT8: Final[int] = 0x02
DTYPE_CODE_UINT16: Final[int] = 0x03
DTYPE_CODE_INT16: Final[int] = 0x04
DTYPE_CODE_UINT32: Final[int] = 0x05
DTYPE_CODE_INT32: Final[int] = 0x06
DTYPE_CODE_UINT64: Final[int] = 0x07
DTYPE_CODE_INT64: Final[int] = 0x08
DTYPE_CODE_FLOAT16: Final[int] = 0x09
DTYPE_CODE_FLOAT32: Final[int] = 0x0A
DTYPE_CODE_FLOAT64: Final[int] = 0x0B
DTYPE_CODE_BOOL: Final[int] = 0x0C

_DTYPE_TO_CODE: Final[dict[np.dtype, int]] = {
    np.dtype(np.uint8): DTYPE_CODE_UINT8,
    np.dtype(np.int8): DTYPE_CODE_INT8,
    np.dtype(np.uint16): DTYPE_CODE_UINT16,
    np.dtype(np.int16): DTYPE_CODE_INT16,
    np.dtype(np.uint32): DTYPE_CODE_UINT32,
    np.dtype(np.int32): DTYPE_CODE_INT32,
    np.dtype(np.uint64): DTYPE_CODE_UINT64,
    np.dtype(np.int64): DTYPE_CODE_INT64,
    np.dtype(np.float16): DTYPE_CODE_FLOAT16,
    np.dtype(np.float32): DTYPE_CODE_FLOAT32,
    np.dtype(np.float64): DTYPE_CODE_FLOAT64,
    np.dtype(np.bool_): DTYPE_CODE_BOOL,
}

_CODE_TO_DTYPE: Final[dict[int, np.dtype]] = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def pack_local_frame(payload: dict[str, Any], *, schema_id: str = "unknown") -> bytes:
    """Serialize ``payload`` into the local frame binary format.

    Nested dicts are supported. Numpy arrays become array entries keyed by
    their dict path; any other value goes into the scalar JSON trailer.
    """
    arrays: list[tuple[list[str], np.ndarray]] = []
    scalars: dict[str, Any] = {}

    for path, value in _walk_payload(payload):
        if isinstance(value, np.ndarray):
            arrays.append((path, value))
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            _insert_scalar(scalars, path, value.item())
        else:
            _insert_scalar(scalars, path, value)

    schema_id_bytes = schema_id.encode("utf-8")
    if len(schema_id_bytes) > 255:
        raise ValueError(f"schema_id too long: {len(schema_id_bytes)} bytes (max 255)")
    if len(arrays) > 0xFFFF:
        raise ValueError(f"too many arrays: {len(arrays)} (max 65535)")

    parts: list[bytes] = [bytes([len(schema_id_bytes)]), schema_id_bytes]
    parts.append(struct.pack(">H", len(arrays)))

    for path, array in arrays:
        parts.append(_encode_array_entry(path, array))

    scalar_json = json.dumps(scalars, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(scalar_json) > 0xFFFFFFFF:
        raise ValueError(f"scalar_json too long: {len(scalar_json)} bytes")
    parts.append(struct.pack(">I", len(scalar_json)))
    parts.append(scalar_json)

    return b"".join(parts)


def unpack_local_frame(frame: bytes) -> dict[str, Any]:
    """Deserialize a local frame into the nested dict the caller sent.

    Numpy arrays are reconstructed via ``np.frombuffer``, so the returned
    arrays are views over ``frame``. Callers must either consume the arrays
    before ``frame`` is freed or copy them with ``.copy()``.
    """
    frame_memoryview = memoryview(frame)
    offset = 0

    schema_id_len = frame_memoryview[offset]
    offset += 1
    _ = frame_memoryview[offset : offset + schema_id_len].tobytes().decode("utf-8")
    offset += schema_id_len

    (num_arrays,) = struct.unpack_from(">H", frame_memoryview, offset)
    offset += 2

    result: dict[str, Any] = {}
    for _ in range(num_arrays):
        path_depth = frame_memoryview[offset]
        offset += 1
        if path_depth == 0:
            raise ValueError("Array path_depth must be >= 1")
        path = []
        for _ in range(path_depth):
            component_len = frame_memoryview[offset]
            offset += 1
            component = frame_memoryview[offset : offset + component_len].tobytes().decode("utf-8")
            offset += component_len
            path.append(component)

        dtype_code = frame_memoryview[offset]
        offset += 1
        if dtype_code not in _CODE_TO_DTYPE:
            raise ValueError(f"Unknown dtype code: 0x{dtype_code:02x}")
        dtype = _CODE_TO_DTYPE[dtype_code]

        ndim = frame_memoryview[offset]
        offset += 1
        shape = struct.unpack_from(f">{ndim}I", frame_memoryview, offset)
        offset += 4 * ndim

        (data_len,) = struct.unpack_from(">Q", frame_memoryview, offset)
        offset += 8
        array = np.frombuffer(frame, dtype=dtype, count=int(np.prod(shape)), offset=offset).reshape(
            shape
        )
        offset += data_len

        _insert_array(result, path, array)

    (scalar_json_len,) = struct.unpack_from(">I", frame_memoryview, offset)
    offset += 4
    scalar_json = frame_memoryview[offset : offset + scalar_json_len].tobytes()
    offset += scalar_json_len
    if offset != len(frame):
        raise ValueError(f"Trailing {len(frame) - offset} bytes in local frame")

    scalars = json.loads(scalar_json.decode("utf-8")) if scalar_json else {}
    _merge_scalars(result, scalars)
    return result


def _encode_array_entry(path: list[str], array: np.ndarray) -> bytes:
    if len(path) == 0 or len(path) > 255:
        raise ValueError(f"path must have 1..255 components, got {len(path)}")
    if array.dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported numpy dtype: {array.dtype}")
    if array.ndim > 255:
        raise ValueError(f"array ndim too large: {array.ndim}")
    for dim in array.shape:
        if dim < 0 or dim > 0xFFFFFFFF:
            raise ValueError(f"array shape dimension out of u32 range: {dim}")

    array_contiguous = np.ascontiguousarray(array)
    data_bytes = array_contiguous.tobytes()

    parts: list[bytes] = [bytes([len(path)])]
    for component in path:
        component_bytes = component.encode("utf-8")
        if len(component_bytes) > 255:
            raise ValueError(f"path component too long: {component!r}")
        parts.append(bytes([len(component_bytes)]))
        parts.append(component_bytes)

    parts.append(bytes([_DTYPE_TO_CODE[array_contiguous.dtype]]))
    parts.append(bytes([array_contiguous.ndim]))
    parts.append(struct.pack(f">{array_contiguous.ndim}I", *array_contiguous.shape))
    parts.append(struct.pack(">Q", len(data_bytes)))
    parts.append(data_bytes)

    return b"".join(parts)


def _walk_payload(
    payload: dict[str, Any], path: list[str] | None = None
) -> Iterator[tuple[list[str], Any]]:
    path = path or []
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"Observation dict keys must be str, got {type(key).__name__}")
        new_path = [*path, key]
        if isinstance(value, dict):
            yield from _walk_payload(value, new_path)
        else:
            yield new_path, value


def _insert_scalar(scalars: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = scalars
    for component in path[:-1]:
        cursor = cursor.setdefault(component, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Scalar path {path} collides with an existing non-dict value")
    cursor[path[-1]] = value


def _insert_array(result: dict[str, Any], path: list[str], array: np.ndarray) -> None:
    cursor = result
    for component in path[:-1]:
        cursor = cursor.setdefault(component, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Array path {path} collides with an existing non-dict value")
    cursor[path[-1]] = array


def _merge_scalars(result: dict[str, Any], scalars: dict[str, Any]) -> None:
    for key, value in scalars.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            _merge_scalars(result[key], value)
        elif key in result and not isinstance(value, dict):
            raise ValueError(f"Scalar key {key!r} collides with an existing array")
        else:
            result[key] = value
