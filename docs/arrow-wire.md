# QUIC wire format — Arrow IPC + local transport framing

This doc specifies two protocols used by the direct-QUIC path
(`FlashTransportPolicy` ↔ `openpi-flash-transport` ↔ `LocalPolicySocketServer`):

1. **Local framing** — a thin binary format spoken between Python and the Rust transport layer over the Unix socket.
2. **QUIC wire format** — Apache Arrow IPC Streaming Format, produced and consumed exclusively by the Rust transport layer.

The QUIC path uses Arrow IPC unconditionally. There is no negotiation and no msgpack_numpy fallback on this path — the Rust transport layer owns the codec translation end to end. Customers who want a pure-Python transport should use the WebSocket path (`openpi_client.WebsocketClientPolicy` ↔ openpi's `WebsocketPolicyServer`), which continues to speak msgpack_numpy as it always has. That path is untouched by this design.

## Envelope

Every message on the Unix socket is:

```
[length: u32 big-endian][type: u8][body: (length - 1) bytes]
```

Where `type` is one of:

- Requests: `0x01` METADATA, `0x02` INFER, `0x03` RESET
- Responses: `0x11` METADATA, `0x12` INFER, `0x13` ERROR, `0x14` RESET

## Metadata handshake

When the client sidecar first connects, it issues a `METADATA` (`0x01`) request. The server replies (`0x11`) with openpi's standard metadata dict, msgpack-packed via `msgpack_numpy.Packer()`. The sidecar forwards these bytes verbatim; metadata is a one-time handshake payload, not an INFER payload, and staying on msgpack here avoids fork-type translation for openpi's own format.

The Rust client transport parses two optional fields out of the metadata blob (via `rmp_serde` in `flash-transport/src/metadata.rs`):

- `image_specs` — per-field image preprocessing rules (see below)
- `action_horizon` — the leading dim of chunked arrays, used when chunking is opted in

Unknown fields are ignored so old servers and new clients stay compatible in both directions.

## Local framing — `INFER` body

The body of an `INFER` request (`0x02`) or `INFER` response (`0x12`) is a sequence of three sections:

```
[SCHEMA_ID_SECTION][ARRAY_SECTION][SCALAR_SECTION]
```

### Schema ID section

```
[schema_id_len: u8][schema_id: utf-8 bytes]
```

`schema_id` is a free-form hint about the robot type (e.g. `"aloha"`, `"droid"`, `"unknown"`). The sidecar doesn't use it for dispatch today; it's reserved for future schema-specific optimizations.

### Array section

```
[num_arrays: u16 big-endian]
for each array:
  [path_depth: u8]
  for each path component:
    [component_len: u8][component: utf-8 bytes]
  [dtype_code: u8]
  [ndim: u8]
  [shape: u32 big-endian × ndim]
  [data_len: u64 big-endian]
  [data: data_len bytes]
```

`path_depth` supports nested dicts. For a flat key like DROID's `observation/exterior_image_1_left`, `path_depth = 1` and the single component is the full string (the `/` is part of the key, not a separator). For ALOHA's nested `images.cam_high`, `path_depth = 2` and the two components are `"images"` and `"cam_high"`.

`dtype_code` values:

| Code | numpy dtype |
|---|---|
| `0x01` | uint8 |
| `0x02` | int8 |
| `0x03` | uint16 |
| `0x04` | int16 |
| `0x05` | uint32 |
| `0x06` | int32 |
| `0x07` | uint64 |
| `0x08` | int64 |
| `0x09` | float16 |
| `0x0A` | float32 |
| `0x0B` | float64 |
| `0x0C` | bool |

Data alignment is **not** guaranteed in the local frame — the sidecar copies into an aligned Arrow buffer when building the RecordBatch. Python can therefore write `ndarray.tobytes()` directly without padding. (Alignment is enforced on the QUIC wire by Arrow IPC itself.)

### Scalar section

```
[scalar_json_len: u32 big-endian]
[scalar_json: utf-8 JSON bytes]
```

Anything that isn't a numpy array goes here as plain JSON. In practice this is `{"prompt": "..."}` on observations and `{"server_timing": {...}, "policy_timing": {...}}` on responses. Paths follow the same nesting convention as the array section (dotted JSON paths), but flat dicts cover every openpi case today.

## QUIC wire format — Arrow IPC Streaming

On the QUIC wire, each logical inference request/response is one Arrow IPC Streaming Format message pair:

1. A Schema message describing the RecordBatch columns.
2. A single RecordBatch containing one row (one observation or one action chunk).
3. An end-of-stream marker.

The schema is sent with every message to keep each request self-contained on a QUIC stream. This adds a small fixed overhead (~100 bytes) but avoids any schema-caching state on the sidecar.

The sidecar implements a simple one-column-per-ndarray mapping: each `LocalArray` becomes an Arrow `Binary` column whose value is the raw tensor bytes; dtype, shape, and path live in Arrow field metadata. See `flash-transport/src/arrow_codec.rs` for the exact mapping. Only our sidecar reads this wire — no interop with other Arrow consumers, so the format stays minimal.

## Implementation split

| Layer | Owner | Notes |
|---|---|---|
| Local framing encode (Python → socket) | Python | Thin writer in `src/hosting/local_frame.py` using `struct.pack` + `ndarray.tobytes()`. |
| Local framing decode (socket → Python) | Python | Thin reader using `np.frombuffer` for zero-copy numpy views. |
| Local framing ↔ Arrow RecordBatch | Rust | `flash-transport/src/local_format.rs` (parser) + `flash-transport/src/arrow_codec.rs` (encoder/decoder). |
| QUIC wire (Arrow IPC Streaming) | Rust | `arrow-ipc` crate. |

The Python side never touches Arrow. `openpi-flash-transport` owns the translation in both directions.

## Server metadata extensions used by openpi-flash-transport

### `image_specs`

A list of per-field rules telling the transport to resize and dtype-convert images before they hit the QUIC wire. Customers can stop calling `openpi_client.image_tools.resize_with_pad` — the transport handles it transparently.

```python
metadata["image_specs"] = [
    {
        "path": ["observation/exterior_image_1_left"],  # nested dict path
        "target_shape": [224, 224, 3],                  # HWC or CHW (auto-detected)
        "dtype": "uint8",                               # uint8 or float32
    },
    ...
]
```

The Rust side (`flash-transport/src/image_preprocess.rs`) uses `fast_image_resize`'s bilinear filter for pixel-parity with PIL (`tf.image.resize_with_pad` semantics) and `fast_image_resize` + `ndarray` for dtype conversion and CHW/HWC layout handling.

### `action_horizon`

Advertises the leading dim the transport can chunk on. When the customer sets `OPENPI_OPEN_LOOP_HORIZON=N`, the transport caches each server response and serves one step per Python `infer()` call — the same behavior `openpi_client.action_chunk_broker.ActionChunkBroker` provides on the WebSocket path, now automatic on QUIC.

```python
metadata["action_horizon"] = 15  # DROID; ALOHA is 50
```

Chunking is opt-in:
- If `OPENPI_OPEN_LOOP_HORIZON` is unset, the transport passes server responses through whole (one Python `infer()` call = one server round-trip).
- If set to `N` and the server advertises `action_horizon=H`, the transport serves up to `min(N, H)` steps before forwarding the next inference request.
- `policy.reset()` clears the cache.

Only fields whose leading dim equals `action_horizon` are sliced; other arrays in the response pass through unchanged each step.

## `server_timing` injection

The **server-side** openpi-flash-transport process measures Python-backend wall time (request received over QUIC → response received from Python) and injects `server_timing.{infer_ms, prev_total_ms}` into the response's scalar JSON before re-encoding Arrow IPC. The Python backend never sets `server_timing` on the QUIC path.

Reasoning: openpi-flash-transport has the most accurate view of "server wall time" (it sees the Python work, the local socket round-trip, and Python serialization), all without burning Python GIL cycles.

`policy_timing.infer_ms` (the model's self-reported forward-pass time, set inside `openpi.policies.Policy.infer()`) is left untouched — it's the model's number, and openpi-flash-transport preserves it through the JSON merge.
