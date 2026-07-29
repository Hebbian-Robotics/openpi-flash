# openpi-flash-transport

The transport layer for `openpi-flash`: a self-contained Rust process that
sits between an openpi inference server and its Python caller, owning the
cross-cutting concerns that used to live on the hot path — QUIC transport,
Arrow IPC codec, image preprocessing, action chunk caching, and server-side
timing instrumentation.

The crate ships one binary (`openpi-flash-transport`) that runs in two modes:

1. `server` (deployed alongside the inference backend)
2. `client` (spawned by the caller).

Both sides speak a small binary frame format over a local Unix
socket to their respective Python shims, and Arrow IPC over QUIC to each
other.

## What it owns

- **QUIC transport** (`quinn`, rustls) — single long-lived bidirectional stream,
  self-signed certificate, tunable congestion window and idle timeout.
- **Arrow IPC codec** (`arrow-ipc`) — the wire format between the two endpoints.
- **Local binary frame format** — `LocalFrame`, the codec used over the Unix
  socket to the Python shim on either side. Flat, zero-copy-friendly layout
  with a small `scalar_json` side-channel.
- **Image preprocessing** (`fast_image_resize`) — `resize_with_pad` with
  aspect-preserving zero padding, PIL-BILINEAR parity, HWC↔CHW transpose.
  Target shapes are learned from the server's handshake metadata.
- **Action chunk caching** — serves one step per `infer()` call from a cached
  chunk, re-fetches every `OPENPI_OPEN_LOOP_HORIZON` steps. Arrow path only.
- **Server timing injection** — the server stamps `server_timing.infer_ms`
  and `prev_total_ms` into the response's `scalar_json` before sending.

## Architecture

```
Python caller                                    Python inference backend
   │                                                          ▲
   │ LocalFrame over Unix socket                              │ LocalFrame over Unix socket
   ▼                                                          │
┌──────────────────────┐    Arrow IPC over QUIC    ┌──────────┴───────────┐
│ transport (client)   │◄─────────────────────────►│ transport (server)   │
│  - resize_with_pad   │                           │  - server_timing     │
│  - chunk cache       │                           │    injection         │
│  - handshake         │                           │                      │
└──────────────────────┘                           └──────────────────────┘
```

Each boundary uses the smallest codec that fits:

- `msgpack_numpy` stays at Python-facing surfaces because that is OpenPI's
  native WebSocket and metadata format. Rust only decodes the metadata blob.
- `LocalFrame` stays on the Unix socket because it is private to the Rust
  binary and Python shim. The Python side writes `ndarray.tobytes()` plus a few
  `struct.pack`s; Rust reads the same layout without adding `pyarrow` to callers.
- Arrow IPC stays on the QUIC wire because it is a maintained cross-language
  standard with aligned buffers for tensor-heavy traffic.

`LocalFrame` is intentionally not schema'd. It is mirrored in `local_frame.py`
and `local_format.rs`, and the cross-language tests keep those copies in
lockstep. Arrow on the Unix socket would pull `pyarrow` into every caller;
`msgpack_numpy` on QUIC would make Rust reimplement a Python convention and
still copy arrays.

## Quickstart

Build:

```bash
cargo build --release
```

Server side (on the inference host, typically inside the Docker container
alongside the Python backend):

```bash
openpi-flash-transport server \
    --listen-port 5555 \
    --backend-socket-path /tmp/openpi-action.sock
```

The hosted service uses `/tmp/openpi-action.sock` for the action slot and `/tmp/openpi-planner.sock` for the planner slot by default.

Client side (on the caller's machine):

```bash
openpi-flash-transport client \
    --server-host <host> \
    --server-port 5555 \
    --local-port 5556 \
    --local-socket-path /tmp/openpi-client.sock
```

Enable chunking on the client (optional):

```bash
OPENPI_OPEN_LOOP_HORIZON=8 openpi-flash-transport client ...
```

## CLI reference

### `server`

| Flag | Default | Purpose |
|---|---|---|
| `--listen-port` | `5555` | UDP port to accept QUIC clients on. |
| `--backend-socket-path` | *required* | Unix socket of the Python inference backend. |
| `--max-idle-timeout-secs` | `10` | QUIC idle timeout. |
| `--keep-alive-interval-secs` | `2` | QUIC keep-alive. |
| `--initial-window-bytes` | `1 MiB` | Initial congestion window. |

### `client`

| Flag | Default | Purpose |
|---|---|---|
| `--server-host` | *required* | Remote QUIC server hostname/IP. |
| `--server-port` | `5555` | Remote QUIC server UDP port. |
| `--local-port` | `5556` | Local UDP port for the QUIC client socket. |
| `--local-socket-path` | *required* | Unix socket exposed to the Python caller. |
| `--max-idle-timeout-secs` | `10` | QUIC idle timeout. |
| `--keep-alive-interval-secs` | `2` | QUIC keep-alive. |
| `--initial-window-bytes` | `1 MiB` | Initial congestion window. |

### Environment variables

| Variable | Effect |
|---|---|
| `OPENPI_OPEN_LOOP_HORIZON` | Enables client-side action chunking. Steps served per server hit, capped at the server-advertised `action_horizon`. |
| `OPENPI_FLASH_TRANSPORT_BINARY` | Override the path the Python shim uses to locate the binary. Defaults to `/usr/local/bin/openpi-flash-transport`, then the cargo `target/` build output. |
| `RUST_LOG` | Standard `tracing-subscriber` filter. Defaults to `info`. |

## Library layout

The crate also exposes a library (`openpi_flash_transport`) so other Rust
code can link against the exact same codec used on the wire:

| Module | Purpose |
|---|---|
| `local_format` | `LocalFrame` / `LocalArray` types and codec for the Unix-socket framing. |
| `arrow_codec` | Arrow IPC encode/decode for the QUIC wire format. |
| `image_preprocess` | `resize_with_pad`, HWC↔CHW transpose, dtype conversion. |
| `chunk_cache` | Action-chunk state machine. |
| `metadata` | Server handshake metadata decoder (msgpack). |

Small helper binaries under `src/bin/` (`arrow_wire_echo`, `metadata_echo`)
are used for cross-language round-trip tests and aren't part of the public
surface.

## Testing

```bash
cargo test
cargo clippy --all-targets --all-features
cargo fmt --check
```

Cross-language round-trip tests live in `tests/test_arrow_wire.py` and `tests/test_metadata_python_emit.py`, and exercise the Python shim against the real binary.

## Non-goals

- **PyO3 / in-process Rust.** The separate-process model is deliberate. It
  keeps the Python GIL away from the hot path, lets either side restart
  independently, and keeps the slow inference server warm while experimenting
  with transport, preprocessing, or inference settings.
- **Policy inference.** The transport layer handles networking and
  preprocessing only; the actual model runs in Python (PyTorch or JAX,
  depending on the configured slot) behind the backend Unix socket.
- **msgpack_numpy compatibility.** The preprocessing and chunking paths
  require Arrow IPC on the wire. The msgpack_numpy fallback still works but
  does not benefit from transport-side resize or chunking.
