# Architecture

This document describes the high-level structure of openpi-flash. It's meant to help you find your way around the codebase. If you want to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Overview

openpi-flash wraps [openpi](https://github.com/Physical-Intelligence/openpi)'s model inference in a network server that speaks QUIC (recommended) and WebSocket. No openpi code is modified — we call `create_trained_policy()` and `Policy.infer()` directly, and for subtask generation we call the pi0.5 JAX model's PaliGemma backbone in an autoregressive decode loop. The main engineering problems are transport latency, torch.compile warmup, and multi-region deployment.

The server has two optional component slots — an **action** slot (PyTorch or JAX policy) and a **planner** slot (JAX subtask generator for pi0.5 two-phase inference). Which slots are configured picks one of three modes at startup (`_resolve_mode` in `serve.py`): `action_only`, `planner_only`, or `combined`. In combined mode the action endpoint transparently consults the planner to augment prompts, and the planner endpoint is independently callable.

There are three deployment paths: local development, Docker on EC2, and Modal (serverless). Each path loads the same slots and serves the same wire protocol; what differs is how the QUIC transport is handled and how the model is provisioned.

## Codemap

### Entry points

- `main.py` — CLI (Typer). `serve` starts the local server, `test *` runs smoke tests against a running instance.
- `modal_app.py` — Modal ASGI deployment (WebSocket only, scales to zero).
- `modal_tunnel_app.py` — Modal with direct TCP tunnel, bypassing ASGI for lower latency.
- `modal_quic_app.py` — Modal with QUIC via NAT traversal (experimental, unreliable).
- `convert_checkpoint_modal.py` — Batch job to convert JAX checkpoints to PyTorch on Modal GPUs.
- `Dockerfile` — Multi-stage build: Rust transport binary compilation, then Python runtime with CUDA.

### src/hosting/ — core library

Server core:

- `serve.py` — Main server. Derives the mode from the loaded `ServiceConfig`, loads each active slot, assembles an `EndpointSpec` per slot (policy + transport triple + metadata), then starts one WebSocket thread, one unix-socket thread, and one `openpi-flash-transport` subprocess per slot. All policies are wrapped in `ThreadSafePolicy` for concurrent access across transports. This is the file to read first.
- `config.py` — Pydantic config model, loaded from JSON via `INFERENCE_CONFIG_PATH` (or `--config`). Defines `ActionConfig`, `PlannerConfig`, and `SlotTransportConfig`; at least one of `action` / `planner` must be set. `OPENPI_*` env vars override JSON values (nested fields use `__` as delimiter).
- `compile_mode.py` — Resolves `OPENPI_PYTORCH_COMPILE_MODE` env var to a torch.compile mode string or `None` (eager).
- `warmup.py` — Generates dummy ALOHA observations for the torch.compile warmup pass.
- `prepare_checkpoint.py` — Implements `main.py prepare-checkpoint`. Downloads PyTorch weights from `lerobot/pi05_base` and normalization stats from `gs://openpi-assets/checkpoints/pi05_base/assets`, assembles a local checkpoint directory the runtime can load directly.

Planner (pi0.5 two-phase inference):

- `subtask_generator.py` — JAX-only subtask decoder. JIT-compiles the PaliGemma prefix + autoregressive decode loop into a single XLA graph (the Python `for` loop is unrolled by the tracer so each iteration has concrete shapes). Restricts generation to printable ASCII tokens via a vocab mask. Holds an internal lock so shared use across endpoints in combined mode is serialized. Warm ≈ 1.1s per generation.
- `subtask_policy.py` — Two `BasePolicy` wrappers: `SubtaskAugmentedPolicy` (combined mode; runs the planner first, splices the subtask into the prompt via `action_prompt_template`, then calls the inner action policy; honors `obs["mode"]` ∈ `default` / `subtask_only` / `action_only`), and `PlannerPolicy` (planner-only endpoint; returns `{"subtask": {"text", "ms"}}` with no `actions`).
- `admin_server.py` — FastAPI admin HTTP endpoint on port 8001, started only when the planner slot is loaded. `GET /config` / `PATCH /config` let operators mutate `generation_prompt_format` at runtime without a restart; `SubtaskGenerator.generate()` re-reads the value on every call. Bound to `0.0.0.0` inside the container, but docker/terraform publishes it on `127.0.0.1` only.

Transports:

- `local_policy_socket_server.py` — Unix socket server that openpi-flash-transport connects to. Receives framed requests, calls `policy.infer()`, sends responses. This is the boundary between Rust (network) and Python (inference).
- `local_transport_protocol.py` — Shared message-type enums (`TransportRequestType`, `TransportResponseType`) and framing helpers for the Unix-socket protocol. Used by both server and client sides of the local transport.
- `local_frame.py` — Python codec for the `LocalFrame` binary format over the Unix socket. Mirrors `flash-transport/src/local_format.rs`; the Rust transport translates these frames to/from Arrow IPC on the QUIC wire. Avoids any serialization framework so encode/decode is ~`ndarray.tobytes()` + `struct.pack` and decode uses `np.frombuffer` (zero-copy view).
- `flash_transport_binary.py` — Locates the `openpi-flash-transport` binary (env override → `/usr/local/bin/...` → local cargo build output) and holds typed `ServerArgs` / `ClientArgs` dataclasses that mirror the Rust clap structs, so a Rust flag rename becomes a Python type error.
- `flash_transport_policy.py` — Python `BasePolicy` wrapper for direct QUIC connections to EC2/Docker. Spawns a local `openpi-flash-transport` client subprocess so robot code keeps the normal `infer()` interface.
- `quic_protocol.py` — Shared QUIC wire format: message types, framing, handshake helpers, and the `serve_quic_connection` loop used by the NAT-traversal server.
- `quic_server.py` — QUIC server using quic-portal for Modal deployments (NAT traversal via STUN).
- `quic_client_policy.py` — Client counterpart to `quic_server.py`, discovers server via Modal Dict.
- `relay.py` — UDP relay registration for NAT traversal fallback (WIP).

Modal glue:

- `modal_asgi.py` — Adapts openpi's `WebsocketPolicyServer` into a Starlette ASGI app for Modal.
- `modal_helpers.py` — Shared Modal utilities: Docker image builder, model loading, transformers patching.
- `modal_dict_names.py` — Shared Modal Dict names (`openpi-quic-info`, `openpi-tunnel-info`) used by both server and client entry points so the two sides can't drift out of sync.

Benchmarking:

- `benchmark.py` — Timing harness used by all test scripts.

### flash-transport/ — Rust binary

Single-file Rust binary (`src/main.rs`) with two modes:

- `server` — Listens for QUIC connections on a configurable UDP port (default 5555 for the action slot; 5556 for the planner slot when combined mode is active) and connects to the Python backend via Unix socket.
- `client` — Connects to a remote direct QUIC server and exposes the same local Unix-socket protocol to the Python client wrapper.

The local transport protocol uses length-prefixed messages with type bytes: `0x01`/`0x11` for metadata, `0x02`/`0x12` for inference, `0x03`/`0x14` for reset, and `0x13` for errors. This exists because keeping QUIC in Rust gives a cleaner low-latency path on both sides while preserving the standard Python policy interface.

Each active slot gets its own supervised `flash-transport` subprocess (one unix socket, one UDP port, one restart loop). Supervision uses exponential backoff via tenacity so a crash-looping binary doesn't thrash the CPU.

### tests/

Smoke tests invoked via `main.py test *`. Each test calls `wait_for_server()` (HTTP health check), connects via the appropriate transport, runs `benchmark.run_benchmark()`, and prints timing results. `helpers.py` has the shared `random_observation_aloha()` generator.

### infra/ — Terraform

Two-tier structure:

- `infra/` (root) — Shared, one-time resources: ECR repository, IAM roles, GitHub Actions OIDC. Apply once.
- `infra/regional-instance/` — Per-region EC2 deployment. Uses Terraform workspaces (`openpi-uswest2`, `openpi-malaysia`, `openpi-seoul`). Calls the reusable module at `infra/modules/regional_inference_instance/`.
- `infra/modules/regional_inference_instance/` — The reusable module: EC2 instance, security group, optional Elastic IP, cloud-init bootstrap via `user_data.yaml.tftpl`.

## Architectural invariants

- **No openpi modifications.** We import from openpi and openpi-client as-is. If something needs to change in openpi, it should be upstreamed.
- **One policy per slot, shared across its transports.** Each active slot (action / planner) loads its policy exactly once and wraps it in `ThreadSafePolicy`; WebSocket, QUIC, and the local unix-socket server for that slot all call the same `infer()` behind a lock. Adding a transport to an existing slot means adding a thread, not duplicating model loading.
- **Slots are independent on the wire, coupled in memory.** Each active slot gets its own websocket port, QUIC port, unix socket, and `flash-transport` supervisor — the two endpoints are independent at the transport layer. In combined mode they share the same `SubtaskGenerator` instance in memory (the planner's internal lock serializes JAX calls) so the action endpoint's prompt augmentation and the planner endpoint return identical subtasks for the same input.
- **Mode is derived, not declared.** `_resolve_mode` in `serve.py` picks `action_only` / `planner_only` / `combined` from which slots are set in `ServiceConfig`. There is no separate mode flag; the pydantic model-validator rejects configs where both slots are empty.
- **Rust owns the network, Python owns inference.** In Docker/EC2, openpi-flash-transport terminates QUIC and speaks a simple framed protocol over a Unix socket to Python. The same local protocol is also used by the client wrapper, which spawns a local openpi-flash-transport process and keeps the Python `BasePolicy` API intact.
- **Infrastructure is separated by blast radius.** Shared resources (ECR, IAM) are in the Terraform root. Regional EC2 instances are in separate workspaces so you can destroy one region without affecting others.

## Cross-cutting concerns

- **torch.compile modes** are threaded through from env var (`OPENPI_PYTORCH_COMPILE_MODE`) to Terraform user data to Docker env to `compile_mode.py` to the openpi config dataclass. The action-slot warmup pass in `serve.py` triggers compilation before the server starts accepting connections.
- **JAX planner warmup.** When the planner slot is loaded, `SubtaskGenerator.warmup()` runs before endpoints come up so the first client call doesn't block on JIT compilation. Warm path is ~1.1s per generation with the default JIT-unrolled decode; in combined mode this adds to every action `infer()` unless the client opts out with `obs["mode"] = "action_only"`.
- **Runtime configuration surface.** The admin HTTP endpoint (port 8001, only started with the planner slot) is the single supported way to mutate server state without a restart. Today it exposes `generation_prompt_format`; any future runtime-tunable knob should go through the same `RuntimeConfig` / `RuntimeConfigUpdate` types so invariants are shared with `PlannerConfig`.
- **Health checks** — Each slot's WebSocket server exposes `/healthz` on its own TCP port. All test scripts and the Terraform cloud-init poll this before attempting inference. The admin endpoint additionally exposes `GET /health` on port 8001.
- **Checkpoint download** — openpi's `maybe_download()` handles `gs://` and local paths. The hosting layer also prepares Hugging Face checkpoints with `huggingface-hub` for Docker/EC2 action-slot deployments.
