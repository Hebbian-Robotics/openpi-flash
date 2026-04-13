# Architecture

This document describes the high-level structure of openpi-flash. It's meant to help you find your way around the codebase. If you want to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Overview

openpi-flash wraps [openpi](https://github.com/Physical-Intelligence/openpi)'s policy inference in a network server that speaks QUIC (recommended) and WebSocket. No openpi code is modified — we call `create_trained_policy()` and `Policy.infer()` directly. The main engineering problems are transport latency, torch.compile warmup, and multi-region deployment.

There are three deployment paths: local development, Docker on EC2, and Modal (serverless). Each path loads the same policy and serves the same wire protocol; what differs is how the QUIC transport is handled and how the model is provisioned.

## Codemap

### Entry points

- `main.py` — CLI (Typer). `serve` starts the local server, `test *` runs smoke tests against a running instance.
- `modal_app.py` — Modal ASGI deployment (WebSocket only, scales to zero).
- `modal_tunnel_app.py` — Modal with direct TCP tunnel, bypassing ASGI for lower latency.
- `modal_quic_app.py` — Modal with QUIC via NAT traversal (experimental, unreliable).
- `convert_checkpoint_modal.py` — Batch job to convert JAX checkpoints to PyTorch on Modal GPUs.
- `Dockerfile` — Multi-stage build: Rust sidecar compilation, then Python runtime with CUDA.

### src/hosting/ — core library

- `serve.py` — Main server. Loads the policy, starts WebSocket thread, starts the Rust QUIC sidecar, wires them together via `ThreadSafePolicy`. This is the file to read first.
- `config.py` — Pydantic config model, loaded from JSON via `INFERENCE_CONFIG_PATH` env var.
- `compile_mode.py` — Resolves `OPENPI_PYTORCH_COMPILE_MODE` env var to a torch.compile mode string or `None` (eager).
- `warmup.py` — Generates dummy ALOHA observations for the torch.compile warmup pass.
- `local_policy_socket_server.py` — Unix socket server that the Rust sidecar connects to. Receives framed requests, calls `policy.infer()`, sends responses. This is the boundary between Rust (network) and Python (inference).
- `direct_quic_client_policy.py` — Python `BasePolicy` wrapper for direct QUIC connections to EC2/Docker. Spawns a local Rust sidecar client process so robot code keeps the normal `infer()` interface.
- `quic_protocol.py` — Shared QUIC wire format: message types, framing, handshake helpers, and the `serve_quic_connection` loop used by the NAT-traversal server.
- `quic_server.py` — QUIC server using quic-portal for Modal deployments (NAT traversal via STUN).
- `quic_client_policy.py` — Client counterpart to `quic_server.py`, discovers server via Modal Dict.
- `modal_asgi.py` — Adapts openpi's `WebsocketPolicyServer` into a Starlette ASGI app for Modal.
- `modal_helpers.py` — Shared Modal utilities: Docker image builder, model loading, transformers patching.
- `benchmark.py` — Timing harness used by all test scripts.
- `relay.py` — UDP relay registration for NAT traversal fallback (WIP).

### quic-sidecar/ — Rust binary

Single-file Rust binary (`src/main.rs`) with two modes:

- `server` — Listens for QUIC connections on UDP 5555 and connects to the Python backend via Unix socket.
- `client` — Connects to a remote direct QUIC server and exposes the same local Unix-socket protocol to the Python client wrapper.

The local sidecar protocol uses length-prefixed messages with type bytes: `0x01`/`0x11` for metadata, `0x02`/`0x12` for inference, `0x03`/`0x14` for reset, and `0x13` for errors. This exists because keeping QUIC in Rust gives a cleaner low-latency path on both sides while preserving the standard Python policy interface.

### tests/

Smoke tests invoked via `main.py test *`. Each test calls `wait_for_server()` (HTTP health check), connects via the appropriate transport, runs `benchmark.run_benchmark()`, and prints timing results. `helpers.py` has the shared `random_observation_aloha()` generator.

### infra/ — Terraform

Two-tier structure:

- `infra/` (root) — Shared, one-time resources: ECR repository, S3 checkpoint bucket, IAM roles, GitHub Actions OIDC. Apply once.
- `infra/regional-instance/` — Per-region EC2 deployment. Uses Terraform workspaces (`openpi-uswest2`, `openpi-malaysia`, `openpi-seoul`). Calls the reusable module at `infra/modules/regional_inference_instance/`.
- `infra/modules/regional_inference_instance/` — The reusable module: EC2 instance, security group, optional Elastic IP, cloud-init bootstrap via `user_data.yaml.tftpl`.

## Architectural invariants

- **No openpi modifications.** We import from openpi and openpi-client as-is. If something needs to change in openpi, it should be upstreamed.
- **One policy, multiple transports.** The policy is loaded once and wrapped in `ThreadSafePolicy`. WebSocket and QUIC both call the same `infer()` behind a lock. Adding a transport means adding a thread, not duplicating model loading.
- **Rust owns the network, Python owns inference.** In Docker/EC2, the Rust sidecar terminates QUIC and speaks a simple framed protocol over a Unix socket to Python. The same local protocol is also used by the direct QUIC client wrapper, which spawns a local Rust sidecar process and keeps the Python `BasePolicy` API intact.
- **Infrastructure is separated by blast radius.** Shared resources (ECR, S3, IAM) are in the Terraform root. Regional EC2 instances are in separate workspaces so you can destroy one region without affecting others.

## Cross-cutting concerns

- **torch.compile modes** are threaded through from env var (`OPENPI_PYTORCH_COMPILE_MODE`) to Terraform user data to Docker env to `compile_mode.py` to the openpi config dataclass. The warmup pass in `serve.py` triggers compilation before the server starts accepting connections.
- **Health checks** — WebSocket server exposes `/healthz` on the same TCP port. All test scripts and the Terraform cloud-init poll this before attempting inference.
- **Checkpoint download** — openpi's `maybe_download()` handles `gs://`, `s3://`, and local paths. EC2 instances authenticate to S3 via IAM instance profile (no credentials in config).
