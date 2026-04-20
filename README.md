# openpi-flash

Real-time inference engine for [openpi](https://github.com/Physical-Intelligence/openpi). Optimized for low-latency policy serving over QUIC and WebSocket. Deploy on AWS EC2 (Docker) or [Modal](https://modal.com).

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A GPU with CUDA support (recommended L40S)

## Setup

Clone both repos side by side under a shared parent directory:

```bash
git clone https://github.com/Physical-Intelligence/openpi
git clone https://github.com/Hebbian-Robotics/openpi-flash
```

Then install dependencies:

```bash
cd openpi-flash
uv venv --python 3.11
uv sync
```

This installs the hosting service and pulls in `openpi` and `openpi-client` (from `openpi/packages/openpi-client`) as editable path dependencies from the sibling `openpi` directory.

## Configuration

The server has two optional component slots — `action` (PyTorch or JAX action policy) and `planner` (JAX subtask generator). At least one must be set. Which slots are present determines the mode:

- `action_only` — serves actions only (the default; matches every example in this README up to the "Subtask generation" section below).
- `planner_only` — serves subtask text only; no PyTorch action model is loaded.
- `combined` — both slots; the action endpoint auto-augments prompts with generated subtasks, and the planner endpoint is independently callable.

Copy [`config.example.json`](config.example.json) (action-only) or [`config.example.planner.json`](config.example.planner.json) (combined) and edit it:

```bash
cp config.example.json config.json
# Edit config.json: set your model and checkpoint
```

### Action slot (`action.*`)

| Field | Description |
|-------|-------------|
| `model_config_name` | openpi training config name (e.g. `pi05_aloha`, `pi0_aloha_sim`, `pi05_droid`) |
| `checkpoint_dir` | Local path, `gs://`, or `hf://` URI to model checkpoint |
| `default_prompt` | Optional default text prompt if not provided per-request |

### Planner slot (`planner.*`)

See the [Subtask generation (planner)](#subtask-generation-planner) section below for end-to-end usage.

| Field | Description |
|-------|-------------|
| `checkpoint_dir` | JAX Orbax checkpoint path (local or `gs://`). JAX-only. |
| `max_generation_tokens` | Max tokens per subtask decode (default: `20`) |
| `generation_prompt_format` | Prompt passed to the planner. Must contain `{task}`. Default: `"Task: {task}. Subtask: "` |
| `action_prompt_template` | Template that splices the generated subtask back into the action prompt in combined mode. Must contain `{task}` and `{subtask}`. Default: `"{task}. Subtask: {subtask}"` |

### Transports

Each loaded slot gets its own port triple (WebSocket + QUIC + unix socket). Defaults:

| Slot | WebSocket | QUIC (UDP) | Unix socket |
|------|-----------|------------|-------------|
| `action` | `8000` | `5555` | `/tmp/openpi-action.sock` |
| `planner` | `8002` | `5556` | `/tmp/openpi-planner.sock` |

The admin HTTP endpoint (port `8001`) is started only when the planner slot is loaded. Override any of these under the `action_transport` / `planner_transport` keys in `config.json`. Environment variables prefixed with `OPENPI_` (e.g. `OPENPI_ACTION__CHECKPOINT_DIR`) override JSON values on a deployed box — nested fields use `__` as the delimiter.

Set `OPENPI_PYTORCH_COMPILE_MODE` to override the serving compile mode at runtime.
Accepted values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. In our testing, `default` gives a good balance of compile time (~80s) and inference speed (~2x faster than eager). `max-autotune-no-cudagraphs` can be slightly faster at inference but takes significantly longer to compile (~5 min); this could be worth it when optimizing for inference time.

## Subtask generation (planner)

The planner slot runs pi0.5 two-phase inference: from `(task, images)` it generates a short subtask string (e.g. `"pick up cup"` from `"pick up the red cup"`), which can either be returned directly or spliced into the action prompt before action inference. See pi0.5 paper §V.E, Figure 7 for the architecture. The planner is **JAX-only** today — point `planner.checkpoint_dir` at a JAX Orbax checkpoint such as `gs://openpi-assets/checkpoints/pi05_base/params`.

### Example combined-mode config

```json
{
  "action": {
    "model_config_name": "pi05_aloha",
    "checkpoint_dir": "/cache/models/pi05_base_openpi"
  },
  "planner": {
    "checkpoint_dir": "gs://openpi-assets/checkpoints/pi05_base/params",
    "max_generation_tokens": 20
  }
}
```

Start the server exactly as in [Running locally](#running-locally) / [Running with Docker](#running-with-docker). At startup you should see:

```
Service ready (mode=combined, slots=['action', 'planner'])
```

### Client: planner-only endpoint

Clients connect to port `8002` (WebSocket) or `5556` (QUIC) and receive `{"subtask": {"text": ..., "ms": ...}}` — no `actions` field.

```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy(host="localhost", port=8002)
result = client.infer({
    "prompt": "pick up the red cup",
    "images": {"cam_high": image_array},
})
print(result["subtask"]["text"])   # e.g. "pick up cup"
print(result["subtask"]["ms"])     # generation latency
```

For QUIC, use the planner port:

```python
from hosting.flash_transport_policy import FlashTransportPolicy

client = FlashTransportPolicy(host="your-ec2-ip", port=5556)
result = client.infer(observation)
```

### Client: combined mode on the action endpoint

In combined mode the action endpoint (port `8000` / `5555`) transparently runs the planner first and augments the prompt using `action_prompt_template` before action inference. The response shape adds a `subtask` field alongside the usual `actions`. Client code needs no changes beyond pointing at the action port.

An optional `obs["mode"]` switches behavior on the action endpoint:

| `mode` | Behavior | Response |
|--------|----------|----------|
| `"default"` (or omitted) | Run planner, then action with augmented prompt | `actions` + `subtask` |
| `"subtask_only"` | Run planner only; skip action inference | `subtask` only |
| `"action_only"` | Skip planner; run action with the original prompt | `actions` only |

### Admin HTTP endpoint

When the planner is loaded, a small FastAPI admin server listens on port `8001`. It lets you mutate the decode prompt without a restart:

```bash
curl http://localhost:8001/config
curl -X PATCH http://localhost:8001/config \
  -H 'content-type: application/json' \
  -d '{"generation_prompt_format": "Task: {task}. Subtask: "}'
```

Swagger UI is available at `http://localhost:8001/docs`. In Docker/EC2 deploys, bind this port to `127.0.0.1` only so it's not internet-reachable.

### Caveats

- Planner is JAX-only. PyTorch is not supported today.
- `generation_prompt_format` must contain `{task}`; `action_prompt_template` must contain both `{task}` and `{subtask}`. Invalid templates fail at config-load time.

## Running locally

```bash
# Prepare the default local checkpoint from upstream sources
uv run python main.py prepare-checkpoint

# Serve
uv run python main.py serve --config config.json
```

For non-Docker local runs, point `checkpoint_dir` at the prepared checkpoint in your local OpenPI cache, typically `$HOME/.cache/openpi/pi05_base_openpi`.

Local serving uses the `openpi-flash-transport` binary for direct QUIC. If you are not using the Docker image, you can either download a pre-built binary from [GitHub Actions](./.github/workflows/docker-build.yml) (look for the `flash-transport-x86_64-linux` or `flash-transport-aarch64-linux` artifact on the latest run) or build it locally with Rust:

```bash
# Option 1: Download from GitHub Actions (no Rust needed)
# Go to Actions → Docker Build → latest run → Artifacts → flash-transport-x86_64-linux
chmod +x openpi-flash-transport
OPENPI_FLASH_TRANSPORT_BINARY=$PWD/openpi-flash-transport \
uv run python main.py serve --config config.json

# Option 2: Build locally (requires Rust)
cd flash-transport && cargo build
cd ..
OPENPI_FLASH_TRANSPORT_BINARY=$PWD/flash-transport/target/debug/openpi-flash-transport \
uv run python main.py serve --config config.json
```

## Running with Docker

```bash
# Build (from this directory)
docker build .. -t openpi-flash -f Dockerfile

# Prepare the checkpoint cache once
docker volume create openpi-inference-cache
docker run --rm \
  -v openpi-inference-cache:/cache \
  openpi-flash \
  python main.py prepare-checkpoint

# Run
docker run --rm --gpus=all \
  -v ./config.json:/config/config.json:ro \
  -v openpi-inference-cache:/cache \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  -p 5555:5555/udp \
  openpi-flash
```

When the planner slot is enabled, also publish its transport triple (and the admin port, bound to localhost):

```bash
  -p 8002:8002 \
  -p 5556:5556/udp \
  -p 127.0.0.1:8001:8001 \
```

Or with Docker Compose:

```bash
docker compose --profile openpi up --build
```

The Docker image builds and runs `openpi-flash-transport` by default for the direct EC2/AWS QUIC path. The Python process still owns policy loading and inference; the transport binary only terminates QUIC and forwards requests over a local Unix socket.

## Running on Modal

[Modal](https://modal.com) provides serverless GPU infrastructure — no EC2 instances to manage. The model loads on container startup and scales to zero when idle.

### Prerequisites

```bash
uv run modal setup  # one-time auth
```

### Development (hot-reload)

```bash
uv run modal serve modal_app.py
```

This starts a dev server with a temporary URL. Modal prints a host ending in `.modal.run`, with a shape like:

```
<your-workspace>--openpi-inference-openpiinference-serve-dev.modal.run
```

### Production deploy

```bash
uv run modal deploy modal_app.py
```

This creates a persistent URL that stays up and scales automatically.

### Customizing the model

Pass Modal parameters to change the model config:

```bash
# Different model config
uv run modal deploy modal_app.py \
  --model-config-name pi0_aloha_sim \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi0_aloha_sim

# With a default prompt
uv run modal deploy modal_app.py \
  --default-prompt "pick up the cube"
```

### Changing the GPU

Edit `modal_app.py` and change the `gpu=` parameter on `@app.cls()` (`"L4"`, `"L40S"`, `"A10G"`, `"A100"`, `"H100"`).

### Converting checkpoints to PyTorch

The default deployment uses PyTorch checkpoints for inference. To convert a JAX checkpoint to PyTorch format on Modal:

```bash
# Convert the default pi05_base checkpoint
uv run modal run convert_checkpoint_modal.py

# Convert a different checkpoint
uv run modal run convert_checkpoint_modal.py \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_droid \
  --config-name pi05_droid \
  --output-name pi05_droid_pytorch
```

The converted checkpoint is saved to the `openpi-model-weights` Modal Volume. Verify with:

```bash
uv run modal volume ls openpi-model-weights/pi05_base_pytorch
```

### Low-latency mode (tunnel)

The default `modal_app.py` serves through Modal's ASGI layer (Starlette). For lower latency, `modal_tunnel_app.py` uses a direct TCP tunnel to run openpi's native `WebsocketPolicyServer`, bypassing the ASGI layer.

```bash
# Deploy the function (registers it, but doesn't start a container)
uv run modal deploy modal_tunnel_app.py

# Start the server (spins up the container and creates the tunnel)
uv run modal run modal_tunnel_app.py
```

The tunnel address is stored in a Modal Dict (`openpi-tunnel-info`) so clients can discover it automatically.

### QUIC portal mode (experimental, unreliable)

`modal_quic_app.py` uses [quic-portal](https://github.com/Hebbian-Robotics/quic-portal) for direct peer-to-peer QUIC transport with automatic NAT traversal via STUN + UDP hole punching. This avoids TCP head-of-line blocking for potentially lower latency than the tunnel mode.

```bash
uv run modal run modal_quic_app.py
```

Connection is coordinated through a shared Modal Dict (`openpi-quic-info`) — no URL or port to manage.

> **Warning:** NAT traversal on Modal is unreliable. Modal containers are assigned unpredictable NAT types, and hole punching frequently fails. Expect connection failures. For reliable QUIC, deploy on EC2 where the server has a stable public IP. For Modal, use WebSocket or the tunnel mode instead.

### QUIC relay fallback (WIP)

> **Status: Not yet working end-to-end.** The relay server forwards traffic correctly, but full QUIC-over-relay has not been verified.

When hole punching fails (symmetric NATs, corporate firewalls), the QUIC app can fall back to a [UDP relay](https://github.com/Hebbian-Robotics/quic-relay) server.

**Prerequisites:**

1. [quic-portal fork](https://github.com/Hebbian-Robotics/quic-portal) with `SO_REUSEPORT` — install with `pip install git+https://github.com/Hebbian-Robotics/quic-portal.git`
2. [quic-relay](https://github.com/Hebbian-Robotics/quic-relay) running on AWS EC2 — see its README for deployment instructions

**Setup:**

Copy `.env.example` to `.env` in the repo root:

```bash
cp .env.example .env
```

```
QUIC_RELAY_IP=your-elastic-ip
QUIC_RELAY_ONLY=true   # skip hole punch, always use relay
```

Deploy:

```bash
uv run modal deploy modal_quic_app.py
```

The Modal app reads `.env` via `modal.Secret.from_dotenv()` and injects `QUIC_RELAY_IP` into the container. The client discovers the relay address from the Modal Dict automatically — no client-side config needed.

### Testing

All test scripts run a warmup inference followed by 5 timed iterations, reporting per-iteration timing and a summary with mean/min/max.

```bash
uv run python main.py test quic <host>                # QUIC direct (recommended for EC2/Docker)
uv run python main.py test ws ws://<host>:8000         # WebSocket (EC2/Docker)
uv run python main.py test ws wss://$MODAL_HOSTNAME    # WebSocket (Modal)
uv run python main.py test modal-tunnel                # Modal tunnel mode
uv run python main.py test modal-quic                  # Modal QUIC portal
```

### Model weight caching

The first cold start downloads model weights and caches them to a Modal Volume (`openpi-model-weights`). Subsequent cold starts load from the volume, which is much faster. For PyTorch inference, convert the checkpoint first (see [Converting checkpoints to PyTorch](#converting-checkpoints-to-pytorch)).

The volume is created automatically on the first deploy. To inspect its contents:

```bash
uv run modal volume ls openpi-model-weights
```

## Connecting

### QUIC (recommended for EC2/Docker)

QUIC provides lower and more consistent latency than WebSocket for direct connections. Use it as the default transport for EC2/Docker deployments where the server has a stable public IP and UDP is not blocked. The Python client keeps the normal `policy.infer()` API, but direct QUIC now runs through a local `openpi-flash-transport` subprocess on the client machine.

> **Note:** On Modal, QUIC requires NAT traversal (STUN + UDP hole punching) which is unreliable — it fails frequently depending on the NAT type assigned to the container. Use WebSocket for Modal deployments.
>
> For direct QUIC, the client machine also needs the `openpi-flash-transport` binary. You can download a pre-built binary from [GitHub Actions](./.github/workflows/docker-build.yml) (no Rust required) or build it locally from `flash-transport/`. Set `OPENPI_FLASH_TRANSPORT_BINARY` if it is not on the default path.

```python
from hosting.flash_transport_policy import FlashTransportPolicy

client = FlashTransportPolicy(host="your-ec2-ip", port=5555)
action = client.infer(observation)
client.close()
```

### WebSocket (fallback)

If QUIC doesn't work (e.g. UDP blocked by firewall, or connecting from a browser), fall back to WebSocket:

```python
from openpi_client import websocket_client_policy as wcp

# For EC2/Docker:
client = wcp.WebsocketClientPolicy(host="localhost", port=8000)

# For Modal, use the hostname printed by modal serve/deploy, for example:
# <your-workspace>--openpi-inference-openpiinference-serve-dev.modal.run
modal_hostname = "your Modal hostname"
client = wcp.WebsocketClientPolicy(
    host=f"wss://{modal_hostname}",
    port=443,
)

action = client.infer(observation)
# action["server_timing"] contains: infer_ms
```

## Health check

```bash
# EC2/Docker
curl http://localhost:8000/healthz

# Modal
curl "https://$MODAL_HOSTNAME/healthz"
```

## Deploying to AWS

### AWS infrastructure setup

The shared AWS resources (ECR and IAM roles) can be set up with Terraform or manually:

- **Terraform/OpenTofu (recommended):** See [`infra/`](infra/) — run `terraform apply` to create everything
- **Manual CLI:** See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md) for step-by-step `aws` commands

### Docker checkpoint preparation

EC2 and Docker deployments prepare a local OpenPI-compatible checkpoint in `/cache/models/pi05_base_openpi` before the inference server starts.

The preparation step combines:

- Hugging Face weights from `lerobot/pi05_base`
- normalization stats from `gs://openpi-assets/checkpoints/pi05_base/assets`

Run it manually with:

```bash
docker volume create openpi-inference-cache
docker run --rm \
  -v openpi-inference-cache:/cache \
  "${ECR_REGISTRY}/openpi-flash:latest" \
  python main.py prepare-checkpoint
```

`docker compose --profile openpi up --build` and the Terraform EC2 bootstrap run this preparation step automatically.

### Launching an EC2 instance

See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md) for full details. The short version:

1. Launch a **g6e.xlarge** (L40S GPU) with **Ubuntu 24.04**, **200 GiB** gp3, IAM profile `ec2-ecr-pull`
2. Install Docker + NVIDIA Container Toolkit
3. Pull, prepare the checkpoint, and run:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Pull and run
docker pull "${ECR_REGISTRY}/openpi-flash:latest"
docker volume create openpi-inference-cache
docker run --rm \
  -v openpi-inference-cache:/cache \
  "${ECR_REGISTRY}/openpi-flash:latest" \
  python main.py prepare-checkpoint
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -v openpi-inference-cache:/cache \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 -p 5555:5555/udp \
  --name openpi-inference \
  "${ECR_REGISTRY}/openpi-flash:latest"
```

If the planner slot is enabled, also add `-p 8002:8002 -p 5556:5556/udp -p 127.0.0.1:8001:8001` to the `docker run` command.

### Updating to a new version

Re-run the ECR login, pull, stop, and run steps from above. You can pin to a specific commit: `openpi-flash:<commit-sha>` instead of `:latest`.

### Pushing dev images

```bash
docker build .. -t "${ECR_REGISTRY}/openpi-flash:dev" -f Dockerfile
docker push "${ECR_REGISTRY}/openpi-flash:dev"
```

ECR keeps `latest` plus the 3 most recent images; older ones are cleaned up automatically.

### HTTPS

**Caddy (recommended for single instance):** See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md#5-optional-https-with-caddy) — auto-provisions TLS via Let's Encrypt.

**ALB (alternative):** Create a Target Group (HTTP 8000, health check `/healthz`), an HTTPS ALB listener with an ACM certificate, and point DNS to the ALB.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed codemap, invariants, and how the pieces fit together.

```
                                                                  Action slot
              QUIC (UDP 5555, recommended)                        (PyTorch or JAX)
client ─┬──────────────────────────────────> QUIC sidecar ─┐
        │                                                   ├─> action Policy.infer() -> {actions, [subtask]}
        └─ WebSocket (TCP 8000, fallback) ─> WS server ────┘
                                               |
                                               +-- /healthz endpoint (HTTP 200)
                                               +-- server_timing (infer_ms, prev_total_ms)
                                               +-- msgpack binary protocol

                                                                  Planner slot (optional, JAX)
              QUIC (UDP 5556)
client ─┬──────────────────────────────────> QUIC sidecar ─┐
        │                                                   ├─> PlannerPolicy.infer() -> {subtask}
        └─ WebSocket (TCP 8002) ───────────> WS server ────┘
                                               |
                                               +-- admin HTTP (port 8001): GET/PATCH /config, /docs
```

In combined mode both slots load from the same process; the action endpoint's `Policy.infer()` internally calls the shared `SubtaskGenerator` before running action inference, which is why the action response can include a `subtask` field.
