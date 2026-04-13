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

Copy [`config.example.json`](config.example.json) and edit it:

```bash
cp config.example.json config.json
# Edit config.json: set your model and checkpoint
```

| Field | Description |
|-------|-------------|
| `model_config_name` | openpi training config name (e.g. `pi05_aloha`, `pi0_aloha_sim`, `pi05_droid`) |
| `checkpoint_dir` | Local path, `gs://`, or `s3://` URI to model checkpoint |
| `default_prompt` | Optional default text prompt if not provided per-request |
| `port` | Server port (default: 8000) |
| `max_concurrent_requests` | Max simultaneous inferences (default: 1) |

## Running locally

```bash
uv run python main.py serve --config config.json
```

Set `OPENPI_PYTORCH_COMPILE_MODE` to override the serving compile mode at runtime.
Accepted values: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`.

Local serving uses the Rust QUIC sidecar for direct QUIC. If you are not using the Docker image, build the sidecar locally and point the server at the binary:

```bash
cd quic-sidecar && cargo build
cd ..
OPENPI_QUIC_SIDECAR_BINARY=$PWD/quic-sidecar/target/debug/openpi-quic-sidecar \
uv run python main.py serve --config config.json
```

## Running with Docker

```bash
# Build (from this directory)
docker build .. -t openpi-flash -f Dockerfile

# Run
docker run --rm --gpus=all \
  -v ./config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  -p 5555:5555/udp \
  openpi-flash
```

Or with Docker Compose:

```bash
docker compose --profile openpi up --build
```

The Docker image builds and runs a Rust QUIC sidecar by default for the direct EC2/AWS QUIC path. The Python process still owns policy loading and inference; the sidecar only terminates QUIC and forwards requests over a local Unix socket.

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

QUIC provides lower and more consistent latency than WebSocket for direct connections. Use it as the default transport for EC2/Docker deployments where the server has a stable public IP and UDP is not blocked. The Python client keeps the normal `policy.infer()` API, but direct QUIC now runs through a local Rust sidecar process on the client machine.

> **Note:** On Modal, QUIC requires NAT traversal (STUN + UDP hole punching) which is unreliable — it fails frequently depending on the NAT type assigned to the container. Use WebSocket for Modal deployments.
>
> For direct QUIC, the client machine also needs the `openpi-quic-sidecar` binary installed. Set `OPENPI_QUIC_SIDECAR_BINARY` if it is not on the default path.

```python
from hosting.direct_quic_client_policy import DirectQuicClientPolicy

client = DirectQuicClientPolicy(host="your-ec2-ip", port=5555)
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

The shared AWS resources (ECR, IAM roles, S3 bucket) can be set up with Terraform or manually:

- **Terraform/OpenTofu (recommended):** See [`infra/`](infra/) — run `terraform apply` to create everything
- **Manual CLI:** See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md) for step-by-step `aws` commands

### PyTorch checkpoint

The Docker image runs PyTorch inference, which requires a pre-converted checkpoint. The default JAX checkpoints from `gs://openpi-assets` won't work in the Docker image (no JAX dependencies).

A pre-converted checkpoint is stored in S3:

```json
{
  "checkpoint_dir": "s3://openpi-checkpoints-us-west-2/pi05_base_pytorch"
}
```

The checkpoint is downloaded on first startup and cached in `/cache/models`. To convert a different model, see [Converting checkpoints to PyTorch](#converting-checkpoints-to-pytorch) and upload to S3:

```bash
modal volume get openpi-model-weights <output-name> /tmp/<output-name>
aws s3 sync /tmp/<output-name>/<output-name>/ s3://openpi-checkpoints-us-west-2/<output-name>/
```

### Launching an EC2 instance

See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md) for full details. The short version:

1. Launch a **g6e.xlarge** (L40S GPU) with **Ubuntu 24.04**, **100 GiB** gp3, IAM profile `ec2-ecr-pull`
2. Install Docker + NVIDIA Container Toolkit
3. Pull and run:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Pull and run
docker pull "${ECR_REGISTRY}/openpi-flash:latest"
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 -p 5555:5555/udp \
  --name openpi-inference \
  "${ECR_REGISTRY}/openpi-flash:latest"
```

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
              QUIC (UDP 5555, recommended)
client ─┬──────────────────────────────────> QUIC sidecar ─┐
        │                                                   ├─> Policy.infer() -> response
        └─ WebSocket (TCP 8000, fallback) ─> WS server ────┘
                                               |
                                               +-- /healthz endpoint (HTTP 200)
                                               +-- server_timing (infer_ms, prev_total_ms)
                                               +-- msgpack binary protocol
```
