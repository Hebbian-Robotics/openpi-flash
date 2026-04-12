# openpi-hosting

Hosted inference service for [openpi](https://github.com/Physical-Intelligence/openpi). Wraps openpi's policy inference in a WebSocket server with concurrency control and health checks. Supports deployment on AWS EC2 (Docker) or [Modal](https://modal.com).

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- A GPU with CUDA support (for inference)

## Setup

Clone both repos side by side under a shared parent directory:

```bash
git clone https://github.com/Physical-Intelligence/openpi
git clone https://github.com/Hebbian-Robotics/openpi-hosting
```

Then install dependencies:

```bash
cd openpi-hosting
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
| `model_version` | Arbitrary string included in logs and responses |
| `default_prompt` | Optional default text prompt if not provided per-request |
| `port` | Server port (default: 8000) |
| `max_concurrent_requests` | Max simultaneous inferences (default: 1) |

## Running locally

```bash
INFERENCE_CONFIG_PATH=config.json uv run python -m hosting.serve
```

## Running with Docker

```bash
# Build (from this directory)
docker build .. -t openpi-hosted -f Dockerfile

# Run
docker run --rm --gpus=all \
  -v ./config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  openpi-hosted
```

Or with Docker Compose:

```bash
docker compose --profile openpi up --build
```

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

# With a model version and default prompt
uv run modal deploy modal_app.py \
  --model-version pi0_sim_v2 \
  --default-prompt "pick up the cube"
```

### Changing the GPU

Edit `modal_app.py` and change the `gpu=` parameter on `@app.cls()`:

```python
@app.cls(
    gpu="A10G",  # or "L4", "L40S", "A100", "H100"
    ...
)
```

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

### QUIC portal mode (experimental)

`modal_quic_app.py` uses [quic-portal](https://github.com/Hebbian-Robotics/quic-portal) for direct peer-to-peer QUIC transport with automatic NAT traversal via STUN + UDP hole punching. This avoids TCP head-of-line blocking for potentially lower latency than the tunnel mode.

```bash
uv run modal run modal_quic_app.py
```

Connection is coordinated through a shared Modal Dict (`openpi-quic-info`) — no URL or port to manage. Note: NAT traversal only works with "easy" NATs. Fall back to the tunnel mode if connectivity issues arise.

### QUIC relay fallback (WIP)

> **Status: Not yet working end-to-end.** The relay server forwards traffic correctly, but full QUIC-over-relay has not been verified.

When hole punching fails (symmetric NATs, corporate firewalls), the QUIC app can fall back to a [UDP relay](https://github.com/Hebbian-Robotics/quic-relay) server.

**Prerequisites:**

1. [quic-portal fork](https://github.com/Hebbian-Robotics/quic-portal) with `SO_REUSEPORT` — install with `pip install git+https://github.com/Hebbian-Robotics/quic-portal.git`
2. [quic-relay](https://github.com/Hebbian-Robotics/quic-relay) running on AWS EC2 — see its README for deployment instructions

**Setup:**

Create `hosting/.env`:

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

#### EC2/Docker or Modal ASGI

```bash
uv run python test_server.py ws://localhost:8000          # EC2/Docker (plain)
uv run python test_server.py wss://$EC2_HTTPS_HOST   # EC2 with HTTPS
uv run python test_server.py wss://$MODAL_HOSTNAME   # Modal
```

#### Tunnel mode (`modal_tunnel_app.py`)

```bash
uv run python test_modal_tunnel.py
```

#### QUIC portal mode (`modal_quic_app.py`)

```bash
uv run python test_modal_quic.py
```

### Model weight caching

The first cold start downloads model weights and caches them to a Modal Volume (`openpi-model-weights`). Subsequent cold starts load from the volume, which is much faster. For PyTorch inference, convert the checkpoint first (see [Converting checkpoints to PyTorch](#converting-checkpoints-to-pytorch)).

The volume is created automatically on the first deploy. To inspect its contents:

```bash
uv run modal volume ls openpi-model-weights
```

## Connecting

Use the standard `openpi-client`:

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

## Linting and typechecking

```bash
uv run ruff check src/hosting/
uv run ruff format src/hosting/
uv run ty check src/hosting/
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
docker pull "${ECR_REGISTRY}/openpi-hosted:latest"
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 -p 5555:5555/udp \
  --name openpi-inference \
  "${ECR_REGISTRY}/openpi-hosted:latest"
```

### Updating to a new version

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"
docker pull "${ECR_REGISTRY}/openpi-hosted:latest"
docker stop openpi-inference && docker rm openpi-inference
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 -p 5555:5555/udp \
  --name openpi-inference \
  "${ECR_REGISTRY}/openpi-hosted:latest"
```

You can pin to a specific commit: `openpi-hosted:<commit-sha>` instead of `:latest`.

### Pushing dev images

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

# Build and push with a dev tag
docker build .. -t "${ECR_REGISTRY}/openpi-hosted:dev" -f Dockerfile
docker push "${ECR_REGISTRY}/openpi-hosted:dev"
```

ECR keeps `latest` plus the 3 most recent images; older ones are cleaned up automatically.

### HTTPS

**Caddy (recommended for single instance):** See [`docs/aws-manual-setup.md`](docs/aws-manual-setup.md#5-optional-https-with-caddy) — auto-provisions TLS via Let's Encrypt.

**ALB (alternative):** Create a Target Group (HTTP 8000, health check `/healthz`), an HTTPS ALB listener with an ACM certificate, and point DNS to the ALB.

## Architecture

```
client -> WebSocket -> WebsocketPolicyServer -> Policy.infer() -> response
                         |
                         +-- /healthz endpoint (HTTP 200)
                         +-- server_timing (infer_ms, prev_total_ms)
                         +-- msgpack binary protocol
```

The server reuses openpi's `create_trained_policy()` and `Policy.infer()` directly. No openpi code is modified. Both JAX and PyTorch checkpoints are supported (auto-detected).
