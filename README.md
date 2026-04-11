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
| `checkpoint_dir` | Local path or `gs://` URI to model checkpoint |
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

This starts a dev server with a temporary URL. Modal prints the URL — it looks like:

```
https://<your-workspace>--openpi-inference-openpiinference-serve-dev.modal.run
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

#### ASGI mode (`modal_app.py`)

```bash
uv run python test_modal.py wss://<your-modal-url>
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

# For Modal (use wss:// with the URL printed by modal serve/deploy):
client = wcp.WebsocketClientPolicy(
    host="wss://your-workspace--openpi-inference-openpiinference-serve.modal.run",
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
curl https://<modal-url>/healthz
```

## Linting and typechecking

```bash
uv run ruff check src/hosting/
uv run ruff format src/hosting/
uv run ty check src/hosting/
```

## Deploying to AWS

### 1. Launch an EC2 GPU instance

**Option A: AWS Console**

1. Go to **EC2 > Launch Instance**
2. Name: `openpi-inference`
3. AMI: search for **Ubuntu 24.04** (select the 64-bit x86 version)
4. Instance type: **g6e.xlarge** (1x L40S GPU, 4 vCPUs, 32 GB RAM)
5. Key pair: select or create one for SSH access
6. Network settings: create or select a security group that allows:
   - SSH (TCP 22) from your IP
   - Custom TCP 8000 (or 443 if using Caddy/ALB for HTTPS)
7. Storage: increase root volume to **100 GiB** (gp3)
8. Launch the instance

**Option B: AWS CLI**

```bash
# Find the latest Ubuntu 24.04 AMI for your region
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g6e.xlarge \
  --key-name your-keypair \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=openpi-inference}]'
```

### 2. Install dependencies on the instance

```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>

# Install Docker with NVIDIA GPU support
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Log out and back in for docker group to take effect
exit
```

### 3. Build and run on the instance

```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>

# Clone both repos
git clone https://github.com/Physical-Intelligence/openpi ~/openpi/openpi
git clone https://github.com/Hebbian-Robotics/openpi-hosting ~/openpi/hosting

# Build the image on the instance (native amd64, no cross-compile issues)
cd ~/openpi/hosting
docker build .. -t openpi-hosted -f Dockerfile

# Create your config
cp config.example.json config.json
# Edit config.json: set your model, etc.

# Run
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  --name openpi-inference \
  openpi-hosted
```

To deploy a new version:

```bash
cd ~/openpi/hosting && git pull
cd ~/openpi/openpi && git pull
cd ~/openpi/hosting
docker build .. -t openpi-hosted -f Dockerfile
docker stop openpi-inference && docker rm openpi-inference
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  --name openpi-inference \
  openpi-hosted
```

### 4. Back up the image to ECR

Push to AWS Elastic Container Registry so you can restore quickly if the instance is replaced.

```bash
# Create an ECR repository (one-time, from any machine with AWS CLI)
aws ecr create-repository --repository-name openpi-hosting --region us-east-1

# On the instance: login, tag, and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag openpi-hosted:latest $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/openpi-hosting:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/openpi-hosting:latest
```

The image is large (several GB due to CUDA + model dependencies). The initial push takes a while but subsequent pushes only upload changed layers.

To restore on a new instance, pull instead of building:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker pull $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/openpi-hosting:latest
docker tag $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/openpi-hosting:latest openpi-hosted
```

Ensure the EC2 instance has an IAM role with `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, and `ecr:PutImage` permissions.

### 5. HTTPS with Caddy (recommended for single instance)

Install Caddy on the instance and reverse-proxy to the Docker container:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
  sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
  sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

Create `/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo systemctl restart caddy
```

Caddy automatically provisions and renews TLS certificates via Let's Encrypt. Point your domain's DNS A record to the instance's public IP.

### 6. HTTPS with ALB (alternative)

If you prefer AWS-managed TLS:

1. Create a Target Group (protocol: HTTP, port: 8000, health check path: `/healthz`)
2. Register your EC2 instance
3. Create an Application Load Balancer with an HTTPS listener (port 443)
4. Attach an ACM certificate for your domain
5. Point your domain's DNS to the ALB

## Architecture

```
client -> WebSocket -> WebsocketPolicyServer -> Policy.infer() -> response
                         |
                         +-- /healthz endpoint (HTTP 200)
                         +-- server_timing (infer_ms, prev_total_ms)
                         +-- msgpack binary protocol
```

The server reuses openpi's `create_trained_policy()` and `Policy.infer()` directly. No openpi code is modified. Both JAX and PyTorch checkpoints are supported (auto-detected).
