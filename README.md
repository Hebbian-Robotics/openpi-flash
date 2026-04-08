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
git clone https://github.com/kstonekuan/openpi-hosting
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

### Model weight caching

The first cold start downloads model weights and caches them to a Modal Volume (`openpi-model-weights`). Subsequent cold starts load from the volume, which is much faster.

The volume is created automatically on the first deploy. To inspect its contents afterwards:

```bash
uv run modal volume ls openpi-model-weights
```

## Connecting

Use the standard `openpi-client`:

```python
from openpi_client import websocket_client_policy as wcp

# For EC2/Docker:
client = wcp.WebsocketClientPolicy(host="localhost", port=8000)

# For Modal (use the URL printed by modal serve/deploy):
client = wcp.WebsocketClientPolicy(
    host="your-workspace--openpi-inference-openpiinference-serve.modal.run",
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
git clone https://github.com/kstonekuan/openpi-hosting ~/openpi/hosting

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

The image is large (several GB due to CUDA + PyTorch + JAX). The initial push takes a while but subsequent pushes only upload changed layers.

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

## VLASH inference (alternative backend)

The hosting service also supports [VLASH](https://github.com/kstonekuan/vlash) policies as an alternative to openpi. VLASH uses async chunk pre-computation: after the first request, the server pre-computes the next action chunk in the background so subsequent responses are near-instant.

### Configuration

Copy [`config.vlash.example.json`](config.vlash.example.json) and edit it:

```bash
cp config.vlash.example.json config.vlash.json
# Edit config.vlash.json: set your model and checkpoint
```

| Field | Description |
|-------|-------------|
| `policy_type` | VLASH policy type (`pi0` or `pi05`) |
| `pretrained_path` | HuggingFace hub name or local path to checkpoint |
| `model_version` | Arbitrary string included in logs and responses |
| `task` | Language prompt / task description |
| `robot_type` | Robot type string for observation preprocessing |
| `compile_model` | Whether to warmup `torch.compile` on startup (default: false) |
| `port` | Server port (default: 8000) |
| `max_concurrent_requests` | Max simultaneous inferences (default: 1) |

### Running locally

```bash
INFERENCE_CONFIG_PATH=config.vlash.json uv run python -m hosting.serve_vlash
```

### Running with Docker

```bash
# Build (from this directory)
docker build .. -t vlash-hosted -f Dockerfile.vlash

# Run
docker run --rm --gpus=all \
  -v ./config.vlash.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 \
  vlash-hosted
```

Or with Docker Compose:

```bash
docker compose --profile vlash up --build
```

Note: the openpi service uses the `openpi` profile (`docker compose --profile openpi up --build`).
