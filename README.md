# openpi-hosting

Hosted inference service for [openpi](https://github.com/Physical-Intelligence/openpi). Wraps openpi's policy inference in a WebSocket server with customer authentication, concurrency control, request tagging, and structured JSON logging.

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

This installs the hosting service and pulls in `openpi` and `openpi-client` as editable path dependencies from the sibling directory.

## Configuration

Copy [`config.example.json`](config.example.json) and edit it:

```bash
cp config.example.json config.json
# Edit config.json: set your model, checkpoint, and API key
```

| Field | Description |
|-------|-------------|
| `model_config_name` | openpi training config name (e.g. `pi05_aloha`, `pi0_aloha_sim`, `pi05_droid`) |
| `checkpoint_dir` | Local path or `gs://` URI to model checkpoint |
| `model_version` | Arbitrary string included in logs and responses |
| `default_prompt` | Optional default text prompt if not provided per-request |
| `port` | Server port (default: 8000) |
| `max_concurrent_requests` | Max simultaneous inferences (default: 1) |
| `customers` | List of `{customer_id, api_key}` objects |

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
docker compose up --build
```

## Connecting

Use the standard `openpi-client` with an `Authorization` header:

```python
from openpi_client import websocket_client_policy as wcp

client = wcp.WebsocketClientPolicy(
    host="localhost",
    port=8000,
    extra_headers={"Authorization": "Bearer your-secret-key"},
)

action = client.infer(observation)
# action["hosting"] contains: customer_id, request_id, model_version
# action["server_timing"] contains: infer_ms
```

## Health check

```bash
curl http://localhost:8000/healthz
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
# Edit config.json: set your model, API key, etc.

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

### 7. CloudWatch logging

The `compose.yml` is preconfigured with the `awslogs` driver. Ensure the instance has an IAM role with `logs:CreateLogGroup` and `logs:PutLogEvents` permissions. Logs appear in the `/openpi/inference` log group.

To use the default Docker logging instead (and view logs with `docker compose logs`), remove the `logging:` block from `compose.yml`.

### 8. Monitoring

Basic health monitoring with a CloudWatch alarm:

```bash
# Create a simple uptime check (requires Route 53 health check or external monitor)
# Or use the /healthz endpoint with your preferred monitoring tool:
curl -sf https://your-domain.com/healthz || echo "DOWN"
```

For inference latency monitoring, the structured JSON logs include `infer_ms` on every request, which CloudWatch Logs Insights can query:

```
fields @timestamp, customer_id, infer_ms
| filter message = "Inference complete"
| stats avg(infer_ms), max(infer_ms), count() by bin(5m)
```

## Architecture

```
client -> WebSocket -> HostedPolicyServer -> Policy.infer() -> response
                         |
                         +-- auth (API key check on upgrade)
                         +-- concurrency gate (semaphore, rejects when busy)
                         +-- request tagging (customer_id, request_id, model_version)
                         +-- structured JSON logging (stdout -> CloudWatch)
```

The server reuses openpi's `create_trained_policy()` and `Policy.infer()` directly. No openpi code is modified.
