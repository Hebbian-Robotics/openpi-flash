# Regional Inference Instance

Terraform root for deploying a single OpenPI inference EC2 instance in any AWS region.

This stack is intentionally separate from the shared [`../`](../README.md) infrastructure:

- Shared stack: ECR, IAM roles, instance profile
- Regional stack: subnet-bound EC2 instance, security group, optional Elastic IP, bootstrap, and runtime service

## What it deploys

- 1 EC2 instance with a GPU-friendly default instance type (`g6e.xlarge`)
- A dedicated security group with ingress for SSH, the action slot (TCP 8000 + UDP 5555, only when the action slot is configured), the planner slot (TCP 8002 + UDP 5556, only when the planner slot is configured), and optional HTTPS. The planner admin endpoint (TCP 8001) is only published on the host's `127.0.0.1` — never exposed to the security group.
- Optional Elastic IP
- Cloud-init bootstrap that:
  - installs Docker, AWS CLI, and NVIDIA Container Toolkit
  - writes `/etc/openpi/config.json`
  - creates a systemd unit that logs in to ECR, pulls the image, and runs the container (publishing only the slots that are enabled in the config)
  - sets runtime env vars for PyTorch compile mode

## Usage

```bash
SHARED_ECR_REPOSITORY_URL=$(terraform -chdir=../ output -raw ecr_repository_url)

cd infra/regional-instance
terraform init
terraform apply \
  -var aws_region=ap-northeast-2 \
  -var ecr_repository_url="$SHARED_ECR_REPOSITORY_URL" \
  -var 'ssh_public_key=ssh-ed25519 AAAA... user@host' \
  -var assign_elastic_ip=true \
  -var openpi_pytorch_compile_mode=default
```

SSH, WebSocket (both slots), and QUIC (both slots) ports default to open (`0.0.0.0/0`). Override with `allowed_ssh_cidr_blocks`, `allowed_websocket_cidr_blocks`, `allowed_quic_cidr_blocks` to restrict; the planner-slot ingress rules reuse the same CIDR lists as the action slot.

When `subnet_id` is omitted, the module discovers the default VPC's first public subnet. Use `availability_zone` to target a specific AZ (required in some regions where GPU instances aren't available in all AZs).

## Notes

- Pass `ecr_repository_url` from the shared root output rather than hardcoding an account-specific ECR registry URL.
- Cross-region ECR pulls are acceptable for testing and steady-state serving, but they increase cold-start time.
- The default AMI is the latest Deep Learning AMI GPU PyTorch (Ubuntu 24.04), resolved via SSM parameter. This AMI ships with Docker, nvidia-container-toolkit, CUDA, and AWS CLI pre-installed. Override with `ami_id` or change the PyTorch version via `dlami_ssm_slug`.
- This stack expects the shared IAM instance profile (`ec2-ecr-pull`) to already exist.
- `ssh_key_name` and `ssh_public_key` are mutually exclusive. Use `ssh_key_name` to reference a pre-existing AWS key pair, or `ssh_public_key` to have Terraform create one.
