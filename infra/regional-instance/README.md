# Regional Inference Instance

Terraform root for deploying a single OpenPI inference EC2 instance in any AWS region.

This stack is intentionally separate from the shared [`../`](../README.md) infrastructure:

- Shared stack: ECR, S3 checkpoints, IAM roles, instance profile
- Regional stack: subnet-bound EC2 instance, security group, optional Elastic IP, bootstrap, and runtime service

## What it deploys

- 1 EC2 instance with a GPU-friendly default instance type (`g6e.xlarge`)
- A dedicated security group for SSH, WebSocket, QUIC, and optional HTTPS ingress
- Optional Elastic IP
- Cloud-init bootstrap that:
  - installs Docker, AWS CLI, and NVIDIA Container Toolkit
  - writes `/etc/openpi/config.json`
  - creates a systemd unit that logs in to ECR, pulls the image, and runs the container
  - sets runtime env vars for PyTorch compile mode and QUIC backend

## Usage

```bash
SHARED_ECR_REPOSITORY_URL=$(terraform -chdir=../ output -raw ecr_repository_url)

cd infra/regional-instance
terraform init
terraform apply \
  -var aws_region=ap-northeast-2 \
  -var ecr_repository_url="$SHARED_ECR_REPOSITORY_URL" \
  -var subnet_id=subnet-xxxxxxxx \
  -var ssh_key_name=your-keypair \
  -var openpi_pytorch_compile_mode=max-autotune-no-cudagraphs \
  -var openpi_quic_backend=rust-sidecar \
  -var='allowed_ssh_cidr_blocks=["203.0.113.10/32"]' \
  -var='allowed_websocket_cidr_blocks=["203.0.113.10/32"]' \
  -var='allowed_quic_cidr_blocks=["203.0.113.10/32"]'
```

## Notes

- Pass `ecr_repository_url` from the shared root output rather than hardcoding an account-specific ECR registry URL.
- The stack still defaults to the shared checkpoint bucket path in `us-west-2`.
- Cross-region ECR pulls and S3 checkpoint downloads are acceptable for testing and steady-state serving, but they increase cold-start time.
- The default AMI lookup uses the latest Deep Learning Base OSS Nvidia Driver GPU AMI for Ubuntu 24.04 x86_64 in the selected region. If you need a different image, pass `ami_id`.
- This stack expects the shared IAM instance profile (`ec2-ecr-pull`) to already exist.
