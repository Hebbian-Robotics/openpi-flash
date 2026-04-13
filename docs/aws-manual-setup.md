# AWS Manual Setup

Step-by-step CLI instructions to set up the AWS infrastructure without Terraform. For the Terraform approach, see [`infra/`](../infra/).

## Shared infrastructure (one-time)

These resources are created once in your primary region and shared across all deployments.

### 1. ECR repository

```bash
aws ecr create-repository \
  --repository-name openpi-flash \
  --region us-west-2

# Lifecycle policy: keep latest + 3 most recent images
aws ecr put-lifecycle-policy \
  --repository-name openpi-flash \
  --region us-west-2 \
  --lifecycle-policy-text '{
    "rules": [{
      "rulePriority": 1,
      "description": "Keep only 3 most recent untagged images",
      "selection": {
        "tagStatus": "untagged",
        "countType": "imageCountMoreThan",
        "countNumber": 3
      },
      "action": { "type": "expire" }
    }]
  }'
```

### 2. GitHub Actions OIDC provider

This lets GitHub Actions assume an AWS role without static credentials.

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list ffffffffffffffffffffffffffffffffffffffff
```

### 3. IAM role for GitHub Actions (ECR push)

```bash
# Get your account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create the role with GitHub OIDC trust
aws iam create-role \
  --role-name github-actions-ecr-push \
  --assume-role-policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Principal\": {
        \"Federated\": \"arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com\"
      },
      \"Action\": \"sts:AssumeRoleWithWebIdentity\",
      \"Condition\": {
        \"StringEquals\": {
          \"token.actions.githubusercontent.com:aud\": \"sts.amazonaws.com\"
        },
        \"StringLike\": {
          \"token.actions.githubusercontent.com:sub\": \"repo:Hebbian-Robotics/openpi-flash:*\"
        }
      }
    }]
  }"

# Attach ECR push permissions
aws iam put-role-policy \
  --role-name github-actions-ecr-push \
  --policy-name ecr-push \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"ecr:GetAuthorizationToken\"],
        \"Resource\": \"*\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [
          \"ecr:BatchCheckLayerAvailability\",
          \"ecr:BatchGetImage\",
          \"ecr:CompleteLayerUpload\",
          \"ecr:GetDownloadUrlForLayer\",
          \"ecr:InitiateLayerUpload\",
          \"ecr:PutImage\",
          \"ecr:UploadLayerPart\"
        ],
        \"Resource\": \"arn:aws:ecr:us-west-2:${ACCOUNT_ID}:repository/openpi-flash\"
      }
    ]
  }"
```

### 4. IAM role for EC2 instances (ECR pull + S3 read)

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create the role
aws iam create-role \
  --role-name ec2-ecr-pull \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }]
  }'

# ECR pull permissions
aws iam attach-role-policy \
  --role-name ec2-ecr-pull \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# S3 checkpoint read permissions
aws iam put-role-policy \
  --role-name ec2-ecr-pull \
  --policy-name s3-checkpoint-read \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::openpi-checkpoints-us-west-2",
        "arn:aws:s3:::openpi-checkpoints-us-west-2/*"
      ]
    }]
  }'

# Create instance profile and attach the role
aws iam create-instance-profile --instance-profile-name ec2-ecr-pull
aws iam add-role-to-instance-profile \
  --instance-profile-name ec2-ecr-pull \
  --role-name ec2-ecr-pull
```

### 5. S3 bucket for checkpoints

```bash
aws s3 mb s3://openpi-checkpoints-us-west-2 --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket openpi-checkpoints-us-west-2 \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket openpi-checkpoints-us-west-2 \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'

# Block public access
aws s3api put-public-access-block \
  --bucket openpi-checkpoints-us-west-2 \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

## Per-instance setup

If you want a reusable, region-agnostic Terraform path instead of manual CLI steps, use [`infra/regional-instance/`](../infra/regional-instance/). The manual steps below are still useful for debugging or one-off experiments.

These steps are repeated for each EC2 instance you launch.

### 1. Launch an EC2 GPU instance

**AWS Console:**

1. Go to **EC2 > Launch Instance**
2. Name: `openpi-inference`
3. AMI: **Ubuntu 24.04** (64-bit x86)
4. Instance type: **g6e.xlarge** (1x L40S GPU, 4 vCPUs, 32 GB RAM)
5. Key pair: select or create one for SSH access
6. Network: create/select a security group allowing:
   - SSH (TCP 22) from your IP
   - TCP 8000 for WebSocket inference
   - UDP 5555 for QUIC transport
   - TCP 443 if using HTTPS (Caddy or ALB)
7. Storage: **100 GiB** gp3 root volume
8. Advanced > IAM instance profile: **ec2-ecr-pull**
9. Launch

**AWS CLI:**

```bash
# Find the latest Ubuntu 24.04 AMI
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
  --iam-instance-profile Name=ec2-ecr-pull \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=openpi-inference}]'
```

### 2. Install Docker + NVIDIA runtime

```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Log out and back in for docker group
exit
```

### 3. Pull and run

```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com"

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Pull
docker pull "${ECR_REGISTRY}/openpi-flash:latest"

# Create config
mkdir -p ~/openpi && cd ~/openpi
cat > config.json << 'EOF'
{
  "model_config_name": "pi05_aloha",
  "checkpoint_dir": "s3://openpi-checkpoints-us-west-2/pi05_base_pytorch",
  "port": 8000,
  "max_concurrent_requests": 1
}
EOF

# Run
docker run -d --restart unless-stopped --gpus=all \
  -v $(pwd)/config.json:/config/config.json:ro \
  -e INFERENCE_CONFIG_PATH=/config/config.json \
  -p 8000:8000 -p 5555:5555/udp \
  --name openpi-inference \
  "${ECR_REGISTRY}/openpi-flash:latest"
```

### 4. Optional: Elastic IP

Assign a static IP so the address doesn't change across instance stop/start:

```bash
# Allocate
aws ec2 allocate-address --domain vpc --region <instance-region>

# Associate (use the allocation ID from above)
aws ec2 associate-address \
  --instance-id <instance-id> \
  --allocation-id <eipalloc-xxx> \
  --region <instance-region>
```

### 5. Optional: HTTPS with Caddy

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

Caddy auto-provisions TLS via Let's Encrypt. Point your domain's DNS A record to the instance IP.
