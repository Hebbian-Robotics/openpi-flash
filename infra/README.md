# AWS Infrastructure

Terraform configs for the shared AWS infrastructure that supports openpi-hosting. These resources are created once and shared across all EC2 inference instances.

## What's managed

| Resource | Purpose |
|----------|---------|
| ECR repository | Docker images built by CI, pulled by EC2 |
| ECR lifecycle policy | Auto-cleanup: keeps `latest` + 3 most recent images |
| IAM OIDC provider | GitHub Actions federates into AWS without static creds |
| IAM role `github-actions-ecr-push` | CI pushes images to ECR |
| IAM role `ec2-ecr-pull` + instance profile | EC2 pulls images from ECR, checkpoints from S3, writes CloudWatch logs |
| S3 bucket | Stores pre-converted PyTorch model checkpoints |

## What's NOT managed

EC2 instances, security groups, Elastic IPs, and ALBs are not in Terraform — they vary per deployment region and can be created via the console or CLI. See [docs/aws-manual-setup.md](../docs/aws-manual-setup.md) for instance setup instructions.

## Usage

### Prerequisites

Install [Terraform](https://developer.hashicorp.com/terraform/install) or [OpenTofu](https://opentofu.org/docs/intro/install/) (configs are compatible with both).

### First-time setup

```bash
cd infra

# Initialize providers
terraform init

# Review what will be created
terraform plan

# Create the resources
terraform apply
```

### Importing existing resources

If the resources already exist (created manually), import them into Terraform state:

```bash
terraform import aws_ecr_repository.inference openpi-hosted
terraform import aws_ecr_lifecycle_policy.inference openpi-hosted
terraform import aws_iam_openid_connect_provider.github_actions arn:aws:iam::438136598620:oidc-provider/token.actions.githubusercontent.com
terraform import aws_iam_role.github_actions_ecr_push github-actions-ecr-push
terraform import aws_iam_role.ec2_inference ec2-ecr-pull
terraform import aws_iam_instance_profile.ec2_inference ec2-ecr-pull
terraform import aws_s3_bucket.checkpoints openpi-checkpoints-us-west-2
terraform import aws_s3_bucket_versioning.checkpoints openpi-checkpoints-us-west-2
terraform import aws_s3_bucket_server_side_encryption_configuration.checkpoints openpi-checkpoints-us-west-2
terraform import aws_s3_bucket_public_access_block.checkpoints openpi-checkpoints-us-west-2
```

After importing, run `terraform plan` to verify no unexpected changes.

### Customizing

Override defaults in a `terraform.tfvars` file:

```hcl
aws_region             = "us-west-2"
aws_profile            = "squash"
github_org             = "Hebbian-Robotics"
github_repo            = "openpi-hosting"
checkpoint_bucket_name = "openpi-checkpoints-us-west-2"
```

### Outputs

After applying, Terraform prints values needed by CI and EC2 setup:

```bash
terraform output
# ecr_repository_url        = "438136598620.dkr.ecr.us-west-2.amazonaws.com/openpi-hosted"
# checkpoint_bucket_name    = "openpi-checkpoints-us-west-2"
# github_actions_role_arn   = "arn:aws:iam::438136598620:role/github-actions-ecr-push"
# ec2_instance_profile_name = "ec2-ecr-pull"
```

Use `github_actions_role_arn` in `.github/workflows/docker-build.yml` and `ec2_instance_profile_name` when launching instances.
