# AWS Infrastructure

Terraform configs for the shared AWS infrastructure that supports openpi-flash. These resources are created once and shared across all EC2 inference instances.

For region-specific EC2 deployments, use [`regional-instance/`](./regional-instance/). Keep the shared root and the regional EC2 root separate so you can deploy the same server shape into different regions without duplicating ECR or IAM resources.

## What's managed

| Resource | Purpose |
|----------|---------|
| ECR repository | Docker images built by CI, pulled by EC2 |
| ECR lifecycle policy | Auto-cleanup: keeps `latest` + 3 most recent images |
| IAM OIDC provider | GitHub Actions federates into AWS without static creds |
| IAM role `github-actions-ecr-push` | CI pushes images to ECR |
| IAM role `ec2-ecr-pull` + instance profile | EC2 pulls images from ECR |

## What's NOT managed

EC2 instances, security groups, Elastic IPs, and ALBs are not managed by this shared root. They vary per deployment region and should be created either with [`regional-instance/`](./regional-instance/) or manually via [docs/aws-manual-setup.md](../docs/aws-manual-setup.md).

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
terraform import aws_ecr_repository.inference openpi-flash
terraform import aws_ecr_lifecycle_policy.inference openpi-flash
terraform import aws_iam_openid_connect_provider.github_actions arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com
terraform import aws_iam_role.github_actions_ecr_push github-actions-ecr-push
terraform import aws_iam_role.ec2_inference ec2-ecr-pull
terraform import aws_iam_role_policy_attachment.ec2_ecr_pull_read_only ec2-ecr-pull/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
terraform import aws_iam_instance_profile.ec2_inference ec2-ecr-pull
```

After importing, run `terraform plan` to verify no unexpected changes.

### Customizing

Override defaults in a `terraform.tfvars` file:

```hcl
aws_region             = "us-west-2"
aws_profile            = "your-profile"
github_org             = "Hebbian-Robotics"
github_repo            = "openpi-flash"
```

### Outputs

After applying, Terraform prints values needed by CI and EC2 setup:

```bash
terraform output
# ecr_repository_url        = "<account-id>.dkr.ecr.us-west-2.amazonaws.com/openpi-flash"
# github_actions_role_arn   = "arn:aws:iam::<account-id>:role/github-actions-ecr-push"
# ec2_instance_profile_name = "ec2-ecr-pull"
```

Use `github_actions_role_arn` in `.github/workflows/docker-build.yml` and `ec2_instance_profile_name` when launching instances.

### GitHub Actions variables

The Docker build workflow expects two GitHub Actions repository variables:

```bash
AWS_ECR_REGISTRY=<account-id>.dkr.ecr.us-west-2.amazonaws.com
AWS_ROLE_TO_ASSUME=arn:aws:iam::<account-id>:role/github-actions-ecr-push
```

Set them from the Terraform outputs and your account ID:

```bash
ECR_REPOSITORY_URL=$(terraform output -raw ecr_repository_url)
GITHUB_ACTIONS_ROLE_ARN=$(terraform output -raw github_actions_role_arn)

gh variable set AWS_ECR_REGISTRY \
  --body "${ECR_REPOSITORY_URL%/openpi-flash}" \
  --repo Hebbian-Robotics/openpi-flash

gh variable set AWS_ROLE_TO_ASSUME \
  --body "$GITHUB_ACTIONS_ROLE_ARN" \
  --repo Hebbian-Robotics/openpi-flash
```
