# AWS Infrastructure

OpenTofu configs for the shared AWS infrastructure that supports openpi-flash. These resources are created once and shared across all EC2 inference instances.

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

Install [OpenTofu](https://opentofu.org/docs/intro/install/). The configs are Terraform-compatible HCL, but this repo uses the `tofu` CLI for consistency.

### First-time setup

```bash
cd infra

# Initialize providers
tofu init

# Review what will be created
tofu plan

# Create the resources
tofu apply
```

### Importing existing resources

If the resources already exist (created manually), import them into OpenTofu state:

```bash
tofu import aws_ecr_repository.inference openpi-flash
tofu import aws_ecr_lifecycle_policy.inference openpi-flash
tofu import aws_iam_openid_connect_provider.github_actions arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com
tofu import aws_iam_role.github_actions_ecr_push github-actions-ecr-push
tofu import aws_iam_role.ec2_inference ec2-ecr-pull
tofu import aws_iam_role_policy_attachment.ec2_ecr_pull_read_only ec2-ecr-pull/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
tofu import aws_iam_instance_profile.ec2_inference ec2-ecr-pull
```

After importing, run `tofu plan` to verify no unexpected changes.

### Customizing

Override defaults in a `terraform.tfvars` file:

```hcl
aws_region             = "us-west-2"
aws_profile            = "your-profile"
github_org             = "Hebbian-Robotics"
github_repo            = "openpi-flash"
```

### Outputs

After applying, OpenTofu prints values needed by CI and EC2 setup:

```bash
tofu output
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

Set them from the OpenTofu outputs and your account ID:

```bash
ECR_REPOSITORY_URL=$(tofu output -raw ecr_repository_url)
GITHUB_ACTIONS_ROLE_ARN=$(tofu output -raw github_actions_role_arn)

gh variable set AWS_ECR_REGISTRY \
  --body "${ECR_REPOSITORY_URL%/openpi-flash}" \
  --repo Hebbian-Robotics/openpi-flash

gh variable set AWS_ROLE_TO_ASSUME \
  --body "$GITHUB_ACTIONS_ROLE_ARN" \
  --repo Hebbian-Robotics/openpi-flash
```
