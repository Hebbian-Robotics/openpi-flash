variable "aws_region" {
  description = "AWS region for shared infrastructure (ECR, S3, IAM)"
  type        = string
  default     = "us-west-2"
}

variable "aws_profile" {
  description = "AWS CLI profile to use (set to null for default/env credentials)"
  type        = string
  default     = null
}

variable "github_org" {
  description = "GitHub organization that owns the hosting repo"
  type        = string
  default     = "Hebbian-Robotics"
}

variable "github_repo" {
  description = "GitHub repository name for CI/CD access"
  type        = string
  default     = "openpi-hosting"
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository for Docker images"
  type        = string
  default     = "openpi-hosted"
}

variable "checkpoint_bucket_name" {
  description = "S3 bucket name for PyTorch model checkpoints"
  type        = string
  default     = "openpi-checkpoints-us-west-2"
}

variable "ecr_max_untagged_images" {
  description = "Number of recent untagged images to keep (in addition to 'latest')"
  type        = number
  default     = 3
}
