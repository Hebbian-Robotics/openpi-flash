variable "aws_region" {
  description = "AWS region to deploy the regional inference instance into"
  type        = string
}

variable "aws_profile" {
  description = "AWS CLI profile to use for the regional deployment"
  type        = string
  default     = null
}

variable "deployment_name" {
  description = "Human-readable name for the regional deployment"
  type        = string
  default     = "openpi-inference"
}

variable "subnet_id" {
  description = "Subnet ID for the EC2 instance"
  type        = string
}

variable "iam_instance_profile_name" {
  description = "Instance profile name for the EC2 host"
  type        = string
  default     = "ec2-ecr-pull"
}

variable "ami_id" {
  description = "Optional explicit AMI ID; defaults to latest Canonical Ubuntu 24.04 x86_64 in the selected region"
  type        = string
  default     = null
}

variable "instance_type" {
  description = "EC2 instance type to launch"
  type        = string
  default     = "g6e.xlarge"
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size in GiB"
  type        = number
  default     = 100
}

variable "ssh_key_name" {
  description = "Optional EC2 key pair name"
  type        = string
  default     = null
}

variable "associate_public_ip_address" {
  description = "Whether to associate a public IP address on launch"
  type        = bool
  default     = true
}

variable "assign_elastic_ip" {
  description = "Whether to allocate and associate an Elastic IP"
  type        = bool
  default     = false
}

variable "allowed_ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH to the instance"
  type        = list(string)
  default     = []
}

variable "allowed_websocket_cidr_blocks" {
  description = "CIDR blocks allowed to reach TCP 8000"
  type        = list(string)
  default     = []
}

variable "allowed_quic_cidr_blocks" {
  description = "CIDR blocks allowed to reach UDP 5555"
  type        = list(string)
  default     = []
}

variable "allowed_https_cidr_blocks" {
  description = "CIDR blocks allowed to reach TCP 443"
  type        = list(string)
  default     = []
}

variable "ecr_repository_url" {
  description = "ECR repository URL to deploy from, typically taken from the shared infra output"
  type        = string
}

variable "ecr_region" {
  description = "Region that hosts the ECR repository"
  type        = string
  default     = "us-west-2"
}

variable "docker_image_tag" {
  description = "ECR image tag to pull"
  type        = string
  default     = "latest"
}

variable "cloudwatch_log_group_name" {
  description = "CloudWatch Logs group for container logs"
  type        = string
  default     = "/openpi/inference"
}

variable "container_name" {
  description = "Docker container name for the inference service"
  type        = string
  default     = "openpi-inference"
}

variable "model_config_name" {
  description = "OpenPI model config name to serve"
  type        = string
  default     = "pi05_aloha"
}

variable "checkpoint_dir" {
  description = "Checkpoint directory for the model"
  type        = string
  default     = "s3://openpi-checkpoints-us-west-2/pi05_base_pytorch"
}

variable "model_version" {
  description = "Model version written into the runtime config"
  type        = string
  default     = "pi05_v1"
}

variable "default_prompt" {
  description = "Optional default prompt for the server runtime config"
  type        = string
  default     = null
}

variable "max_concurrent_requests" {
  description = "Maximum concurrent requests allowed by the server"
  type        = number
  default     = 1
}

variable "extra_bootstrap_commands" {
  description = "Optional extra shell commands appended to instance bootstrap"
  type        = string
  default     = ""
}

variable "user_data_replace_on_change" {
  description = "Whether user_data changes should replace the instance"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags applied to regional resources"
  type        = map(string)
  default     = {}
}
