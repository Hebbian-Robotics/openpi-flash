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
  description = "Subnet ID for the EC2 instance. When null, discovers the default VPC's first public subnet."
  type        = string
  default     = null
}

variable "availability_zone" {
  description = "Preferred availability zone. When set, subnet auto-discovery filters to this AZ. Ignored when subnet_id is provided."
  type        = string
  default     = null
}

variable "iam_instance_profile_name" {
  description = "Instance profile name for the EC2 host"
  type        = string
  default     = "ec2-ecr-pull"
}

variable "ami_id" {
  description = "Explicit AMI ID. When null, the latest Deep Learning AMI GPU PyTorch (Ubuntu 24.04) is resolved via SSM."
  type        = string
  default     = null
}

variable "dlami_ssm_slug" {
  description = "SSM parameter slug under /aws/service/deeplearning/ami/x86_64/ used to resolve the default AMI."
  type        = string
  default     = "oss-nvidia-driver-gpu-pytorch-2.10-ubuntu-24.04"
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
  description = "Optional EC2 key pair name. Mutually exclusive with ssh_public_key."
  type        = string
  default     = null
}

variable "ssh_public_key" {
  description = "Optional SSH public key material. When set, Terraform creates a managed key pair instead of using ssh_key_name."
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
  default     = ["0.0.0.0/0"]
}

variable "allowed_websocket_cidr_blocks" {
  description = "CIDR blocks allowed to reach TCP 8000"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_quic_cidr_blocks" {
  description = "CIDR blocks allowed to reach UDP 5555"
  type        = list(string)
  default     = ["0.0.0.0/0"]
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

variable "openpi_pytorch_compile_mode" {
  description = "Value for OPENPI_PYTORCH_COMPILE_MODE inside the inference container"
  type        = string
  default     = "default"
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
