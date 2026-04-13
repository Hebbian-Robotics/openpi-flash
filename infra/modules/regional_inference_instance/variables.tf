variable "deployment_name" {
  description = "Human-readable name for the regional inference deployment"
  type        = string
}

variable "subnet_id" {
  description = "Subnet to launch the inference instance into. When null, uses the default VPC's first available subnet."
  type        = string
  default     = null
}

variable "iam_instance_profile_name" {
  description = "Instance profile name for the EC2 host"
  type        = string
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
  description = "EC2 instance type for inference"
  type        = string
  default     = "g6e.xlarge"
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size in GiB"
  type        = number
  default     = 100
}

variable "ssh_key_name" {
  description = "Optional EC2 key pair name for SSH access. Mutually exclusive with ssh_public_key."
  type        = string
  default     = null
}

variable "ssh_public_key" {
  description = "Optional SSH public key material. When set, Terraform creates a managed key pair instead of using ssh_key_name."
  type        = string
  default     = null
}

variable "associate_public_ip_address" {
  description = "Whether to request a public IP on the primary network interface"
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
  description = "CIDR blocks allowed to reach the WebSocket and health endpoint on TCP 8000"
  type        = list(string)
  default     = []
}

variable "allowed_quic_cidr_blocks" {
  description = "CIDR blocks allowed to reach the QUIC endpoint on UDP 5555"
  type        = list(string)
  default     = []
}

variable "allowed_https_cidr_blocks" {
  description = "CIDR blocks allowed to reach HTTPS on TCP 443"
  type        = list(string)
  default     = []
}

variable "ecr_repository_url" {
  description = "Repository URL for the OpenPI inference image"
  type        = string
}

variable "ecr_region" {
  description = "AWS region that hosts the ECR repository"
  type        = string
}

variable "docker_image_tag" {
  description = "Image tag to deploy from ECR"
  type        = string
  default     = "latest"
}

variable "cloudwatch_log_group_name" {
  description = "CloudWatch Logs group for container stdout and stderr"
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
  description = "Checkpoint directory for the model, typically an s3:// path"
  type        = string
}

variable "openpi_pytorch_compile_mode" {
  description = "Value for OPENPI_PYTORCH_COMPILE_MODE inside the inference container"
  type        = string
  default     = "default"
}

variable "model_version" {
  description = "Model version string written into the runtime config"
  type        = string
  default     = "pi05_v1"
}

variable "default_prompt" {
  description = "Optional default prompt injected by the server"
  type        = string
  default     = null
}

variable "max_concurrent_requests" {
  description = "Maximum concurrent requests allowed by the service config"
  type        = number
  default     = 1
}

variable "extra_bootstrap_commands" {
  description = "Optional extra shell commands appended to the cloud-init bootstrap"
  type        = string
  default     = ""
}

variable "user_data_replace_on_change" {
  description = "Whether user_data changes should force instance replacement"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags to apply to the regional resources"
  type        = map(string)
  default     = {}
}
