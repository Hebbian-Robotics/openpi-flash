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
  default     = 200
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
  default     = true
}

variable "allowed_ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH to the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_websocket_cidr_blocks" {
  description = "CIDR blocks allowed to reach TCP 8000 (action) and TCP 8002 (planner, when enabled)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_quic_cidr_blocks" {
  description = "CIDR blocks allowed to reach UDP 5555 (action) and UDP 5556 (planner, when enabled)"
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

# -- Component slots -----------------------------------------------------------
#
# See infra/modules/regional_inference_instance/variables.tf for the full
# explanation of the three server modes.

variable "action" {
  description = "Action slot config. Set to null to deploy a planner-only server."
  type = object({
    model_config_name = string
    checkpoint_dir    = string
    default_prompt    = optional(string)
  })
  default = {
    model_config_name = "pi05_aloha"
    checkpoint_dir    = "/cache/models/pi05_base_openpi"
  }
}

variable "planner" {
  description = "JAX subtask planner slot config. Set to null to deploy an action-only server."
  type = object({
    checkpoint_dir           = string
    max_generation_tokens    = optional(number)
    generation_prompt_format = optional(string)
    action_prompt_template   = optional(string)
  })
  default = null
}

# -- Checkpoint prep (action slot only) ---------------------------------------

variable "prepare_checkpoint" {
  description = "Run the one-shot Docker checkpoint preparation step on first boot. Action-slot-only."
  type        = bool
  default     = true
}

variable "checkpoint_prep_model_id" {
  description = "Hugging Face model ID used by the Docker checkpoint preparation step"
  type        = string
  default     = "lerobot/pi05_base"
}

variable "checkpoint_prep_openpi_assets_uri" {
  description = "OpenPI assets URI containing normalization stats used by the Docker checkpoint preparation step"
  type        = string
  default     = "gs://openpi-assets/checkpoints/pi05_base/assets"
}

variable "checkpoint_prep_output_dir" {
  description = "Output directory written by the Docker checkpoint preparation step"
  type        = string
  default     = "/cache/models/pi05_base_openpi"
}

variable "openpi_pytorch_compile_mode" {
  description = "Value for OPENPI_PYTORCH_COMPILE_MODE inside the inference container"
  type        = string
  default     = "default"
}

variable "xla_python_client_mem_fraction" {
  description = "JAX GPU memory fraction (0.0-1.0). Set to 0.5 for combined mode (JAX+PyTorch co-resident). Empty string disables the override."
  type        = string
  default     = ""
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
