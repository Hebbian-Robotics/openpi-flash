variable "deployment_name" {
  description = "Human-readable name for the regional inference deployment"
  type        = string
}

variable "subnet_id" {
  description = "Subnet to launch the inference instance into. When null, uses the default VPC's first available subnet."
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
  default     = 200
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
  default     = true
}

variable "allowed_ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH to the instance"
  type        = list(string)
  default     = []
}

variable "allowed_websocket_cidr_blocks" {
  description = "CIDR blocks allowed to reach the action-slot WebSocket/health endpoint on TCP 8000 (and the planner-slot WebSocket on TCP 8002 when planner is enabled)"
  type        = list(string)
  default     = []
}

variable "allowed_quic_cidr_blocks" {
  description = "CIDR blocks allowed to reach the action-slot QUIC endpoint on UDP 5555 (and the planner-slot QUIC endpoint on UDP 5556 when planner is enabled)"
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

variable "container_name" {
  description = "Docker container name for the inference service"
  type        = string
  default     = "openpi-inference"
}

# -- Component slots -----------------------------------------------------------
#
# The Python server boots in one of three modes derived from which slots are
# set:
#
#   action != null, planner == null  → action_only  (current production default)
#   action == null, planner != null  → planner_only (JAX subtask planner only)
#   action != null, planner != null  → combined     (two-phase, both endpoints)
#
# At least one must be non-null; the Python config will fail validation at
# startup if both are null.

variable "action" {
  description = "Action slot config. Set to null to deploy a planner-only server. When set, the server exposes TCP 8000 (WebSocket) + UDP 5555 (QUIC)."
  type = object({
    model_config_name = string
    checkpoint_dir    = string
    default_prompt    = optional(string)
  })
  default = null
}

variable "planner" {
  description = "JAX subtask planner slot config. Set to null to deploy an action-only server. When set, the server exposes TCP 8002 (WebSocket) + UDP 5556 (QUIC) + TCP 8001 (admin HTTP, bound to 127.0.0.1)."
  type = object({
    checkpoint_dir           = string
    max_generation_tokens    = optional(number)
    generation_prompt_format = optional(string)
    action_prompt_template   = optional(string)
  })
  default = null
}

# -- Checkpoint prep (action slot only) ---------------------------------------
#
# Optional one-shot step on first boot that prepares a Hugging Face checkpoint
# into an OpenPI-compatible layout under /cache/models. Only relevant when the
# action slot uses a locally prepared checkpoint path — planner checkpoints are
# pulled directly from gs:// at SubtaskGenerator.load() time and don't need
# prep. Disable with prepare_checkpoint = false for planner-only deployments.

variable "prepare_checkpoint" {
  description = "Run the one-shot Docker checkpoint preparation step on first boot. Action-slot-only; safe to disable for planner-only deployments."
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
  description = "JAX GPU memory fraction (0.0-1.0). Set to 0.5 when both JAX and PyTorch are co-resident (combined mode). Empty string disables the override."
  type        = string
  default     = ""
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
