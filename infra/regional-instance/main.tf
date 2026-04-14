terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "openpi"
      ManagedBy = "terraform"
    }
  }
}

module "regional_inference_instance" {
  source = "../modules/regional_inference_instance"

  allowed_https_cidr_blocks     = var.allowed_https_cidr_blocks
  allowed_quic_cidr_blocks      = var.allowed_quic_cidr_blocks
  allowed_ssh_cidr_blocks       = var.allowed_ssh_cidr_blocks
  allowed_websocket_cidr_blocks = var.allowed_websocket_cidr_blocks
  availability_zone             = var.availability_zone
  ami_id                        = var.ami_id
  dlami_ssm_slug                = var.dlami_ssm_slug
  associate_public_ip_address   = var.associate_public_ip_address
  assign_elastic_ip             = var.assign_elastic_ip
  checkpoint_dir                = var.checkpoint_dir
  container_name                = var.container_name
  default_prompt                = var.default_prompt
  deployment_name               = var.deployment_name
  docker_image_tag              = var.docker_image_tag
  ecr_region                    = var.ecr_region
  ecr_repository_url            = var.ecr_repository_url
  extra_bootstrap_commands      = var.extra_bootstrap_commands
  iam_instance_profile_name     = var.iam_instance_profile_name
  instance_type                 = var.instance_type
  model_config_name             = var.model_config_name
  openpi_pytorch_compile_mode   = var.openpi_pytorch_compile_mode
  root_volume_size_gib          = var.root_volume_size_gib
  ssh_key_name                  = var.ssh_key_name
  ssh_public_key                = var.ssh_public_key
  subnet_id                     = var.subnet_id
  tags                          = var.tags
  user_data_replace_on_change   = var.user_data_replace_on_change
}
