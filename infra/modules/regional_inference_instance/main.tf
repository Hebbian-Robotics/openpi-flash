terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# -- Default VPC/subnet discovery (used when subnet_id is not provided) --
data "aws_vpc" "default" {
  count   = var.subnet_id == null ? 1 : 0
  default = true
}

data "aws_subnets" "default_vpc" {
  count = var.subnet_id == null ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }

  dynamic "filter" {
    for_each = var.availability_zone != null ? [var.availability_zone] : []
    content {
      name   = "availability-zone"
      values = [filter.value]
    }
  }
}

data "aws_subnet" "selected" {
  id = local.effective_subnet_id
}

# Resolve the latest Deep Learning AMI GPU PyTorch (Ubuntu 24.04) via SSM.
# This AMI ships with Docker, nvidia-container-toolkit, CUDA, and AWS CLI
# pre-installed, so the bootstrap script only needs to write config and start
# the inference service.
data "aws_ssm_parameter" "dlami_pytorch" {
  count = var.ami_id == null ? 1 : 0
  name  = "/aws/service/deeplearning/ami/x86_64/${var.dlami_ssm_slug}/latest/ami-id"
}

locals {
  effective_subnet_id = var.subnet_id != null ? var.subnet_id : data.aws_subnets.default_vpc[0].ids[0]
  effective_ami_id    = var.ami_id != null ? var.ami_id : nonsensitive(data.aws_ssm_parameter.dlami_pytorch[0].value)

  effective_tags = merge(
    var.tags,
    {
      Name = var.deployment_name
    },
  )

  # Derived mode flags — mirror the Python server's _resolve_mode().
  action_enabled  = var.action != null
  planner_enabled = var.planner != null

  # Service config JSON mirrors src/hosting/config.py:ServiceConfig. Pydantic
  # validates the nested schema at process startup — only pass through what's
  # set so defaults in the Python layer apply.
  effective_service_config_json = jsonencode(merge(
    local.action_enabled ? {
      action = merge(
        {
          model_config_name = var.action.model_config_name
          checkpoint_dir    = var.action.checkpoint_dir
        },
        var.action.default_prompt == null ? {} : {
          default_prompt = var.action.default_prompt
        },
      )
    } : {},
    local.planner_enabled ? {
      planner = merge(
        {
          checkpoint_dir = var.planner.checkpoint_dir
        },
        var.planner.max_generation_tokens == null ? {} : {
          max_generation_tokens = var.planner.max_generation_tokens
        },
        var.planner.generation_prompt_format == null ? {} : {
          generation_prompt_format = var.planner.generation_prompt_format
        },
        var.planner.action_prompt_template == null ? {} : {
          action_prompt_template = var.planner.action_prompt_template
        },
      )
    } : {},
  ))

  ecr_registry_host   = split("/", var.ecr_repository_url)[0]
  effective_image_url = "${var.ecr_repository_url}:${var.docker_image_tag}"
  effective_public_ip = var.assign_elastic_ip ? aws_eip.inference[0].public_ip : aws_instance.inference.public_ip
}

resource "aws_security_group" "inference" {
  name_prefix = "${var.deployment_name}-"
  description = "Security group for ${var.deployment_name} inference instance"
  vpc_id      = data.aws_subnet.selected.vpc_id

  tags = local.effective_tags

  lifecycle {
    precondition {
      condition     = local.action_enabled || local.planner_enabled
      error_message = "At least one of var.action or var.planner must be non-null. The Python server will refuse to boot otherwise."
    }
  }
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each = toset(var.allowed_ssh_cidr_blocks)

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
  description       = "SSH access"
}

# -- Action slot: TCP 8000 (WebSocket) + UDP 5555 (QUIC) --

resource "aws_vpc_security_group_ingress_rule" "action_websocket" {
  for_each = local.action_enabled ? toset(var.allowed_websocket_cidr_blocks) : toset([])

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 8000
  to_port           = 8000
  ip_protocol       = "tcp"
  description       = "OpenPI action-slot WebSocket/health"
}

resource "aws_vpc_security_group_ingress_rule" "action_quic" {
  for_each = local.action_enabled ? toset(var.allowed_quic_cidr_blocks) : toset([])

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 5555
  to_port           = 5555
  ip_protocol       = "udp"
  description       = "OpenPI action-slot QUIC"
}

# -- Planner slot: TCP 8002 (WebSocket) + UDP 5556 (QUIC) --

resource "aws_vpc_security_group_ingress_rule" "planner_websocket" {
  for_each = local.planner_enabled ? toset(var.allowed_websocket_cidr_blocks) : toset([])

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 8002
  to_port           = 8002
  ip_protocol       = "tcp"
  description       = "OpenPI planner-slot WebSocket"
}

resource "aws_vpc_security_group_ingress_rule" "planner_quic" {
  for_each = local.planner_enabled ? toset(var.allowed_quic_cidr_blocks) : toset([])

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 5556
  to_port           = 5556
  ip_protocol       = "udp"
  description       = "OpenPI planner-slot QUIC"
}

resource "aws_vpc_security_group_ingress_rule" "https" {
  for_each = toset(var.allowed_https_cidr_blocks)

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  description       = "HTTPS access"
}

resource "aws_vpc_security_group_egress_rule" "all_outbound" {
  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "Allow all outbound traffic"
}

resource "aws_instance" "inference" {
  ami                         = local.effective_ami_id
  instance_type               = var.instance_type
  subnet_id                   = local.effective_subnet_id
  vpc_security_group_ids      = [aws_security_group.inference.id]
  iam_instance_profile        = var.iam_instance_profile_name
  key_name                    = var.ssh_public_key != null ? aws_key_pair.inference[0].key_name : var.ssh_key_name
  associate_public_ip_address = var.associate_public_ip_address
  user_data_replace_on_change = var.user_data_replace_on_change

  lifecycle {
    precondition {
      condition     = !(var.ssh_key_name != null && var.ssh_public_key != null)
      error_message = "Only one of ssh_key_name or ssh_public_key may be set, not both."
    }
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.root_volume_size_gib
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = templatefile(
    "${path.module}/templates/user_data.yaml.tftpl",
    {
      config_json                       = local.effective_service_config_json
      checkpoint_prep_model_id          = var.checkpoint_prep_model_id
      checkpoint_prep_openpi_assets_uri = var.checkpoint_prep_openpi_assets_uri
      checkpoint_prep_output_dir        = var.checkpoint_prep_output_dir
      container_name                    = var.container_name
      ecr_region                        = var.ecr_region
      ecr_registry_host                 = local.ecr_registry_host
      extra_bootstrap_commands          = var.extra_bootstrap_commands
      image_url                         = local.effective_image_url
      openpi_pytorch_compile_mode       = var.openpi_pytorch_compile_mode
      prepare_checkpoint                = var.prepare_checkpoint
      prepare_planner_checkpoint        = var.prepare_planner_checkpoint
      planner_prep_hf_repo              = var.planner_prep_hf_repo
      planner_prep_tar_path_in_repo     = var.planner_prep_tar_path_in_repo
      planner_prep_output_dir           = var.planner_prep_output_dir
      xla_python_client_mem_fraction    = var.xla_python_client_mem_fraction
      action_enabled                    = local.action_enabled
      planner_enabled                   = local.planner_enabled
    }
  )

  tags = local.effective_tags
}

resource "aws_eip" "inference" {
  count = var.assign_elastic_ip ? 1 : 0

  instance = aws_instance.inference.id
  domain   = "vpc"

  tags = local.effective_tags
}

resource "aws_key_pair" "inference" {
  count = var.ssh_public_key != null ? 1 : 0

  key_name   = "${var.deployment_name}-ssh"
  public_key = var.ssh_public_key

  tags = local.effective_tags
}
