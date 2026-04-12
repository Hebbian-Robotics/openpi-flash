data "aws_region" "current" {}

data "aws_subnet" "selected" {
  id = var.subnet_id
}

data "aws_ami" "ubuntu_noble" {
  count       = var.ami_id == null ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  effective_ami_id = var.ami_id != null ? var.ami_id : data.aws_ami.ubuntu_noble[0].id

  effective_tags = merge(
    var.tags,
    {
      Name = var.deployment_name
    },
  )

  effective_service_config_json = jsonencode(
    {
      model_config_name       = var.model_config_name
      checkpoint_dir          = var.checkpoint_dir
      model_version           = var.model_version
      default_prompt          = var.default_prompt
      port                    = 8000
      max_concurrent_requests = var.max_concurrent_requests
    }
  )

  ecr_registry_host   = split("/", var.ecr_repository_url)[0]
  effective_image_url = "${var.ecr_repository_url}:${var.docker_image_tag}"
  effective_public_ip = var.assign_elastic_ip ? aws_eip.inference[0].public_ip : aws_instance.inference.public_ip
}

resource "aws_security_group" "inference" {
  name_prefix = "${var.deployment_name}-"
  description = "Security group for ${var.deployment_name} inference instance"
  vpc_id      = data.aws_subnet.selected.vpc_id

  tags = local.effective_tags
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

resource "aws_vpc_security_group_ingress_rule" "websocket" {
  for_each = toset(var.allowed_websocket_cidr_blocks)

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 8000
  to_port           = 8000
  ip_protocol       = "tcp"
  description       = "OpenPI WebSocket and health access"
}

resource "aws_vpc_security_group_ingress_rule" "quic" {
  for_each = toset(var.allowed_quic_cidr_blocks)

  security_group_id = aws_security_group.inference.id
  cidr_ipv4         = each.value
  from_port         = 5555
  to_port           = 5555
  ip_protocol       = "udp"
  description       = "OpenPI QUIC access"
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
  subnet_id                   = var.subnet_id
  vpc_security_group_ids      = [aws_security_group.inference.id]
  iam_instance_profile        = var.iam_instance_profile_name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = var.associate_public_ip_address
  user_data_replace_on_change = var.user_data_replace_on_change

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
    "${path.module}/templates/user_data.sh.tftpl",
    {
      cloudwatch_log_group_name = var.cloudwatch_log_group_name
      config_json               = local.effective_service_config_json
      container_name            = var.container_name
      ecr_region                = var.ecr_region
      ecr_registry_host         = local.ecr_registry_host
      extra_bootstrap_commands  = var.extra_bootstrap_commands
      image_url                 = local.effective_image_url
      log_region                = data.aws_region.current.name
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
