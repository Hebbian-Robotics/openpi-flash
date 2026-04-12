output "instance_id" {
  description = "Instance ID for the regional inference host"
  value       = aws_instance.inference.id
}

output "public_ip" {
  description = "Public IP for the inference host"
  value       = local.effective_public_ip
}

output "public_dns" {
  description = "Public DNS name for the inference host"
  value       = aws_instance.inference.public_dns
}

output "private_ip" {
  description = "Private IP for the inference host"
  value       = aws_instance.inference.private_ip
}

output "security_group_id" {
  description = "Security group ID attached to the inference host"
  value       = aws_security_group.inference.id
}

output "elastic_ip_allocation_id" {
  description = "Elastic IP allocation ID when one is attached"
  value       = var.assign_elastic_ip ? aws_eip.inference[0].allocation_id : null
}

output "ssh_key_pair_name" {
  description = "Name of the SSH key pair (managed or external)"
  value       = var.ssh_public_key != null ? aws_key_pair.inference[0].key_name : var.ssh_key_name
}

output "effective_subnet_id" {
  description = "Subnet ID used by the instance (explicit or auto-discovered)"
  value       = local.effective_subnet_id
}
