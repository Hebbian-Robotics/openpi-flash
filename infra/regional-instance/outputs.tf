output "instance_id" {
  description = "Instance ID for the deployed regional inference host"
  value       = module.regional_inference_instance.instance_id
}

output "public_ip" {
  description = "Public IP for the deployed inference host"
  value       = module.regional_inference_instance.public_ip
}

output "public_dns" {
  description = "Public DNS for the deployed inference host"
  value       = module.regional_inference_instance.public_dns
}

output "private_ip" {
  description = "Private IP for the deployed inference host"
  value       = module.regional_inference_instance.private_ip
}

output "security_group_id" {
  description = "Security group attached to the deployed inference host"
  value       = module.regional_inference_instance.security_group_id
}

output "elastic_ip_allocation_id" {
  description = "Elastic IP allocation ID when enabled"
  value       = module.regional_inference_instance.elastic_ip_allocation_id
}
