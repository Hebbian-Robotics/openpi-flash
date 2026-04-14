output "ecr_repository_url" {
  description = "ECR repository URL for Docker images"
  value       = aws_ecr_repository.inference.repository_url
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions CI (use in workflow)"
  value       = aws_iam_role.github_actions_ecr_push.arn
}

output "ec2_instance_profile_name" {
  description = "Instance profile name to attach to EC2 inference instances"
  value       = aws_iam_instance_profile.ec2_inference.name
}
