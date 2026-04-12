# ECR repository for the inference Docker image.
# Images are pushed by CI on merge to main, pulled by EC2 instances.

resource "aws_ecr_repository" "inference" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = false
  }
}

# Keep 'latest' forever, retain only the N most recent other images.
resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only ${var.ecr_max_untagged_images} most recent untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = var.ecr_max_untagged_images
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
