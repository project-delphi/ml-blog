terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "5.17.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"
}
resource "aws_apprunner_auto_scaling_configuration_version" "hello" {
  auto_scaling_configuration_name = "hello"
  # scale between 1-5 containers
  min_size = 1
  max_size = 1
}

resource "aws_apprunner_service" "hello" {
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.hello.arn
  service_name = "hello-app-runner"

  source_configuration {
    image_repository {
      image_configuration {
        port = "8000" #The port that your application listens to in the container
      }

      image_identifier = "public.ecr.aws/aws-containers/hello-app-runner:latest"
      image_repository_type = "ECR_PUBLIC"
    }
    auto_deployments_enabled = false

  }
}
output "apprunner_service_hello" {
  value = aws_apprunner_service.hello
}
