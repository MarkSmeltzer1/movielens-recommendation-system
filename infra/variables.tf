variable "aws_region" {
  description = "AWS Region"
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 Instance Type"
  default     = "c7i-flex.large"
}

variable "key_name" {
  description = "Name of existing AWS Key Pair for SSH access"
  type        = string
}

variable "neon_dsn" {
  description = "Neon Connection String (postgresql://user:pass@host/db)"
  type        = string
  sensitive   = true # Hides it from CLI output
}
