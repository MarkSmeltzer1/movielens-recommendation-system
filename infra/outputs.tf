output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.ml_server.public_ip
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "data_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.ml_server.public_ip}"
}
