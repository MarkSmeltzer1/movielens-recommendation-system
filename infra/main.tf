terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- 1. S3 Bucket for MLflow Artifacts ---
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "movielens-mlflow-artifacts-${random_id.bucket_suffix.hex}"
  force_destroy = true # Allow destruction even if not empty (for dev/testing)

  tags = {
    Name        = "MovieLens MLflow Artifacts"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket" "data_bucket" {
  bucket = "movielens-data-${random_id.bucket_suffix.hex}"
  force_destroy = true

  tags = {
    Name        = "MovieLens Data"
    Environment = "Dev"
  }
}

# --- 2. IAM Role for EC2 ---
resource "aws_iam_role" "ec2_mlflow_role" {
  name = "movielens_ec2_mlflow_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Allow EC2 to access S3 Bucket
resource "aws_iam_policy" "s3_access" {
  name        = "MovieLensS3Access"
  description = "Allow access to MLflow artifacts bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          aws_s3_bucket.data_bucket.arn
        ]
      },
      {
        Sid      = "BucketObjects"
        Effect   = "Allow"
        Action   = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_s3" {
  role       = aws_iam_role.ec2_mlflow_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "movielens_ec2_profile"
  role = aws_iam_role.ec2_mlflow_role.name
}

# --- 3. EC2 Instance ---
# Get latest Ubuntu 22.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# Security Group
resource "aws_security_group" "ml_sg" {
  name        = "movielens_ml_sg"
  description = "Allow SSH and MLflow"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # WARNING: Open to world for demo. Restrict in prod.
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "ml_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name
  security_groups      = [aws_security_group.ml_sg.name]
  key_name             = var.key_name 

  tags = {
    Name = "MovieLens-ML-Server"
  }

  root_block_device {
    volume_size = 30 # Default is 8GB (too small for ML + Swap)
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              
              # 0. Create 2GB Swap (Prevents OOM on t3.micro)
              fallocate -l 2G /swapfile
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile
              echo '/swapfile none swap sw 0 0' >> /etc/fstab
              
              # Set Environment Variables for all users
              echo 'export S3_BUCKET_NAME="${aws_s3_bucket.data_bucket.bucket}"' > /etc/profile.d/movielens.sh
              echo 'export MLFLOW_TRACKING_URI="http://localhost:5000"' >> /etc/profile.d/movielens.sh
              chmod 644 /etc/profile.d/movielens.sh
              
              apt-get update
              apt-get install -y python3-pip git default-jre build-essential awscli

              # Setup for 'ubuntu' user
              su - ubuntu -c '
                # 1. Install uv
                curl -LsSf https://astral.sh/uv/install.sh | sh
                source $HOME/.local/bin/env
                
                # 2. Install MLflow globally (with Postgres driver & Gunicorn for prod)
                uv tool install mlflow --with psycopg2-binary --with gunicorn
                
                # 3. Clone Repo (Using HTTPS)
                git clone https://github.com/DjTuner13/movielens-rating-prediction.git
                
                # 4. Sync Dependencies
                cd movielens-rating-prediction
                uv sync
                
                # 5. Download Data (Automated!)
                # Ensure the directory exists
                mkdir -p data/rawData
                # Download from the specific data bucket created by Terraform
                aws s3 cp s3://${aws_s3_bucket.data_bucket.bucket}/merged_movielens.csv data/rawData/merged_movielens.csv
              '
              
              # 6. Create MLflow Systemd Service
              cat <<EOT > /etc/systemd/system/mlflow.service
              [Unit]
              Description=MLflow Tracking Server
              After=network.target

              [Service]
              User=ubuntu
              WorkingDirectory=/home/ubuntu
              # Configured via Terraform Variable
              ExecStart=/home/ubuntu/.local/bin/mlflow server \
                --host 0.0.0.0 \
                --port 5000 \
                --allowed-hosts "*" \
                --backend-store-uri ${var.neon_dsn} \
                --default-artifact-root s3://${aws_s3_bucket.mlflow_artifacts.bucket}
              Restart=always

              [Install]
              WantedBy=multi-user.target
              EOT

              # 6. Start MLflow
              systemctl daemon-reload
              systemctl enable mlflow
              systemctl start mlflow

              # 7. Create FastAPI Systemd Service
              cat <<EOT > /etc/systemd/system/fastapi.service
              [Unit]
              Description=MovieLens Prediction API
              After=network.target mlflow.service

              [Service]
              User=ubuntu
              WorkingDirectory=/home/ubuntu/movielens-rating-prediction
              Environment="MLFLOW_TRACKING_URI=http://localhost:5000"
              # Use the full path to uv
              ExecStart=/home/ubuntu/.local/bin/uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000
              Restart=always
              RestartSec=10

              [Install]
              WantedBy=multi-user.target
              EOT

              # 8. Start FastAPI
              systemctl daemon-reload
              systemctl enable fastapi
              systemctl start fastapi
              EOF
}
