# AWS Infrastructure Deployment

This directory contains Terraform scripts to provision the cloud resources for the MovieLens project.

## Prerequisites
1.  **Terraform**: [Install Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).
2.  **AWS CLI**: [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
3.  **AWS Account**: Ensure you have an AWS account and credentials configured (`aws configure`).
4.  **SSH Key Pair**: You must have an existing EC2 Key Pair in your AWS region (e.g., `us-east-1`).

## 1. Deploy Resources
Initialize and apply the Terraform configuration:
```bash
cd infra
terraform init
terraform apply
```
*   You will be prompted for `key_name` (The name of your AWS SSH Key Pair).
*   Type `yes` to confirm.

## 2. Connect to EC2
After deployment, Terraform will output the **SSH Command**.
Example:
```bash
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_IP>
```
*   The instance comes pre-installed with `uv`, `git`, and `python3`.
*   **Permissions**: The instance has an **IAM Role** attached that automatically grants read/write access to the S3 bucket. You do not need to run `aws configure` on the server for S3 access.

## 3. S3 Artifacts
*   **Bucket Name**: Output by Terraform as `s3_bucket_name`.
*   **Access**:
    *   **From EC2**: Automatic (via Instance Profile).
    *   **From Local**: Requires your local `~/.aws/credentials`.

## 4. Setup on EC2
Once logged in via SSH:
1.  Clone your repo:
    ```bash
    git clone https://github.com/YOUR_USERNAME/movielens-rating-prediction.git
    cd movielens-rating-prediction
    ```
2.  Run the pipeline:
    ```bash
    uv sync
    # Export Neon DB URL (Required for MLflow tracking)
    export MLFLOW_TRACKING_URI="postgresql://user:pass@ep-xyz.us-east-1.aws.neon.tech/neondb"
    # Export S3 Bucket (Found in terraform output)
    export MLFLOW_ARTIFACT_ROOT="s3://movielens-mlflow-artifacts-XXXX"
    
    # Train
    uv run python -m src.models.train_xgboost
    ```

## 5. Cleanup
To destroy all resources and stop billing:
```bash
terraform destroy
```
