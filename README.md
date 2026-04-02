# MovieLens Rating Prediction Project

## Project Overview
This project builds a machine learning pipeline to predict movie ratings (1-5 stars) using the MovieLens 1M dataset. It features an automated AWS cloud workflow with **MLflow** for experiment tracking and **FastAPI** for model serving.

---

## 🚀 1. Accessing Services (Cloud)
If the project is deployed to AWS, here is how you interact with it.

### Service URLs
Replace `<PUBLIC_IP>` with your EC2 instance IP (e.g., `100.48.xx.xx`).

| Service | Port | URL | Description |
| :--- | :--- | :--- | :--- |
| **MLflow UI** | 5000 | `http://<PUBLIC_IP>:5000` | View Experiments, Metrics, and Model Registry. |
| **API Docs** | 8000 | `http://<Public-IP>:8000/docs` | Interactive Swagger UI to test predictions. |

### 🔍 How to make Predictions (Swagger UI)
1.  Go to the **API Docs** URL.
2.  Choose the specific endpoint for the model you want to use:
    *   `POST /predict/xgboost` (Recommended)
    *   `POST /predict/glm`
    *   `POST /predict/randomforest`
3.  Click **Try it out** -> **Execute**.

---

## ☁️ 2. AWS Cloud Deployment (Production)
This section is for DevOps/Admins deploying the infrastructure.

### Cost Management
You can **Stop** the instance when not in use to save money.

#### Option A: AWS Console
*   **Stop**: AWS Console -> Instance State -> Stop.
*   **Start**: AWS Console -> Instance State -> Start.

#### Option B: AWS CLI
Prerequisite: You need the Instance ID.

**Find Instance ID (Running only):**
```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=MovieLens-ML-Server" "Name=instance-state-name,Values=running" \
  --query "Reservations[*].Instances[*].InstanceId" \
  --output text
```

```bash
# 1. Stop Instance
aws ec2 stop-instances --instance-ids <INSTANCE_ID>

# 2. Start Instance
aws ec2 start-instances --instance-ids <INSTANCE_ID>

# 3. Get New Public IP (After starting)
aws ec2 describe-instances \
  --instance-ids <INSTANCE_ID> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

# 4. Check Status (Table View) 
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=MovieLens-ML-Server" \
  --query "Reservations[*].Instances[*].{ID:InstanceId,State:State.Name,PublicIP:PublicIpAddress}" \
  --output table
```

**Note**: When you start the instance, MLflow and FastAPI start **automatically**. Just SSH in with the new IP!


### Maintenance & Updates
Terraform installs everything **once** at creation. To **update** the running server with your latest code:

1.  **SSH into the Server**:
    ```bash
    ssh -i ~/.ssh/mlflow-key.pem ubuntu@<PUBLIC_IP>
    ```
2.  **Pull Latest Changes**:
    ```bash
    cd movielens-rating-prediction
    git pull
    /home/ubuntu/.local/bin/uv sync
    # Restart API to apply changes
    systemctl restart fastapi
    ```
3.  **Check Services**:
    ```bash
    systemctl status fastapi
    systemctl status mlflow
    ```

### Run MLOps Pipeline (Remote)
Execute these scripts on the server to train and update models:
1.  **Train**: `uv run python -m src.models.train_xgboost`
2.  **Compare**: `uv run scripts/post_training/compare_models.py`
3.  **Register**: `uv run scripts/post_training/register_model.py`

---

## 💻 3. Local Development
This section is for Developers attempting to run the code on their machine.

### Prerequisites
*   **Python 3.10+**
*   **uv** (Package Manager)
*   **AWS CLI** (For data access)

### Installation
1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Install AWS CLI**:
    ```bash
    # macOS
    brew install awscli
    # Linux
    sudo apt install awscli
    ```
3.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

### Data Pipeline
Run the orchestration script to process the data:
```bash
uv run scripts/run_pipeline.py
```
*   **Input**: `data/rawData/merged_movielens.csv` (Will download from S3 if missing and AWS creds are set).
*   **Output**: `data/processed/{train,validate,test}.csv`.


### Training Models Locally
You can train models on your laptop (logs to local MLflow or Cloud if URI is set):
```bash
# Train XGBoost
uv run python -m src.models.train_xgboost

# Train ElasticNet
uv run python -m src.models.train_glm
```

### Experiment Tracking (Local)
View results without the cloud. Local runs are stored in `./mlruns` to **keep production clean** and allow offline development.

```bash
uv run mlflow ui
```
Open `http://127.0.0.1:5000`.
