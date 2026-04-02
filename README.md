# MovieLens Recommendation System

## Overview

This project builds an end-to-end machine learning pipeline to predict user movie ratings (1–5 stars) using the MovieLens 1M dataset.

The system demonstrates a full production-style workflow including:

* Data processing
* Model training and evaluation
* Experiment tracking
* Model registry
* API-based model serving
* Cloud deployment using AWS

The goal of this project is to showcase a real-world MLOps pipeline rather than just a single model.

---

## Key Features

* Predicts movie ratings using structured user and movie data
* Trains and compares multiple machine learning models
* Tracks experiments, metrics, and parameters using MLflow
* Registers and versions models in the MLflow Model Registry
* Serves predictions through FastAPI endpoints
* Deploys infrastructure using Terraform on AWS
* Supports both local and cloud-based workflows

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn, H2O, XGBoost
* MLflow
* FastAPI
* Terraform
* AWS (EC2, S3)
* uv (Python package manager)

---

## Project Structure

```
movielens-recommendation-system/
├── docs/                  # project outputs, screenshots, drift analysis
├── infra/                 # Terraform configuration for AWS
├── scripts/               # pipeline and post-training scripts
├── src/
│   ├── app/               # FastAPI application
│   ├── data/              # data handling modules
│   └── models/            # model training scripts
├── data/                  # raw and processed data (ignored in Git)
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

---

## Machine Learning Workflow

1. Raw MovieLens data is collected and prepared
2. Data is split into train, validation, and test sets
3. Multiple models are trained
4. Results are logged to MLflow
5. Models are compared and evaluated
6. The best model is registered as the champion
7. The model is deployed through FastAPI
8. Infrastructure is managed using Terraform on AWS

---

## Models Used

* XGBoost (Primary model)
* Generalized Linear Model (ElasticNet)
* Distributed Random Forest (H2O)
* H2O AutoML workflows

Models are compared using RMSE, MAE, and other evaluation metrics.

---

## Local Development

### Prerequisites

* Python 3.10+
* uv package manager
* AWS CLI (optional, for cloud data access)

### Install Dependencies

```bash
uv sync
```

### Run Data Pipeline

```bash
uv run scripts/run_pipeline.py
```

### Train Models

```bash
uv run python -m src.models.train_xgboost
uv run python -m src.models.train_glm
uv run python -m src.models.train_drf
uv run python -m src.models.train_h2o
```

### Run MLflow Locally

```bash
uv run mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

### Run FastAPI Locally

```bash
uv run uvicorn src.app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## AWS Deployment

This project can be deployed to AWS using Terraform (see `infra/`).

Once deployed, services include:

* MLflow UI (port 5000)
* FastAPI API (port 8000)

Access via:

```
http://<PUBLIC_IP>:5000
http://<PUBLIC_IP>:8000/docs
```

---

## Maintenance and Updates

To update a running EC2 instance:

```bash
ssh -i ~/.ssh/mlflow-key.pem ubuntu@<PUBLIC_IP>

cd movielens-rating-prediction
git pull
/home/ubuntu/.local/bin/uv sync

systemctl restart fastapi
systemctl restart mlflow
```

Check service status:

```bash
systemctl status fastapi
systemctl status mlflow
```

---

## Example Pipeline Commands

```bash
uv run python -m src.models.train_xgboost
uv run scripts/post_training/compare_models.py
uv run scripts/post_training/register_model.py
```

---

## Notes

* Raw datasets, MLflow runs, and large artifacts are excluded using `.gitignore`
* This repository focuses on code, pipeline design, and deployment

---

## Results

The models were evaluated on the test set using RMSE, MAE, and R².

| Model                        | RMSE       | MAE    | R²     |
| ---------------------------- | ---------- | ------ | ------ |
| **Random Forest** (Champion) | **1.0423** | 0.8483 | 0.1269 |
| XGBoost                      | 1.0536     | 0.8465 | 0.1080 |
| GLM (ElasticNet)             | 1.1061     | 0.9153 | 0.0168 |

The **Random Forest model** achieved the lowest RMSE and was selected as the champion model for deployment.

---

## Screenshots

### Champion Model Selection (MLflow)
![Champion Table](docs/Champion%20SS/Champion%20Model%20Table.png)

![Model Registered](docs/Champion%20SS/Champion%20Registered.png)

![Model Version](docs/Champion%20SS/Champion%20Version.png)

---

### FastAPI Predictions (Model Inference)

#### XGBoost
![XGBoost Request](docs/FastAPI%20Test/xgBoost_post_part1.png)
![XGBoost Response](docs/FastAPI%20Test/xgBoost_response_part2.png)

#### Random Forest
![RF Request](docs/FastAPI%20Test/randomForest_post_part1.png)
![RF Response](docs/FastAPI%20Test/randomForest_response_part2.png)

#### GLM
![GLM Request](docs/FastAPI%20Test/glm_post_part1.png)
![GLM Response](docs/FastAPI%20Test/glm_response_part2.png)

---

### Model Drift Analysis
![Drift Summary](docs/Model%20Drift%20Analysis/model_drift_summary_report_0.png)
![Drift Detail](docs/Model%20Drift%20Analysis/model_drift_summary_report_userId_details.png)

---

### Key Takeaways

* Tree-based ensemble models (Random Forest, XGBoost) significantly outperformed linear models
* Random Forest provided the best overall balance of error metrics
* MLflow enabled systematic comparison across models and experiments
* The pipeline supports reproducible model training and evaluation

---

## Future Improvements

* CI/CD pipeline for automated deployment
* Model monitoring and drift detection alerts
* Batch prediction endpoints
* Frontend dashboard for model interaction

---

## Author

**Mark Smeltzer**
MS Data Science, Rowan University
