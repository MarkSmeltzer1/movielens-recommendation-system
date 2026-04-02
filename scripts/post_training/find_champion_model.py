import math
import tempfile
import pandas as pd
import joblib
import xgboost as xgb
import mlflow
import os
import sys
import boto3

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dynamic Configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "movielens_cloud"
TEST_CSV = os.path.join(os.getcwd(), "data/processed/test.csv")
TARGET = "rating"
MODELS_TO_COMPARE = ["GLM", "RandomForest", "XGBoost"]

def RMSE(y, p): return math.sqrt(mean_squared_error(y, p))

def encode_gender_if_needed(df):
    if "gender" in df.columns and df["gender"].dtype == "object":
        df = df.copy()
        df["gender"] = df["gender"].map({"F": 0, "M": 1})
    return df

def get_best_run_id(client, experiment_id, model_name):
    """Find the run with the lowest RMSE for a given model type."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"params.model_type = '{model_name}'",
        order_by=["metrics.valid_rmse ASC"],
        max_results=1
    )
    if not runs:
        print(f"Warning: No runs found for model type '{model_name}'")
        return None
    return runs[0].info.run_id

def main():
    print(f"Connecting to MLflow at {TRACKING_URI}...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Get Experiment ID
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        sys.exit(1)
    
    experiment_id = exp.experiment_id

    # Load Test Data
    if not os.path.exists(TEST_CSV):
        print(f"Error: Test data not found at {TEST_CSV}")
        sys.exit(1)

    print("Loading test data...")
    df = pd.read_csv(TEST_CSV)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    
    if "rating_datetime" in X.columns:
        X = X.drop(columns=["rating_datetime"])
    X = encode_gender_if_needed(X)

    rows = []

    print("\n--- Starting Model Evaluation ---")
    for model_name in MODELS_TO_COMPARE:
        run_id = get_best_run_id(client, experiment_id, model_name)
        if not run_id:
            continue
            
        print(f"Evaluating {model_name} (RunID: {run_id})...")

        try:
            with tempfile.TemporaryDirectory() as td:
                if model_name == "GLM":
                    model_path = client.download_artifacts(run_id, "model.joblib", td)
                    scaler_path = client.download_artifacts(run_id, "scaler.joblib", td)
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    if hasattr(scaler, "feature_names_in_"):
                        X_use = X[list(scaler.feature_names_in_)].copy()
                    else:
                        X_use = X.copy()
                    
                    X_use = encode_gender_if_needed(X_use)
                    X_use.loc[:, X_use.columns] = scaler.transform(X_use)
                    pred = model.predict(X_use)

                elif model_name == "RandomForest":
                    model_path = client.download_artifacts(run_id, "model.joblib", td)
                    model = joblib.load(model_path)
                    if hasattr(model, "feature_names_in_"):
                        X_use = X[list(model.feature_names_in_)].copy()
                    else:
                        X_use = X.copy()
                    X_use = encode_gender_if_needed(X_use)
                    pred = model.predict(X_use)

                elif model_name == "XGBoost":
                    model_path = client.download_artifacts(run_id, "model.json", td)
                    booster = xgb.Booster()
                    booster.load_model(model_path)
                    feats = booster.feature_names
                    X_use = X[feats].copy() if feats else X.copy()
                    X_use = encode_gender_if_needed(X_use)
                    pred = booster.predict(xgb.DMatrix(X_use))

                rows.append({
                    "Model": model_name,
                    "RunID": run_id,
                    "RMSE": RMSE(y, pred),
                    "MAE": mean_absolute_error(y, pred),
                    "R2": r2_score(y, pred),
                })
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    if not rows:
        print("No models evaluated.")
        return

    out = pd.DataFrame(rows).sort_values(["RMSE"], ascending=[True])
    
    # Store the human-readable report
    report_lines = []
    report_lines.append("=== CHAMPION TABLE (Test Set) ===")
    report_lines.append(out.to_string(index=False))

    champ = out.iloc[0]
    report_lines.append(f"\nCHAMPION = {champ['Model']} (lowest RMSE)")
    
    report_content = "\n".join(report_lines)
    print("\n" + report_content)

    # Save to text file (easier for humans to read)
    report_file = "champion_report.txt"
    with open(report_file, "w") as f:
        f.write(report_content)
    print(f"\nSaved report: {report_file}")
    
    # Also save clean CSV for machine processing (optional, keeping it just in case)
    out.to_csv("champion_table.csv", index=False)

    # Upload to S3 if bucket is specified
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if s3_bucket:
        print(f"\nUploading to S3 bucket: {s3_bucket}...")
        try:
            s3 = boto3.client("s3")
            s3.upload_file(report_file, s3_bucket, report_file)
            print(f"Successfully uploaded to s3://{s3_bucket}/{report_file}")
            
            # optional: upload csv too if needed, but per request focusing on the 'pdf'/report view
        except Exception as e:
            print(f"Error uploading to S3: {e}")
        except Exception as e:
            print(f"Error uploading to S3: {e}")
    else:
        print("\nSkipping S3 upload. Set S3_BUCKET_NAME env var to upload.")

if __name__ == "__main__":
    main()
