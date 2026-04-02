import pandas as pd
import numpy as np
import os
import logging
import mlflow
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("movielens_cloud")

def load_data(path):
    """Load data and encode features for GLM."""
    df = pd.read_csv(path)

    feature_cols = (
        ['userId', 'movieId', 'gender', 'age', 'occupation'] +
        [c for c in df.columns if c.startswith('genre_')]
    )
    target_col = 'rating'

    for col in ['gender']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_model(train_path, valid_path, alpha=0.1, l1_ratio=0.5, output_dir="models/glm"):
    logger.info("Initializing GLM (ElasticNet) training...")

    with mlflow.start_run(run_name="GLM"):
        # Log Params
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "model_type": "GLM"
        }
        mlflow.log_params(params)

        # 1. Load Data
        X_train, y_train = load_data(train_path)
        X_valid, y_valid = load_data(valid_path)

        # 2. Scale Data (Important for Linear Models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        logger.info(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")

        # 3. Train Model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 4. Evaluate
        preds_valid = model.predict(X_valid_scaled)

        rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
        mae = mean_absolute_error(y_valid, preds_valid)
        r2 = r2_score(y_valid, preds_valid)

        logger.info(f"Validation RMSE: {rmse:.4f}")
        logger.info(f"Validation MAE: {mae:.4f}")
        logger.info(f"Validation R2: {r2:.4f}")

        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mae", mae)
        mlflow.log_metric("valid_r2", r2)

        # 5. Plot Artifact: Actual vs Predicted
        plt.figure()
        plt.scatter(y_valid, preds_valid, alpha=0.5)
        plt.xlabel("Actual Ratings")
        plt.ylabel("Predicted Ratings")
        plt.title("Actual vs Predicted Ratings (GLM ElasticNet)")
        plot_path = "actual_vs_predicted_glm.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # 6. Plot Artifact: Residuals vs Actual
        residuals = preds_valid - y_valid

        plt.figure()
        plt.scatter(y_valid, residuals, alpha=0.5)
        plt.axhline(0)
        plt.xlabel("Actual Ratings")
        plt.ylabel("Residual (Predicted - Actual)")
        plt.title("Residuals vs Actual Ratings (GLM ElasticNet)")
        residual_plot_path = "residuals_vs_actual_glm.png"
        plt.savefig(residual_plot_path)
        plt.close()
        mlflow.log_artifact(residual_plot_path)

        # 7. Save Model AND Scaler
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.joblib")
        scaler_path = os.path.join(output_dir, "scaler.joblib")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    if os.path.exists("data/processed/train.csv"):
        train_model(
            train_path="data/processed/train.csv",
            valid_path="data/processed/validate.csv"
        )
    else:
        logger.warning("Data not found.")
