import xgboost as xgb
import pandas as pd
import numpy as np
import os
import logging
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Configuration
# If MLFLOW_TRACKING_URI is not set, it defaults to local ./mlruns
# If MLFLOW_ARTIFACT_ROOT is set (e.g., s3://...), we don't set it manually in set_experiment,
# but we trust the user environment or set it explicitly if needed.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
experiment_name = "movielens_cloud"
mlflow.set_experiment(experiment_name)

def load_data(path):
    """Load data and prepare for XGBoost (Label Encode Categoricals)."""
    df = pd.read_csv(path)

    feature_cols = (
        ['userId', 'movieId', 'gender', 'age', 'occupation'] +
        [c for c in df.columns if c.startswith('genre_')]
    )
    target_col = 'rating'

    # Map categorical string columns to codes
    for col in ['gender']:
        df[col] = df[col].astype('category').cat.codes

    X = df[feature_cols]
    y = df[target_col]

    return X, y

def train_model(train_path, valid_path, n_estimators=100, max_depth=6, learning_rate=0.1, output_dir="models/xgboost"):
    logger.info("Initializing XGBoost training...")

    with mlflow.start_run(run_name="XGBoost"):
        # Log Params
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "model_type": "XGBoost"
        }
        mlflow.log_params(params)

        # 1. Load Data
        X_train, y_train = load_data(train_path)
        X_valid, y_valid = load_data(valid_path)

        logger.info(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")

        # 2. Train Model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False
        )

        # 3. Evaluate
        preds_valid = model.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
        mae = mean_absolute_error(y_valid, preds_valid)
        r2 = r2_score(y_valid, preds_valid)

        logger.info(f"Validation RMSE: {rmse:.4f}")
        logger.info(f"Validation MAE: {mae:.4f}")
        logger.info(f"Validation R2: {r2:.4f}")

        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mae", mae)
        mlflow.log_metric("valid_r2", r2)

        # 4. Plot Artifact: Actual vs Predicted
        plt.figure()
        plt.scatter(y_valid, preds_valid, alpha=0.5)
        plt.xlabel("Actual Ratings")
        plt.ylabel("Predicted Ratings")
        plt.title("Actual vs Predicted Ratings (XGBoost)")
        plot_path = "actual_vs_predicted_xgb.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # 5. Plot Artifact: Residuals vs Actual
        residuals = preds_valid - y_valid

        plt.figure()
        plt.scatter(y_valid, residuals, alpha=0.5)
        plt.axhline(0)
        plt.xlabel("Actual Ratings")
        plt.ylabel("Residual (Predicted - Actual)")
        plt.title("Residuals vs Actual Ratings (XGBoost)")
        residual_plot_path = "residuals_vs_actual_xgb.png"
        plt.savefig(residual_plot_path)
        plt.close()
        mlflow.log_artifact(residual_plot_path)

        # 6. Save
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.json")
        model.get_booster().save_model(model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    if os.path.exists("data/processed/train.csv"):
        train_model(
            train_path="data/processed/train.csv",
            valid_path="data/processed/validate.csv"
        )
    else:
        logger.warning("Data not found.")
