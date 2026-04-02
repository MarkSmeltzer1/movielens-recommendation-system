import tempfile
import mlflow
import joblib
import xgboost as xgb
import pandas as pd
import os
import sys
from mlflow.tracking import MlflowClient

# Dynamic Configuration
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "movielens_cloud"
MODEL_NAME = "movielens_top3"   # Registered Model Name
TEST_CSV = os.path.join(os.getcwd(), "data/processed/test.csv")
TARGET = "rating"
MODELS_TO_REGISTER = ["GLM", "RandomForest", "XGBoost"]

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

# --- PyFunc Wrapper Classes ---
class GLMPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.cols = list(getattr(self.scaler, "feature_names_in_", []))

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        X.loc[:, X.columns] = self.scaler.transform(X)
        return self.model.predict(X)

class JoblibPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.cols = list(getattr(self.model, "feature_names_in_", []))

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        return self.model.predict(X)

class XGBoostPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["model"])
        self.cols = self.booster.feature_names or []

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        return self.booster.predict(xgb.DMatrix(X))

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

    # Load small input example for signature logging
    print("Loading input example from test data...")
    df = pd.read_csv(TEST_CSV).drop(columns=[TARGET])
    if "rating_datetime" in df.columns:
        df = df.drop(columns=["rating_datetime"])
    df = encode_gender_if_needed(df)
    input_example = df.head(5)

    # Start Registration Run
    print(f"Starting registration process in experiment '{EXPERIMENT_NAME}'...")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="register_top3_models") as run:
        for model_name in MODELS_TO_REGISTER:
            src_run_id = get_best_run_id(client, experiment_id, model_name)
            if not src_run_id:
                print(f"Skipping {model_name} (No run found)")
                continue

            print(f"Registering {model_name} (Source Run: {src_run_id})...")
            
            with tempfile.TemporaryDirectory() as td:
                if model_name == "GLM":
                    m = client.download_artifacts(src_run_id, "model.joblib", td)
                    s = client.download_artifacts(src_run_id, "scaler.joblib", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=GLMPyFunc(),
                        artifacts={"model": m, "scaler": s},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{model_name.lower()}"

                elif model_name == "RandomForest":
                    m = client.download_artifacts(src_run_id, "model.joblib", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=JoblibPyFunc(),
                        artifacts={"model": m},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{model_name.lower()}"

                elif model_name == "XGBoost":
                    m = client.download_artifacts(src_run_id, "model.json", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{model_name.lower()}",
                        python_model=XGBoostPyFunc(),
                        artifacts={"model": m},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{model_name.lower()}"

            # Register
            mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
            client.set_model_version_tag(MODEL_NAME, mv.version, "source_run_id", src_run_id)
            client.set_model_version_tag(MODEL_NAME, mv.version, "model_label", model_name)
            print(f"✅ Registered {model_name} as {MODEL_NAME} v{mv.version}")

if __name__ == "__main__":
    main()
