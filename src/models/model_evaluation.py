import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv

import src.utils as utils

logger = utils.configure_logger(__name__, log_file="model_evaluation.log")


def setup_mlflow_tracking():
    load_dotenv()
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "pxxthik"
    repo_name = "Titanic-Survival-Prediction"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_validation_data(data_path: str = "data/processed/validation.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Validation data loaded from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return pd.DataFrame()


def evaluate_model(clf, X_test, y_test) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
        }

        logger.info("Model evaluation completed")
        return metrics
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def save_json(data, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")


def log_to_mlflow(clf, metrics: dict, run_id_path: str, metrics_path: str):
    try:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        if hasattr(clf, "get_params"):
            for k, v in clf.get_params().items():
                mlflow.log_param(k, v)

        mlflow.sklearn.log_model(clf, artifact_path="rf_tuned")
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(run_id_path)
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")


def main():
    try:
        setup_mlflow_tracking()

        mlflow.set_experiment("dvc-pipeline")
        with mlflow.start_run() as run:
            clf = load_model("models/model.pkl")

            df = load_validation_data()
            if df.empty:
                logger.warning("Validation data is empty. Exiting.")
                return

            X_test = df.drop("Survived", axis=1)
            y_test = df["Survived"]

            metrics = evaluate_model(clf, X_test, y_test)

            metrics_path = "reports/metrics.json"
            run_id_path = "reports/run_info.json"
            save_json(metrics, metrics_path)
            save_json({"run_id": run.info.run_id, "model_name": "rf_tuned"}, run_id_path)

            log_to_mlflow(clf, metrics, run_id_path, metrics_path)

    except Exception as e:
        logger.error(f"Main evaluation process failed: {e}")
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
