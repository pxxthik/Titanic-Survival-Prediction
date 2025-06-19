import mlflow

def get_latest_model_version(model_name, dagshub_url, repo_owner, repo_name):
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

def load_model(model_name, dagshub_url, repo_owner, repo_name):
    version = get_latest_model_version(model_name, dagshub_url, repo_owner, repo_name)
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(model_uri)
