from flask import Flask
from dotenv import load_dotenv
import os

from routes.main_routes import main_blueprint

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

app = Flask(__name__)
app.register_blueprint(main_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
