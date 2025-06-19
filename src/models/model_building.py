import pandas as pd

import os
import pickle

import src.utils as utils

from sklearn.ensemble import RandomForestClassifier

# logging configure
logger = utils.configure_logger(__name__, log_file="model_building.log")


def load_data() -> pd.DataFrame:
    try:
        logger.debug("Loading training Data")
        data_path = os.path.join("data", "processed")
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        return train_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":

    # load params
    params = utils.load_params("params.yaml", section="model_building", logger=logger)
    # load data
    train_data = load_data()

    # split X and y
    X_train = train_data.drop("Survived", axis=1)
    y_train = train_data["Survived"]

    # train model
    model = RandomForestClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        n_estimators=params["n_estimators"]
    )
    logger.info("Training Model")
    model.fit(X_train, y_train)

    # save model
    model_path = os.path.join("models")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    logger.info("Model saved successfully")
