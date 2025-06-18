import pandas as pd

import os
import logging

import yaml


def load_params(params_filepath: str, section: str, logger: logging) -> dict:
    try:
        logger.debug("Getting Params")
        with open(params_filepath, "r") as f:
            if section == "all":
                params = yaml.safe_load(f)
                return params
            params = yaml.safe_load(f).get(section, {})
        return params
    except FileNotFoundError as e:
        logger.error(f"Params file not found: {e}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing params file: {e}")
        return {}


def configure_logger(
    name: str, log_file: str, log_level: str = "DEBUG"
) -> logging.Logger:
    # logging configure
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel("DEBUG")

    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel("ERROR")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_data(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        data_path: str,
        logger: logging) -> None:
    try:
        logger.debug("Saving Data")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
