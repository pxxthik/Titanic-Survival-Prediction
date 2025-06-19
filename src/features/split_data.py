import os
import pandas as pd
from sklearn.model_selection import train_test_split
import src.utils as utils

logger = utils.configure_logger(__name__, log_file="split_data.log")


def load_data() -> pd.DataFrame:
    try:
        logger.debug("Loading training Data")
        data_path = os.path.join("data", "processed")
        train_data = pd.read_csv(os.path.join(data_path, "features.csv"))
        return train_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    # load params
    params = utils.load_params("params.yaml", section="split_data", logger=logger)

    # load data
    df = load_data()

    if df.empty:
        logger.warning("Data is empty. Aborting data split.")
        return

    try:
        test_size = params["test_size"]
        random_state = params["random_state"]

        logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        if train_df.empty or test_df.empty:
            logger.warning("One of the splits is empty. Aborting saving step.")
            return

        data_path = os.path.join("data", "processed")
        utils.save_data(train_df, "train.csv", data_path, logger=logger)
        utils.save_data(test_df, "validation.csv", data_path, logger=logger)
    except Exception as e:
        logger.error(f"Error during dataset split: {e}")


if __name__ == "__main__":
    main()
