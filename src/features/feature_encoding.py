import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import src.utils as utils

logger = utils.configure_logger(__name__, log_file="feature_engineering.log")


def load_data(data_dir: str = "data/interim") -> pd.DataFrame:
    try:
        logger.info(f"Loading cleaned dataset from {data_dir}")
        df = pd.read_csv(os.path.join(data_dir, "feature_engineering.csv"))
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def encode_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    try:
        logger.info(f"Encoding columns: {columns}")
        for column in columns:
            if column in df.columns:
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
            else:
                logger.warning(f"Column {column} not found in DataFrame")
        return df
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return df


def drop_unused_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    try:
        logger.info(f"Dropping columns: {columns_to_drop}")
        return df.drop(columns=columns_to_drop, errors='ignore')
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        return df


def main():
    df = load_data()
    if df.empty:
        logger.error("Training data not loaded. Exiting pipeline.")
        return

    columns_to_encode = ["Sex", "Embarked", "Title", "FareBin", "AgeBin"]
    feature_encoding = encode_columns(df, columns_to_encode)

    features = drop_unused_columns(feature_encoding, ["Name", "SibSp", "Parch"])

    data_path = os.path.join("data", "processed")
    utils.save_data(features, "features.csv", data_path, logger=logger)


if __name__ == "__main__":
    main()
