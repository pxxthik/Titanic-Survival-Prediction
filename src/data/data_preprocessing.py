import os
import pandas as pd
import src.utils as utils

logger = utils.configure_logger(__name__, log_file="preprocessing.log")


def load_data(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info("Loading training and test datasets.")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting preprocessing on training data.")

        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

        columns_to_drop = ["PassengerId", "Cabin", "Ticket"]
        df.drop(columns=columns_to_drop, inplace=True)

        logger.info("Preprocessing complete.")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return pd.DataFrame()


def main():
    train_df, _ = load_data()
    if train_df.empty:
        logger.error("Training data not loaded. Exiting pipeline.")
        return

    train_cleaned = preprocess_data(train_df)

    data_path = os.path.join("data", "interim")
    utils.save_data(train_cleaned, "titanic_cleaned.csv", data_path, logger=logger)


if __name__ == "__main__":
    main()
