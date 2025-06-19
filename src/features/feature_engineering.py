import os
import pandas as pd
import src.utils as utils

logger = utils.configure_logger(__name__, log_file="feature_engineering.log")


def load_data(data_dir: str = "data/interim") -> pd.DataFrame:
    try:
        logger.info("Loading processed training dataset.")
        df = pd.read_csv(os.path.join(data_dir, "titanic_cleaned.csv"))
        return df
    except Exception as e:
        logger.error(f"Error loading processed training data: {e}")
        return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Starting feature engineering.")

        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["isAlone"] = (df["FamilySize"] > 1).astype(int)

        df["Title"] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

        df["FareBin"] = pd.qcut(df["Fare"], 4, duplicates='drop')
        df["AgeBin"] = pd.cut(df["Age"].astype(int), 5)

        top_min = 4
        top_title_names = df["Title"].value_counts().head(top_min).index
        df["Title"] = df["Title"].apply(lambda x: "Misc" if x not in top_title_names else x)

        logger.info("Feature engineering completed.")
        return df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        return df

def main():
    df = load_data()
    if df.empty:
        logger.error("Training data not loaded. Exiting pipeline.")
        return

    feature_engineering = engineer_features(df)

    data_path = os.path.join("data", "interim")
    utils.save_data(feature_engineering, "feature_engineering.csv", data_path, logger=logger)


if __name__ == "__main__":
    main()
