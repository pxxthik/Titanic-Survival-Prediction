import os

import zipfile
from pathlib import Path
from typing import Optional
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

import src.utils as utils


# logging configure
logger = utils.configure_logger(__name__, log_file="data_ingestion.log")

class KaggleDataIngestion:
    """A class to handle Kaggle data ingestion with proper error handling."""

    def __init__(self):
        self.api = None

    def setup_kaggle_credentials(self) -> bool:
        """
        Set up Kaggle API credentials from environment variables.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Load .env if running locally (ignored in CI if env vars already set)
            load_dotenv()

            username = os.getenv("KAGGLE_USERNAME")
            key = os.getenv("KAGGLE_KEY")

            if not username or not key:
                raise ValueError(
                    "KAGGLE_USERNAME and KAGGLE_KEY must be set in environment variables"
                )

            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)

            kaggle_json_path = kaggle_dir / "kaggle.json"

            # Write credentials to kaggle.json
            with open(kaggle_json_path, "w") as f:
                f.write(f'{{"username":"{username}","key":"{key}"}}')

            # Set proper permissions (600 = read/write for owner only)
            os.chmod(kaggle_json_path, 0o600)

            logger.info("Kaggle credentials setup successfully")
            return True

        except ValueError as e:
            logger.error(f"Environment variable error: {e}")
            return False
        except OSError as e:
            logger.error(f"File system error during credential setup: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during credential setup: {e}")
            return False

    def authenticate_api(self) -> bool:
        """
        Authenticate with Kaggle API.

        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Kaggle API authenticated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle API: {e}")
            return False

    def create_output_directory(self, output_dir: str) -> bool:
        """
        Create output directory if it doesn't exist.

        Args:
            output_dir (str): Path to output directory

        Returns:
            bool: True if directory creation successful, False otherwise
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {output_dir}")
            return True

        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating directory {output_dir}: {e}")
            return False

    def download_competition_files(
        self, competition_name: str, output_dir: str
    ) -> bool:
        """
        Download competition files from Kaggle.

        Args:
            competition_name (str): Name of the Kaggle competition
            output_dir (str): Directory to save downloaded files

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            if not self.api:
                logger.error("API not authenticated. Call authenticate_api() first.")
                return False

            zip_path = Path(output_dir) / f"{competition_name}.zip"

            logger.info(f"Downloading competition '{competition_name}' to {output_dir}")
            self.api.competition_download_files(competition_name, path=output_dir)

            if not zip_path.exists():
                logger.error(f"Downloaded zip file not found: {zip_path}")
                return False

            logger.info(f"Competition files downloaded successfully: {zip_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download competition '{competition_name}': {e}")
            return False

    def extract_zip_file(
        self,
        competition_name: str,
        output_dir: str,
        exclude_files: Optional[list] = None,
    ) -> bool:
        """
        Extract downloaded zip file and clean up, with option to exclude specific files.

        Args:
            competition_name (str): Name of the competition (used for zip filename)
            output_dir (str): Directory containing the zip file
            exclude_files (list, optional): List of filenames to exclude from extraction

        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            zip_path = Path(output_dir) / f"{competition_name}.zip"

            if not zip_path.exists():
                logger.error(f"Zip file not found: {zip_path}")
                return False

            if exclude_files is None:
                exclude_files = []

            logger.info(f"Extracting {zip_path} to {output_dir}")
            if exclude_files:
                logger.info(f"Excluding files: {exclude_files}")

            extracted_files = []

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    # Skip directories and excluded files
                    if file_info.is_dir():
                        continue

                    filename = os.path.basename(file_info.filename)

                    if filename in exclude_files:
                        logger.info(f"Skipping excluded file: {filename}")
                        continue

                    # Extract the file
                    zip_ref.extract(file_info, output_dir)
                    extracted_files.append(filename)
                    logger.info(f"Extracted: {filename}")

            # Clean up zip file
            zip_path.unlink()
            logger.info(
                f"Files extracted successfully ({len(extracted_files)} files) and zip file removed"
            )
            logger.info(f"Extracted files: {extracted_files}")
            return True

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file {zip_path}: {e}")
            return False
        except OSError as e:
            logger.error(f"File system error during extraction: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
            return False

    def download_and_extract_competition(
        self,
        competition_name: str,
        output_dir: str,
        exclude_files: Optional[list] = None,
    ) -> bool:
        """
        Complete workflow to download and extract competition data.

        Args:
            competition_name (str): Name of the Kaggle competition
            output_dir (str): Directory to save extracted files
            exclude_files (list, optional): List of filenames to exclude from extraction

        Returns:
            bool: True if entire process successful, False otherwise
        """
        try:
            logger.info(f"Starting data ingestion for competition: {competition_name}")

            # Step 1: Setup credentials
            if not self.setup_kaggle_credentials():
                logger.error("Failed to setup Kaggle credentials")
                return False

            # Step 2: Authenticate API
            if not self.authenticate_api():
                logger.error("Failed to authenticate with Kaggle API")
                return False

            # Step 3: Create output directory
            if not self.create_output_directory(output_dir):
                logger.error(f"Failed to create output directory: {output_dir}")
                return False

            # Step 4: Download competition files
            if not self.download_competition_files(competition_name, output_dir):
                logger.error(
                    f"Failed to download competition files: {competition_name}"
                )
                return False

            # Step 5: Extract zip file (excluding specified files)
            if not self.extract_zip_file(competition_name, output_dir, exclude_files):
                logger.error(f"Failed to extract competition files: {competition_name}")
                return False

            logger.info(f"Data ingestion completed successfully for {competition_name}")
            return True

        except Exception as e:
            logger.error(f"Unexpected error in download_and_extract_competition: {e}")
            return False


def main():
    """Main function to run the data ingestion process."""
    try:
        # load params
        params = utils.load_params(
            "params.yaml", section="data_ingestion", logger=logger
        )
        if not params:
            raise ValueError("Params not found")
        
        # Initialize the data ingestion class
        ingestion = KaggleDataIngestion()

        # Download and extract Titanic competition data (excluding submission.csv)
        competition_name = params["kaggle_competition"]
        output_dir = "data/raw"
        exclude_files = params["exclude_files"]  # Common submission file name

        success = ingestion.download_and_extract_competition(
            competition_name, output_dir, exclude_files=exclude_files
        )

        if success:
            logger.info("Data ingestion process completed successfully!")
            logger.info("Only train.csv and test.csv have been extracted")
            return 0
        else:
            logger.error("Data ingestion process failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
