"""
load_data.py

This module is responsible for loading raw flight datasets from disk
and combining them into one pandas DataFrame for downstream processing.
"""

# Import the pathlib module so we can build file paths safely
from pathlib import Path

# Import pandas for data loading and tabular data handling
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Load the business, economy, and clean datasets from the local raw data folder.

    Returns:
        pd.DataFrame:
            A single combined DataFrame containing all rows from the three files.
    """

    # Define the project root directory by moving one level up from the src folder
    project_root = Path(__file__).resolve().parent.parent

    # Define the folder where the raw CSV files are expected to exist
    raw_path = project_root / "data" / "raw"

    # Build the full path to the business dataset file
    business_file = raw_path / "business.csv"

    # Build the full path to the economy dataset file
    economy_file = raw_path / "economy.csv"

    # Build the full path to the clean dataset file
    clean_file = raw_path / "clean_dataset.csv"

    # Check that the business file exists before trying to read it
    if not business_file.exists():
        raise FileNotFoundError(f"Missing file: {business_file}")

    # Check that the economy file exists before trying to read it
    if not economy_file.exists():
        raise FileNotFoundError(f"Missing file: {economy_file}")

    # Check that the clean dataset file exists before trying to read it
    if not clean_file.exists():
        raise FileNotFoundError(f"Missing file: {clean_file}")

    # Read the business CSV file into a DataFrame
    business_df = pd.read_csv(business_file)

    # Read the economy CSV file into a DataFrame
    economy_df = pd.read_csv(economy_file)

    # Read the clean dataset CSV file into a DataFrame
    clean_df = pd.read_csv(clean_file)

    # Combine the three DataFrames row-wise into one DataFrame
    combined_df = pd.concat([business_df, economy_df, clean_df], ignore_index=True)

    # Return the final merged DataFrame
    return combined_df


# This block runs only if the file is executed directly
if __name__ == "__main__":
    # Load the combined dataset
    df = load_data()

    # Print the shape of the dataset to confirm it loaded correctly
    print("Loaded data shape:", df.shape)

    # Print the first five rows for quick inspection
    print(df.head())
