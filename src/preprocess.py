"""
preprocess.py

This module prepares the raw flight dataset for machine learning.
It handles:
- duplicate removal
- missing values
- target separation
- one-hot encoding of categorical variables
- train/test splitting
"""

# Import pandas for DataFrame operations
import pandas as pd

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    """
    Clean and preprocess the raw dataset.

    Args:
        df (pd.DataFrame):
            The raw combined flight DataFrame.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test, processed_df
    """

    # Create a copy of the input DataFrame to avoid modifying the original
    data = df.copy()

    # Standardize all column names:
    # - remove leading/trailing spaces
    # - convert to lowercase
    # - replace spaces with underscores
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Remove duplicate rows to reduce repeated information
    data = data.drop_duplicates()

    # Drop rows with missing values for simplicity in this baseline version
    data = data.dropna()

    # Check whether the target column exists in the dataset
    if "price" not in data.columns:
        raise ValueError("Expected a 'price' column in the dataset.")

    # Separate the target variable from the feature columns
    y = data["price"]

    # Remove the target column from the feature set
    X = data.drop(columns=["price"])

    # Convert categorical columns into numeric dummy/indicator columns
    X = pd.get_dummies(X, drop_first=True)

    # Split the features and target into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,                  # feature matrix
        y,                  # target vector
        test_size=0.2,      # use 20% of data for testing
        random_state=42     # fixed seed for reproducibility
    )

    # Return the split datasets plus the cleaned original data
    return X_train, X_test, y_train, y_test, data
