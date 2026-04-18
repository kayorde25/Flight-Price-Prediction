"""
predict.py

This script demonstrates how to load the trained model and generate predictions
for a small batch of example data from the original dataset.
"""

# Import pathlib for file path handling
from pathlib import Path

# Import joblib for loading saved model artifacts
import joblib

# Import pandas for preparing input data
import pandas as pd

# Import the local data loading helper
from load_data import load_data


def main():
    """
    Load the saved model and run prediction on a small sample.
    """

    # Define the project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Define the models directory path
    models_dir = project_root / "models"

    # Load the trained random forest model
    model = joblib.load(models_dir / "random_forest_model.pkl")

    # Load the feature column names used during training
    trained_features = joblib.load(models_dir / "model_features.pkl")

    # Load the original raw data
    df = load_data()

    # Create a copy of the DataFrame to avoid changing the original
    data = df.copy()

    # Standardize the column names in the same way as training
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Drop rows with missing values for this simple prediction demo
    data = data.dropna()

    # Remove duplicates for consistency
    data = data.drop_duplicates()

    # Remove the target column because we only want feature inputs
    X = data.drop(columns=["price"])

    # Apply one-hot encoding to categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Reindex columns so they match exactly what the model saw during training
    X = X.reindex(columns=trained_features, fill_value=0)

    # Select the first 5 rows as an example prediction batch
    sample_input = X.head(5)

    # Use the trained model to predict prices
    predicted_prices = model.predict(sample_input)

    # Print each prediction result clearly
    print("Predicted prices for sample rows:")
    for i, price in enumerate(predicted_prices, start=1):
        print(f"Row {i}: {price:.2f}")


# Run the prediction script if executed directly
if __name__ == "__main__":
    main()
