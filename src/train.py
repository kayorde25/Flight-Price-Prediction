"""
train.py

This script loads the raw data, preprocesses it, trains a Random Forest model,
evaluates performance, and saves the trained model and useful outputs.
"""

# Import the pathlib module so we can work with project folders safely
from pathlib import Path

# Import joblib for saving trained models to disk
import joblib

# Import matplotlib for creating and saving plots
import matplotlib.pyplot as plt

# Import pandas for handling tabular feature importance output
import pandas as pd

# Import the random forest regressor model
from sklearn.ensemble import RandomForestRegressor

# Import the MAE metric to evaluate prediction error
from sklearn.metrics import mean_absolute_error

# Import local helper function to load the raw data
from load_data import load_data

# Import local helper function to preprocess the dataset
from preprocess import preprocess_data


def main():
    """
    Main training pipeline for the flight price prediction project.
    """

    # Define the project root path
    project_root = Path(__file__).resolve().parent.parent

    # Define where the trained model should be saved
    models_dir = project_root / "models"

    # Define where figures should be saved
    figures_dir = project_root / "results" / "figures"

    # Define where metrics should be saved
    metrics_dir = project_root / "results" / "metrics"

    # Create the directories if they do not already exist
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load the raw combined dataset
    df = load_data()

    # Preprocess the dataset and split into train/test sets
    X_train, X_test, y_train, y_test, processed_df = preprocess_data(df)

    # Create a Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=100,   # number of trees in the forest
        random_state=42,    # seed for reproducibility
        n_jobs=-1           # use all available CPU cores
    )

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Generate predictions on the test data
    predictions = model.predict(X_test)

    # Calculate Mean Absolute Error between true and predicted prices
    mae = mean_absolute_error(y_test, predictions)

    # Print the model performance to the terminal
    print(f"Mean Absolute Error: {mae:.2f}")

    # Save the trained model as a .pkl file
    joblib.dump(model, models_dir / "random_forest_model.pkl")

    # Save the list of feature column names so prediction input can match training
    joblib.dump(list(X_train.columns), models_dir / "model_features.pkl")

    # Save MAE to a text file for documentation and GitHub screenshots
    with open(metrics_dir / "mae.txt", "w", encoding="utf-8") as f:
        f.write(f"Mean Absolute Error: {mae:.2f}\n")

    # Extract feature importances from the trained random forest model
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    # Sort features from most important to least important
    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    ).head(15)

    # Create a horizontal bar chart of the top 15 most important features
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - Random Forest")

    # Adjust layout so labels fit neatly
    plt.tight_layout()

    # Save the feature importance chart to disk
    plt.savefig(figures_dir / "feature_importance.png")

    # Close the figure to free memory
    plt.close()

    # Confirm model and outputs have been saved
    print("Training complete.")
    print(f"Model saved to: {models_dir / 'random_forest_model.pkl'}")
    print(f"Metrics saved to: {metrics_dir / 'mae.txt'}")
    print(f"Figure saved to: {figures_dir / 'feature_importance.png'}")


# Run the training pipeline if this file is executed directly
if __name__ == "__main__":
    main()
