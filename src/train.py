"""
train.py

Advanced training pipeline for the TripSmooth Flight Price Prediction project.

This script:
1. Loads and preprocesses the dataset
2. Trains a Random Forest baseline model
3. Trains an XGBoost model
4. Tunes XGBoost hyperparameters
5. Compares model performance
6. Saves the best model
7. Exports charts and metrics for GitHub
"""

# Import pathlib for safe project-relative file paths
from pathlib import Path

# Import joblib to save trained models and metadata
import joblib

# Import matplotlib for plots
import matplotlib.pyplot as plt

# Import pandas for tables and metrics output
import pandas as pd

# Import NumPy for numeric operations
import numpy as np

# Import baseline and advanced regressors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import model selection utility for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Import regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import local project helpers
from load_data import load_data
from preprocess import preprocess_data


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model on the test set.

    Args:
        model:
            Trained model object.
        X_test:
            Test feature matrix.
        y_test:
            True target values for the test set.
        model_name:
            Name of the model being evaluated.

    Returns:
        dict:
            Dictionary containing regression performance metrics.
    """

    # Generate predictions on the test set
    predictions = model.predict(X_test)

    # Compute Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)

    # Compute Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Compute R-squared score
    r2 = r2_score(y_test, predictions)

    # Return metrics in a structured dictionary
    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def save_feature_importance_plot(model, feature_names, output_path: Path, title: str):
    """
    Save a feature importance plot for models that expose feature_importances_.

    Args:
        model:
            Trained model with feature_importances_ attribute.
        feature_names:
            List of feature names.
        output_path:
            Path where the plot image will be saved.
        title:
            Title for the plot.
    """

    # Build a DataFrame of features and their importance values
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    # Sort descending and keep top 15
    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    ).head(15)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.close()


def save_actual_vs_predicted_plot(y_true, y_pred, output_path: Path, title: str):
    """
    Save a scatter plot of actual vs predicted values.

    Args:
        y_true:
            True target values.
        y_pred:
            Predicted target values.
        output_path:
            Path where the plot image will be saved.
        title:
            Title for the plot.
    """

    # Create figure
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)

    # Plot a reference line for perfect predictions
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])

    # Add labels
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()

    # Save chart
    plt.savefig(output_path)
    plt.close()


def main():
    """
    Main model training workflow.
    """

    # Define important project directories
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    figures_dir = project_root / "results" / "figures"
    metrics_dir = project_root / "results" / "metrics"

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dataset
    df = load_data()

    # Preprocess and split data
    X_train, X_test, y_train, y_test, processed_df = preprocess_data(df)

    # Save the feature list used for training
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, models_dir / "model_features.pkl")

    # ---------------------------------------------------------
    # 1. Train Random Forest baseline
    # ---------------------------------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate Random Forest
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Save feature importance plot for Random Forest
    save_feature_importance_plot(
        rf_model,
        feature_names,
        figures_dir / "rf_feature_importance.png",
        "Top 15 Feature Importances - Random Forest"
    )

    # Save actual vs predicted plot for Random Forest
    rf_predictions = rf_model.predict(X_test)
    save_actual_vs_predicted_plot(
        y_test,
        rf_predictions,
        figures_dir / "rf_actual_vs_predicted.png",
        "Actual vs Predicted Prices - Random Forest"
    )

    # ---------------------------------------------------------
    # 2. Train initial XGBoost model
    # ---------------------------------------------------------
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate initial XGBoost
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (initial)")

    # Save feature importance plot for initial XGBoost
    save_feature_importance_plot(
        xgb_model,
        feature_names,
        figures_dir / "xgb_feature_importance.png",
        "Top 15 Feature Importances - XGBoost"
    )

    # Save actual vs predicted plot for initial XGBoost
    xgb_predictions = xgb_model.predict(X_test)
    save_actual_vs_predicted_plot(
        y_test,
        xgb_predictions,
        figures_dir / "xgb_actual_vs_predicted.png",
        "Actual vs Predicted Prices - XGBoost"
    )

    # ---------------------------------------------------------
    # 3. Hyperparameter tuning for XGBoost
    # ---------------------------------------------------------
    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    }

    # Create tuning object
    tuning = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=42,
            objective="reg:squarederror"
        ),
        param_distributions=param_distributions,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Fit tuning search
    tuning.fit(X_train, y_train)

    # Extract best tuned model
    best_xgb_model = tuning.best_estimator_

    # Evaluate tuned XGBoost
    tuned_xgb_metrics = evaluate_model(best_xgb_model, X_test, y_test, "XGBoost (tuned)")

    # Save tuned XGBoost charts
    save_feature_importance_plot(
        best_xgb_model,
        feature_names,
        figures_dir / "xgb_tuned_feature_importance.png",
        "Top 15 Feature Importances - Tuned XGBoost"
    )

    tuned_xgb_predictions = best_xgb_model.predict(X_test)
    save_actual_vs_predicted_plot(
        y_test,
        tuned_xgb_predictions,
        figures_dir / "xgb_tuned_actual_vs_predicted.png",
        "Actual vs Predicted Prices - Tuned XGBoost"
    )

    # ---------------------------------------------------------
    # 4. Compare models
    # ---------------------------------------------------------
    results_df = pd.DataFrame([rf_metrics, xgb_metrics, tuned_xgb_metrics])

    # Sort by lowest MAE (best model first)
    results_df = results_df.sort_values(by="mae", ascending=True)

    # Save metrics table to CSV
    results_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    # Save best hyperparameters to text file
    with open(metrics_dir / "best_xgb_params.txt", "w", encoding="utf-8") as f:
        f.write("Best XGBoost Parameters:\n")
        for key, value in tuning.best_params_.items():
            f.write(f"{key}: {value}\n")

    # ---------------------------------------------------------
    # 5. Save best model
    # ---------------------------------------------------------
    best_model_name = results_df.iloc[0]["model"]

    if best_model_name == "Random Forest":
        best_model = rf_model
        best_model_filename = "best_model_random_forest.pkl"
    elif best_model_name == "XGBoost (initial)":
        best_model = xgb_model
        best_model_filename = "best_model_xgboost_initial.pkl"
    else:
        best_model = best_xgb_model
        best_model_filename = "best_model_xgboost_tuned.pkl"

    # Save the best model artifact
    joblib.dump(best_model, models_dir / best_model_filename)

    # Also save a standard best-model path for easier inference
    joblib.dump(best_model, models_dir / "best_model.pkl")

    # ---------------------------------------------------------
    # 6. Export model comparison bar chart
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["mae"])
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Model")
    plt.title("Model Comparison by MAE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_mae.png")
    plt.close()

    # Print final results
    print("\nModel comparison:")
    print(results_df)
    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to: {models_dir / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
