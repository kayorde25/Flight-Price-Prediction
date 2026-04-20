import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from load_data import load_raw_business_economy
from preprocess import preprocess_for_price_model


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    figures_dir = project_root / "results" / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_business_economy()
    X, y, _ = preprocess_for_price_model(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("Random Forest MAE:", mae)
    print("Random Forest RMSE:", rmse)
    print("Random Forest R2:", r2)

    joblib.dump(model, models_dir / "best_model.pkl")
    joblib.dump(list(X_train.columns), models_dir / "model_features.pkl")

    print(f"Saved model to: {models_dir / 'best_model.pkl'}")
    print(f"Saved feature list to: {models_dir / 'model_features.pkl'}")


if __name__ == "__main__":
    main()
