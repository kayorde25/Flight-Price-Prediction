import pandas as pd


def preprocess_for_price_model(df: pd.DataFrame):
    data = df.copy()

    data = data.drop_duplicates()

    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"])

    for col in data.columns:
        if data[col].dtype == "object":
            mode = data[col].mode()
            data[col] = data[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
        else:
            data[col] = data[col].fillna(
                data[col].median() if data[col].notna().sum() > 0 else 0
            )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    data["departure_hour"] = pd.to_datetime(data["dep_time"], errors="coerce").dt.hour
    data["arrival_hour"] = pd.to_datetime(data["arr_time"], errors="coerce").dt.hour

    data = data.drop(columns=["date", "dep_time", "arr_time"])

    if "time_taken" in data.columns and data["time_taken"].dtype == "object":
        duration_str = (
            data["time_taken"]
            .str.replace("h", "*60+", regex=False)
            .str.replace("m", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        data["time_taken"] = duration_str.apply(
            lambda x: eval(x[:-1] if isinstance(x, str) and x.endswith("+") else x)
            if isinstance(x, str) and len(x) > 0 else None
        )

    for col in ["ch_code", "num_code"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    y = data["price"]
    X = data.drop(columns=["price"])

    for col in ["airline", "from", "to", "stop", "class"]:
        if col in X.columns and X[col].dtype == "object":
            top_categories = X[col].value_counts().nlargest(10).index
            X[col] = X[col].where(X[col].isin(top_categories), "other")

    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, data
