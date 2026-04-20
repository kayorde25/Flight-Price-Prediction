import pandas as pd

def preprocess(df):
    data = df.copy()

    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"])

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    data["departure_hour"] = pd.to_datetime(data["dep_time"], errors="coerce").dt.hour
    data["arrival_hour"] = pd.to_datetime(data["arr_time"], errors="coerce").dt.hour

    data = data.drop(columns=["date", "dep_time", "arr_time"])

    y = data["price"]
    X = data.drop(columns=["price"])

    X = pd.get_dummies(X, drop_first=True)

    return X, y
