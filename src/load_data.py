import pandas as pd
from pathlib import Path


def load_data(use_sample: bool = False):
    project_root = Path(__file__).resolve().parent.parent

    if use_sample:
        data_path = project_root / "data" / "sample" / "sample_flights.csv"
        return pd.read_csv(data_path)

    raw_path = project_root / "data" / "raw"

    business = pd.read_csv(raw_path / "business.csv")
    economy = pd.read_csv(raw_path / "economy.csv")
    clean = pd.read_csv(raw_path / "clean_dataset.csv")

    df = pd.concat([business, economy, clean], ignore_index=True)
    return df


if __name__ == "__main__":
    df = load_data(use_sample=True)
    print(df.shape)
    print(df.head())
