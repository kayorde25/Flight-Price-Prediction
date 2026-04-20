from pathlib import Path
import pandas as pd

def load_data():
    data_dir = Path("data")

    business = pd.read_csv(data_dir / "business.csv")
    economy = pd.read_csv(data_dir / "economy.csv")

    business.columns = business.columns.str.lower().str.replace(" ", "_")
    economy.columns = economy.columns.str.lower().str.replace(" ", "_")

    business["class"] = "Business"
    economy["class"] = "Economy"

    df = pd.concat([business, economy], ignore_index=True)
    return df
