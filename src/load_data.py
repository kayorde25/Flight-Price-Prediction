from pathlib import Path
import pandas as pd


def load_raw_business_economy():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    business = pd.read_csv(data_dir / "business.csv")
    economy = pd.read_csv(data_dir / "economy.csv")

    business.columns = (
        business.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    economy.columns = (
        economy.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    business["class"] = "Business"
    economy["class"] = "Economy"

    df = pd.concat([business, economy], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return df
