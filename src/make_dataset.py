# src/make_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("data/raw/train.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load raw data
    df = pd.read_csv(RAW_PATH)

    # 2) Basic sanity checks (great to mention in README)
    assert "SalePrice" in df.columns, "Target column SalePrice missing"
    assert df.shape[0] > 1000, "Unexpectedly small dataset"

    # 3) Create modeling target variants
    df["LogSalePrice"] = np.log(df["SalePrice"])

    # 4) Example feature engineering (simple + interpretable)
    # Total square footage proxy (not perfect but useful)
    for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
        if col not in df.columns:
            df[col] = 0

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # 5) Missingness indicators (common real-world technique)
    # These can improve models because "missing" often means "none"
    df["HasGarage"] = df["GarageArea"].fillna(0).gt(0).astype(int)
    df["HasBasement"] = df["TotalBsmtSF"].fillna(0).gt(0).astype(int)

    # 6) Save processed dataset
    out_path = OUT_DIR / "train_processed.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}  rows={df.shape[0]} cols={df.shape[1]}")

if __name__ == "__main__":
    main()