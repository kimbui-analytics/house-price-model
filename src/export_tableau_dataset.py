# src/export_tableau_dataset.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

PROCESSED_PATH = Path("data/processed/train_processed.parquet")
OUT_PATH = Path("data/processed/tableau_model_dataset.csv")

def main():
    df = pd.read_parquet(PROCESSED_PATH)

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42
    )

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
    ])

    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_numeric)

    export_df = df.copy()
    export_df["PredictedPrice"] = preds
    export_df["Residual"] = export_df["SalePrice"] - export_df["PredictedPrice"]
    export_df["PercentError"] = export_df["Residual"] / export_df["SalePrice"]

    export_df.to_csv(OUT_PATH, index=False)

    print(f"Saved Tableau dataset to {OUT_PATH}")

if __name__ == "__main__":
    main()