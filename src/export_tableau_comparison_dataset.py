# src/export_tableau_comparison_dataset.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

PROCESSED_PATH = Path("data/processed/train_processed.parquet")
OUT_PATH = Path("data/processed/tableau_model_comparison_dataset.csv")

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def build_model_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def main():
    df = pd.read_parquet(PROCESSED_PATH)

    # Base data
    y_actual = df["SalePrice"]
    y_log = np.log(y_actual)

    X = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number])

    # Train/test split
    X_train, X_test, y_train_actual, y_test_actual = train_test_split(
        X_numeric, y_actual, test_size=0.2, random_state=42
    )

    _, _, y_train_log, y_test_log = train_test_split(
        X_numeric, y_log, test_size=0.2, random_state=42
    )

    # Model definitions
    alphas = np.logspace(-3, 3, 50)

    ridge = build_model_pipeline(RidgeCV(alphas=alphas, cv=5))
    lasso = build_model_pipeline(LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42))

    # ---------------------------
    # Fit Actual Target Models
    # ---------------------------
    ridge.fit(X_train, y_train_actual)
    lasso.fit(X_train, y_train_actual)

    ridge_preds_actual = ridge.predict(X_numeric)
    lasso_preds_actual = lasso.predict(X_numeric)

    # ---------------------------
    # Fit Log Target Models
    # ---------------------------
    ridge.fit(X_train, y_train_log)
    lasso.fit(X_train, y_train_log)

    ridge_preds_log = np.exp(ridge.predict(X_numeric))
    lasso_preds_log = np.exp(lasso.predict(X_numeric))

    # ---------------------------
    # Build long-format export
    # ---------------------------
    export_rows = []

    for i in range(len(df)):
        for model_name, preds_actual, preds_log in [
            ("Ridge", ridge_preds_actual, ridge_preds_log),
            ("Lasso", lasso_preds_actual, lasso_preds_log)
        ]:
            # Actual Target Version
            pred_actual = preds_actual[i]
            residual_actual = df["SalePrice"].iloc[i] - pred_actual

            export_rows.append({
                "Model": model_name,
                "TargetType": "Actual",
                "SalePrice": df["SalePrice"].iloc[i],
                "PredictedPrice": pred_actual,
                "Residual": residual_actual,
                "PercentError": residual_actual / df["SalePrice"].iloc[i],
                "AbsolutePercentError": abs(residual_actual / df["SalePrice"].iloc[i]),
                "Neighborhood": df.get("Neighborhood", None).iloc[i] if "Neighborhood" in df else None,
                "OverallQual": df.get("OverallQual", None).iloc[i] if "OverallQual" in df else None,
                "TotalSF": df.get("TotalSF", None).iloc[i] if "TotalSF" in df else None,
                "HasGarage": df.get("HasGarage", None).iloc[i] if "HasGarage" in df else None
            })

            # Log Target Version
            pred_log = preds_log[i]
            residual_log = df["SalePrice"].iloc[i] - pred_log

            export_rows.append({
                "Model": model_name,
                "TargetType": "Log",
                "SalePrice": df["SalePrice"].iloc[i],
                "PredictedPrice": pred_log,
                "Residual": residual_log,
                "PercentError": residual_log / df["SalePrice"].iloc[i],
                "AbsolutePercentError": abs(residual_log / df["SalePrice"].iloc[i]),
                "Neighborhood": df.get("Neighborhood", None).iloc[i] if "Neighborhood" in df else None,
                "OverallQual": df.get("OverallQual", None).iloc[i] if "OverallQual" in df else None,
                "TotalSF": df.get("TotalSF", None).iloc[i] if "TotalSF" in df else None,
                "HasGarage": df.get("HasGarage", None).iloc[i] if "HasGarage" in df else None
            })

    export_df = pd.DataFrame(export_rows)

    export_df.to_csv(OUT_PATH, index=False)
    print(f"Saved comparison dataset to {OUT_PATH}")

if __name__ == "__main__":
    main()
    